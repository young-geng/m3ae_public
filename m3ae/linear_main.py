from typing import Callable, Optional, Any
import dataclasses
import os
import pprint
from functools import partial

import absl.app
import absl.flags
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb

from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training import train_state, common_utils
from tqdm.auto import tqdm, trange

from .data import ImageNetDataset
from .jax_utils import (
    JaxRNG, cross_entropy_loss, next_rng, sync_state_across_devices
)
from .model import LinearCLS, MaskedAutoencoder, extract_patches
from .vqgan import get_image_tokenizer
from .utils import (
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    load_checkpoint,
    load_pickle,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=90,
    batch_size=2,
    discretized_image=False,
    image_tokenizer_type='maskgit',
    global_pool=False,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=10,
    save_model_freq=0,
    momentum=0.9,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.5e-4,
    lr_warmup_epochs=0,
    weight_decay=0.05,
    load_pretrained="",
    load_checkpoint="",
    train_data=ImageNetDataset.get_default_config(),
    val_data=ImageNetDataset.get_default_config(),
    mae=MaskedAutoencoder.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS


class LinearCLSTrainState(train_state.TrainState):
    tokenizer_params: Optional[flax.core.FrozenDict[str, Any]] = None
    backbone_params: Optional[flax.core.FrozenDict[str, Any]] = None
    batch_stats: Optional[flax.core.FrozenDict[str, Any]] = None


def create_train_step(model, backbone, learning_rate):
    @partial(jax.pmap, axis_name="pmap", donate_argnums=[0])
    def train_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)

        def loss_fn(params):
            image_patches = extract_patches(image, 16)
            representation = backbone.apply(
                state.backbone_params,
                image_patches,
                deterministic=False,
                rngs=rng_generator(keys=backbone.rng_keys()),
                method=backbone.forward_representation,
            )

            logits, new_model_state = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                jax.lax.stop_gradient(representation),
                train=True,
                mutable=["batch_stats"],
            )
            loss = cross_entropy_loss(logits, label)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == label)
            aux = dict(loss=loss, accuracy=accuracy)
            return loss, (aux, new_model_state["batch_stats"])

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (aux, batch_stats)), grads = jax.lax.pmean(
            grad_fn(state.params), axis_name="pmap",
        )
        aux["learning_rate"] = learning_rate(state.step)
        state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        return state, aux, rng_generator()

    @partial(jax.pmap, axis_name="pmap")
    def eval_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)

        image_patches = extract_patches(image, 16)
        representation = backbone.apply(
            state.backbone_params,
            image_patches,
            deterministic=True,
            rngs=rng_generator(keys=backbone.rng_keys()),
            method=backbone.forward_representation,
        )

        logits = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            jax.lax.stop_gradient(representation),
            train=False,
            mutable=False
        )
        accuracy = jax.lax.pmean(
            jnp.mean(jnp.argmax(logits, axis=-1) == label),
            axis_name='pmap'
        )
        aux = dict(accuracy=accuracy)
        return aux, rng_generator()

    return train_step_fn, eval_step_fn


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    assert FLAGS.batch_size % jax_process_count == 0
    variant["process_batch_size"] = process_batch_size = (
        FLAGS.batch_size // jax_process_count
    )
    variant["device_batch_size"] = process_batch_size // jax.local_device_count()
    lr_scale = FLAGS.batch_size / 256
    variant["effective_lr"] = FLAGS.lr_peak_value * lr_scale
    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert process_batch_size % n_devices == 0

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    train_dataset = ImageNetDataset(FLAGS.train_data, jax_process_index / jax_process_count)
    val_dataset = ImageNetDataset(FLAGS.val_data, jax_process_index / jax_process_count)

    steps_per_epoch = int(len(train_dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs
    val_steps = int(len(val_dataset) / FLAGS.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=process_batch_size,
        shuffle=False,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
        drop_last=True,
    )

    if FLAGS.discretized_image:
        tokenizer_params, encode_image, decode_image, image_vocab_size = get_image_tokenizer(
            FLAGS.image_tokenizer_type
        )
        image_output_dim = image_vocab_size
    else:
        tokenizer_params, encode_image, decode_image, image_vocab_size = (
            None, None, None, -1
        )
        image_output_dim = 768

    backbone = MaskedAutoencoder(
        config_updates=FLAGS.mae,
        image_output_dim=image_output_dim
    )

    backbone_params = backbone.init(
        next_rng(keys=backbone.rng_keys()),
        jnp.zeros((2, 256, 768), dtype=jnp.float32),
        deterministic=False
    )

    if FLAGS.load_pretrained != "":
        checkpoint_data = load_checkpoint(FLAGS.load_pretrained)
        checkpoint_params = checkpoint_data["state"].params["params"]
        checkpoint_params = flax.core.unfreeze(checkpoint_params)
        backbone_params = flax.core.unfreeze(backbone_params["params"])

        for key in backbone_params.keys():
            assert key in checkpoint_params.keys(), f"pretrained model miss key={key}"
            backbone_params[key] = checkpoint_params[key]
        backbone_params = flax.core.freeze({"params": backbone_params})

    model = LinearCLS(train_dataset.num_classes(), FLAGS.global_pool)

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=FLAGS.lr_init_value * lr_scale,
        peak_value=FLAGS.lr_peak_value * lr_scale,
        warmup_steps=FLAGS.lr_warmup_epochs * steps_per_epoch,
        decay_steps=total_steps,
        end_value=FLAGS.lr_end_value * lr_scale,
    )

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(
            load_pickle(FLAGS.load_checkpoint), jax_devices
        )
        start_step = checkpoint_data["step"]

        del tokenizer_params, backbone_params
    else:
        emb_dim = MaskedAutoencoder.get_default_config(FLAGS.mae).emb_dim
        dummy_input = jnp.zeros((2, 257, emb_dim), dtype=jnp.float32)
        variables = model.init(next_rng(), dummy_input)
        params, batch_stats = variables["params"], variables["batch_stats"]
        state = LinearCLSTrainState.create(
            params=params,
            tokenizer_params=tokenizer_params,
            backbone_params=backbone_params,
            batch_stats=batch_stats,
            apply_fn=None,
            tx=optax.lars(
                learning_rate=learning_rate,
                weight_decay=FLAGS.weight_decay,
                momentum=FLAGS.momentum,
            )
        )
        state = flax.jax_utils.replicate(state, jax_devices)
        start_step = 0

        del params, tokenizer_params, backbone_params

    train_step_fn, eval_step_fn = create_train_step(
        model, backbone, learning_rate
    )
    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    def generate_batch(iterator):
        while True:
            for batch in iterator:
                imgs = batch[0].numpy()
                imgs = imgs.reshape(n_devices, -1, *imgs.shape[1:])
                labels = batch[1].numpy()
                labels = labels.reshape(n_devices, -1, *labels.shape[1:])
                yield tuple([imgs, labels])

    train_iterator = prefetch_to_device(generate_batch(train_loader), 2, jax_devices)
    val_iterator = prefetch_to_device(generate_batch(val_loader), 2, jax_devices)

    best_val_acc = 0.0
    step_counter = trange(start_step, total_steps, ncols=0)

    for step, (image, label) in zip(step_counter, train_iterator):
        epoch = step // steps_per_epoch
        if step % steps_per_epoch == 0:
            train_metrics = []

        image = image.astype(jnp.float32)
        label = label.astype(jnp.int32)

        state, metrics, sharded_rng = train_step_fn(
            state, sharded_rng, image, label
        )
        train_metrics.append(metrics)

        if step % FLAGS.log_freq == 0:
            log_metrics = common_utils.get_metrics(train_metrics)
            log_metrics = {
                f"train_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(flax.jax_utils.unreplicate(state)),
                "best_val_acc": best_val_acc,
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")

        if step % steps_per_epoch == 0:
            val_metrics = []
            for _, (image, label) in zip(trange(val_steps, ncols=0), val_iterator):
                image = image.astype(jnp.float32)
                label = label.astype(jnp.int32)

                metrics, sharded_rng = eval_step_fn(
                    state, sharded_rng, image, label
                )
                val_metrics.append(metrics)

            log_metrics = common_utils.get_metrics(val_metrics)
            log_metrics = {
                f"val_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            accuracy = common_utils.get_metrics(val_metrics)["accuracy"].mean()

            if accuracy > best_val_acc:
                best_val_acc = accuracy

                if FLAGS.save_model_freq > 0:
                    save_data = {
                        "epoch": epoch,
                        "step": step,
                        "variant": variant,
                        "state": jax.device_get(flax.jax_utils.unreplicate(state)),
                        "best_val_acc": best_val_acc,
                    }
                    if jax_process_index == 0:
                        logger.save_pickle(save_data, "best_model.pkl")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    absl.app.run(main)
