from typing import Callable, Optional, Any
import dataclasses
import os
import pprint
from functools import partial
from copy import deepcopy

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
    JaxRNG, cross_entropy_loss, next_rng, mixup_cutmix, sync_state_across_devices
)
from .model import ViTClassifier, MaskedAutoencoder, extract_patches, M3AETrainState
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
    warmup_epochs=10,
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
    weight_decay=0.0,
    clip_gradient=1e9,
    mixup_alpha = 0.8,
    cutmix_alpha = 1.0,
    switch_prob = 0.5,
    label_smoothing = 0.1,
    layer_decay = 0.75,
    load_pretrained="",
    load_checkpoint="",
    train_data=ImageNetDataset.get_default_config(),
    val_data=ImageNetDataset.get_default_config(),
    mae=MaskedAutoencoder.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS


def create_train_step(model, mixup_cutmix_fn, xe_loss_fn):
    @partial(jax.pmap, axis_name="pmap", donate_argnums=0)
    def train_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)
        def loss_fn(params):
            augmented_image, augmented_label = mixup_cutmix_fn(image, label, rng)
            image_patches =  extract_patches(augmented_image, 16)
            logits = model.apply(
                params, image_patches, deterministic=False,
                rngs=rng_generator(keys=model.rng_keys())
            )
            loss = xe_loss_fn(logits, augmented_label)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(augmented_label, axis=-1))
            aux = dict(loss=loss, accuracy=accuracy)
            return loss, aux

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = jax.lax.pmean(grad_fn(state.params), axis_name="pmap")
        state = state.apply_gradients(grads=grads)
        return state, aux, rng_generator()

    @partial(jax.pmap, axis_name="pmap")
    def eval_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)

        image_patches= extract_patches(image, 16)
        logits = model.apply(
            state.params, image_patches, deterministic=True,
            rngs=JaxRNG(rng)(keys=model.rng_keys())
        )
        accuracy = jax.lax.pmean(
            jnp.mean(jnp.argmax(logits, axis=-1) == label),
            axis_name="pmap"
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

    model = ViTClassifier(
        base_model=MaskedAutoencoder(
            config_updates=FLAGS.mae,
            image_output_dim=image_output_dim
        ),
        num_classes=train_dataset.num_classes(),
        global_pool=FLAGS.global_pool,
    )

    params = model.init(
        next_rng(model.rng_keys()),
        jnp.zeros((2, 256, 768), dtype=jnp.float32),
        deterministic=False
    )
    params = flax.core.unfreeze(params)

    if FLAGS.load_pretrained != "":
        checkpoint_data = load_checkpoint(FLAGS.load_pretrained)
        checkpoint_params = checkpoint_data["state"].params["params"]
        params['params']['base_model'] = checkpoint_params

    def get_transform_label(param_info, layer_decay=.75):
        def get_learning_rate(lr_scale=1.0):
            return optax.warmup_cosine_decay_schedule(
                init_value=FLAGS.lr_init_value * lr_scale,
                peak_value=FLAGS.lr_peak_value * lr_scale,
                warmup_steps=FLAGS.lr_warmup_epochs * steps_per_epoch,
                decay_steps=total_steps,
                end_value=FLAGS.lr_end_value * lr_scale,
            )

        def get_weight_decay_mask(layer_params):
            no_decay_list = model.base_model.no_decay_list() + ['bias']

            decay_ids = set([
                i for key, (i, d) in param_info.items()
                if all([k not in no_decay_list for k in key]) and d != 1
            ])
            return [id(p) in decay_ids for p in layer_params]

        def get_optim(lr_scale):
            optim = optax.chain(
                optax.clip_by_global_norm(FLAGS.clip_gradient),
                optax.adamw(
                    learning_rate=get_learning_rate(lr_scale),
                    weight_decay=FLAGS.weight_decay,
                    b1=0.9, b2=0.999,
                    mask=get_weight_decay_mask,
                )
            )
            return optim

        num_layer = FLAGS.mae.depth + 1
        transforms = {i: get_optim(layer_decay ** (num_layer - i)) for i in range(num_layer + 1)}

        def name_to_layer_id(name):
            if len(name) > 3 and name[3].startswith('Block'):
                layer_id = int(name[3].split('_')[1]) + 1
            elif len(name) > 3 and name[1] == 'base_model' and name[3] in ['LayerNorm_0', 'patch_emb']:
                layer_id = 0
            elif len(name) <= 3 and name[1] == 'base_model' and name[2] in ['cls_token', 'patch_mask_emb']:
                layer_id = 0
            else:
                layer_id = num_layer
            return layer_id

        def get_param_labels(params):
            flattened_params = flax.traverse_util.flatten_dict(
                flax.core.frozen_dict.unfreeze(params)
            )
            return flax.traverse_util.unflatten_dict(
                {key: name_to_layer_id(key) for key, value in flattened_params.items()}
            )

        return transforms, get_param_labels

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(
            load_pickle(FLAGS.load_checkpoint), jax_devices
        )
        start_step = checkpoint_data["step"]
    else:
        param_info = {
            key: (id(value), jnp.ndim(value))
            for key, value in flax.traverse_util.flatten_dict(
                flax.core.frozen_dict.unfreeze(params)
            ).items()
        }
        transforms, get_param_labels = get_transform_label(param_info, layer_decay=FLAGS.layer_decay)
        tx = optax.multi_transform(transforms, get_param_labels)
        state = M3AETrainState.create(
            apply_fn=None, params=params, tx=tx,
            tokenizer_params=tokenizer_params
        )
        state = flax.jax_utils.replicate(state, jax_devices)
        start_step = 0

    mixup_cutmix_fn = partial(
        mixup_cutmix,
        num_classes=train_dataset.num_classes(),
        mixup_alpha=FLAGS.mixup_alpha,
        cutmix_alpha=FLAGS.cutmix_alpha,
        switch_prob=FLAGS.switch_prob,
        label_smoothing=FLAGS.label_smoothing
    )
    xe_loss_fn = partial(cross_entropy_loss, smoothing_factor = 0. if FLAGS.mixup_alpha > 0. else FLAGS.label_smoothing)
    train_step_fn, eval_step_fn = create_train_step(
        model, mixup_cutmix_fn, xe_loss_fn
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
