import dataclasses
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
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

from .data import ImageTextDataset, ImageNetDataset
from .jax_utils import (
    JaxRNG, get_metrics, next_rng, accumulated_gradient,
    sync_state_across_devices
)
from .model import (
    MaskedAutoencoder, extract_patches,
    merge_patches, cross_entropy_loss_and_accuracy,
    patch_mse_loss, M3AETrainState, mask_intersection, mask_not,
    mask_select
)
from .utils import (
    WandBLogger, define_flags_with_default, get_user_flags,
    image_float2int, load_pickle, set_random_seed, create_log_images
)
from .vqgan import get_image_tokenizer

FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=200,
    batch_size=0,
    accumulate_grad_steps=1,
    discretized_image=False,
    image_tokenizer_type='maskgit',
    image_all_token_loss=False,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=50,
    plot_freq=1000,
    save_model_freq=0,
    clip_gradient=1e9,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.5e-4,
    lr_warmup_epochs=0,
    weight_decay=0.05,
    load_checkpoint="",
    dataset="cc12m",
    mae=MaskedAutoencoder.get_default_config(),
    cc12m_data=ImageTextDataset.get_default_config(),
    imagenet_data=ImageNetDataset.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS


def create_train_step(model, learning_rate, encode_image=None, decode_image=None):
    @partial(jax.pmap, axis_name="pmap", donate_argnums=0)
    def train_step_fn(state, rng, accumulated_grads, accumulated_steps, image):
        rng_generator = JaxRNG(rng)

        def loss_fn(params):
            image_patches = extract_patches(image, 16)
            if FLAGS.discretized_image:
                encoded_image = encode_image(state.tokenizer_params, image)

            image_output, image_mask, *_ = model.apply(
                params,
                image_patches,
                deterministic=False,
                rngs=rng_generator(keys=model.rng_keys()),
            )
            if FLAGS.discretized_image:
                loss, image_accuracy = cross_entropy_loss_and_accuracy(
                    image_output, encoded_image,
                    None if FLAGS.image_all_token_loss else image_mask
                )
            else:
                loss = patch_mse_loss(
                    image_output, image_patches,
                    None if FLAGS.image_all_token_loss else image_mask
                )
                image_accuracy = 0.0

            aux = dict(
                loss=loss,
                image_accuracy=image_accuracy,
            )
            return loss, aux


        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        loss, aux = jax.lax.pmean((loss, aux), axis_name="pmap")
        aux["train_state_step"] = state.step
        aux["learning_rate"] = learning_rate(state.step)

        if FLAGS.accumulate_grad_steps > 1:
            state, accumulated_grads, accumulated_steps = accumulated_gradient(
                state, accumulated_grads, accumulated_steps, grads,
                FLAGS.accumulate_grad_steps,
                lambda s, g: s.apply_gradients(grads=jax.lax.pmean(g, axis_name="pmap"))
            )
        else:
            state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="pmap"))
        return state, aux, rng_generator(), accumulated_grads, accumulated_steps

    @partial(jax.pmap, axis_name="pmap")
    def patch_predict_fn(state, rng, image):
        rng_generator = JaxRNG(rng)

        image_patches = extract_patches(image, 16)
        if FLAGS.discretized_image:
            encoded_image = encode_image(state.tokenizer_params, image)

        image_output, image_mask, *_ = model.apply(
            state.params,
            image_patches,
            deterministic=True,
            rngs=rng_generator(keys=model.rng_keys()),
        )

        if FLAGS.discretized_image:
            image_output = jnp.argmax(image_output, -1)
            predicted_image = decode_image(state.tokenizer_params, image_output)
            predicted_image_combined = decode_image(
                state.tokenizer_params,
                mask_select(image_mask, encoded_image, image_output)
            )
        else:
            predicted_image = merge_patches(image_output, 16)
            predicted_image_combined = merge_patches(
                mask_select(image_mask, image_patches, image_output), 16
            )

        return image, predicted_image, predicted_image_combined

    return train_step_fn, patch_predict_fn


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

    if FLAGS.dataset == "cc12m":
        FLAGS.cc12m_data.image_only = True
        dataset = ImageTextDataset(FLAGS.cc12m_data, jax_process_index / jax_process_count)
    elif FLAGS.dataset == "imagenet":
        FLAGS.imagenet_data.image_only = True
        dataset = ImageNetDataset(FLAGS.imagenet_data, jax_process_index / jax_process_count)
    else:
        raise ValueError("Unsupported dataset!")

    steps_per_epoch = int(len(dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
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

    model = MaskedAutoencoder(
        config_updates=FLAGS.mae,
        image_output_dim=image_output_dim
    )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=FLAGS.lr_init_value * lr_scale,
        peak_value=FLAGS.lr_peak_value * lr_scale,
        warmup_steps=FLAGS.lr_warmup_epochs * steps_per_epoch // FLAGS.accumulate_grad_steps,
        decay_steps=total_steps // FLAGS.accumulate_grad_steps,
        end_value=FLAGS.lr_end_value * lr_scale,
    )

    def get_weight_decay_mask(params):
        flattened_params = flax.traverse_util.flatten_dict(
            flax.core.frozen_dict.unfreeze(params)
        )

        def decay(key):
            return all([k not in model.no_decay_list() for k in key])

        return flax.traverse_util.unflatten_dict(
            {key: decay(key) for key in flattened_params.keys()}
        )


    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(checkpoint_data["state"], jax_devices)
        start_step = checkpoint_data["step"]
        del tokenizer_params
    else:
        image = jnp.zeros((2, 256, 768), dtype=jnp.float32)
        rngs = next_rng(keys=model.rng_keys())
        params = model.init(rngs, image, deterministic=False)

        state = flax.jax_utils.replicate(
            M3AETrainState.create(
                params=flax.core.frozen_dict.unfreeze(params),
                tokenizer_params=tokenizer_params,
                apply_fn=None,
                tx=optax.chain(
                    optax.clip_by_global_norm(FLAGS.clip_gradient),
                    optax.adamw(
                        learning_rate=learning_rate, weight_decay=FLAGS.weight_decay,
                        b1=0.9, b2=0.95, mask=get_weight_decay_mask,
                    ),
                ),
            ),
            jax_devices,
        )
        start_step = 0
        del params, tokenizer_params

    train_step_fn, patch_predict_fn = create_train_step(
        model, learning_rate, encode_image, decode_image
    )

    def generate_batch(iterator):
        while True:
            for images in iterator:
                yield images.numpy().reshape(n_devices, -1, *images.shape[1:])

    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    if FLAGS.accumulate_grad_steps > 1:
        accumulated_grads = flax.jax_utils.replicate(
            jax.tree_map(jnp.zeros_like, flax.jax_utils.unreplicate(state).params),
            jax_devices
        )
        accumulated_steps = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)
    else:
        accumulated_grads = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)
        accumulated_steps = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)

    data_iterator = prefetch_to_device(generate_batch(dataloader), 2, jax_devices)
    step_counter = trange(start_step, total_steps, ncols=0)

    step = 0
    for step, image in zip(step_counter, data_iterator):
        epoch = int(step * jax_process_count / len(dataloader))
        image = image.astype(jnp.float32)
        state, metrics, sharded_rng, accumulated_grads, accumulated_steps = train_step_fn(
            state, sharded_rng, accumulated_grads, accumulated_steps, image
        )
        if step % FLAGS.log_freq == 0:
            log_metrics = {"step": step, "epoch": epoch}
            log_metrics.update(get_metrics(metrics, unreplicate=True))
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.plot_freq > 0 and step % FLAGS.plot_freq == 0:
            log_image = create_log_images(
                jax.device_get(patch_predict_fn(state, sharded_rng, image)),
                mean=dataset.image_mean, std=dataset.image_std
            )
            if jax_process_index == 0:
                logger.log({"image_prediction": wandb.Image(log_image)})

        if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(flax.jax_utils.unreplicate(state)),
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")

    if FLAGS.save_model_freq > 0:
        save_data = {
            "step": step,
            "epoch": epoch,
            "variant": variant,
            "state": jax.device_get(flax.jax_utils.unreplicate(state)),
        }
        if jax_process_index == 0:
            logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    absl.app.run(main)
