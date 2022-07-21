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
from tqdm.auto import tqdm, trange

from .data import ImageTextDataset, TextDataset
from .jax_utils import (
    JaxRNG, get_metrics, next_rng, accumulated_gradient,
    sync_state_across_devices
)
from .model import (
    MaskedMultimodalAutoencoder, extract_patches,
    merge_patches, cross_entropy_loss_and_accuracy,
    patch_mse_loss, M3AETrainState, mask_intersection, mask_not,
    mask_select, all_mask
)
from .utils import (
    WandBLogger, define_flags_with_default, get_user_flags,
    image_float2int, load_pickle, set_random_seed, create_log_images
)
from .vqgan import get_image_tokenizer


FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=200,
    batch_size=2,
    accumulate_grad_steps=1,
    discretized_image=False,
    image_tokenizer_type='maskgit',
    image_all_token_loss=False,
    text_all_token_loss=False,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=50,
    plot_freq=1000,
    save_model_freq=0,
    image_loss_weight=1.0,
    text_loss_weight=0.1,
    unpaired_text_loss_weight=0.0,
    clip_gradient=1e9,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.5e-4,
    lr_warmup_epochs=0,
    weight_decay=0.05,
    load_checkpoint="",
    m3ae=MaskedMultimodalAutoencoder.get_default_config(),
    data=ImageTextDataset.get_default_config(),
    unpaired_text_data=TextDataset.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=True,
)
FLAGS = absl.flags.FLAGS


def create_train_step(model, learning_rate, encode_image=None, decode_image=None):
    @partial(jax.pmap, axis_name="pmap", donate_argnums=(0, 5))
    def train_step_fn(state, rng, accumulated_grads, accumulated_steps, batch):
        rng_generator = JaxRNG(rng)
        image = batch['image']
        text = batch['text']
        text_padding_mask = batch['text_padding_mask']

        def loss_fn(params):
            image_patches = extract_patches(image, 16)
            if FLAGS.discretized_image:
                encoded_image = encode_image(state.tokenizer_params, image)

            image_output, text_output, image_mask, text_mask = model.apply(
                params,
                image_patches,
                text,
                text_padding_mask,
                deterministic=False,
                rngs=rng_generator(keys=model.rng_keys()),
            )

            if FLAGS.discretized_image:
                image_loss, image_accuracy = cross_entropy_loss_and_accuracy(
                    image_output, encoded_image,
                    None if FLAGS.image_all_token_loss else image_mask
                )
            else:
                image_loss = patch_mse_loss(
                    image_output, image_patches,
                    None if FLAGS.image_all_token_loss else image_mask
                )
                image_accuracy = 0.0

            text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
                text_output, text,
                mask_intersection(
                    all_mask(text) if FLAGS.text_all_token_loss else text_mask,
                    mask_not(text_padding_mask)
                )
            )

            loss = (
                FLAGS.image_loss_weight * image_loss
                + FLAGS.text_loss_weight * text_loss
            )

            average_text_length = jnp.mean(jnp.sum(mask_not(text_padding_mask), axis=-1))

            if FLAGS.unpaired_text_loss_weight > 0.0:
                unpaired_text = batch['unpaired_text']
                unpaired_text_padding_mask = batch['unpaired_text_padding_mask']
                _, unpaired_text_output, _, unpaired_text_mask = model.apply(
                    params,
                    None,
                    unpaired_text,
                    unpaired_text_padding_mask,
                    deterministic=False,
                    rngs=rng_generator(keys=model.rng_keys()),
                )
                unpaired_text_loss, unpaired_text_accuracy = cross_entropy_loss_and_accuracy(
                    unpaired_text_output, unpaired_text,
                    mask_intersection(
                        all_mask(unpaired_text) if FLAGS.text_all_token_loss else unpaired_text_mask,
                        mask_not(unpaired_text_padding_mask)
                    )
                )
                loss = loss + FLAGS.unpaired_text_loss_weight * unpaired_text_loss
                average_unpaired_text_lenght = jnp.mean(
                    jnp.sum(mask_not(unpaired_text_padding_mask), axis=-1)
                )

            aux = dict(
                image_loss=image_loss,
                text_loss=text_loss,
                loss=loss,
                image_accuracy=image_accuracy,
                text_accuracy=text_accuracy,
                text_token_ratio=jnp.mean(
                    jnp.sum((1.0 - text_padding_mask), axis=-1) / text_mask.shape[-1]
                ),
                average_text_length=average_text_length,
            )
            if FLAGS.unpaired_text_loss_weight > 0.0:
                aux['unpaired_text_loss'] = unpaired_text_loss
                aux['unpaired_text_accuracy'] = unpaired_text_accuracy
                aux['average_unpaired_text_lenght'] = average_unpaired_text_lenght
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
    def patch_predict_fn(state, rng, batch):
        rng_generator = JaxRNG(rng)
        image = batch['image']
        text = batch['text']
        text_padding_mask = batch['text_padding_mask']

        image_patches = extract_patches(image, 16)
        if FLAGS.discretized_image:
            encoded_image = encode_image(state.tokenizer_params, image)

        image_output, text_output, image_mask, text_mask = model.apply(
            state.params,
            image_patches,
            text,
            text_padding_mask,
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

    dataset = ImageTextDataset(FLAGS.data, jax_process_index / jax_process_count)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
    )

    if FLAGS.unpaired_text_loss_weight > 0.0:
        unpaired_text_dataset = TextDataset(FLAGS.unpaired_text_data, jax_process_index / jax_process_count)
        unpaired_text_dataloader = torch.utils.data.DataLoader(
            unpaired_text_dataset,
            batch_size=process_batch_size,
            shuffle=FLAGS.dataloader_shuffle,
            drop_last=True,
            num_workers=FLAGS.dataloader_n_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    steps_per_epoch = int(len(dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs

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

    model = MaskedMultimodalAutoencoder(
        config_updates=FLAGS.m3ae,
        text_vocab_size=dataset.vocab_size,
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
        text = jnp.zeros((2, dataset.config.tokenizer_max_length), dtype=jnp.int32)
        text_padding_mask = jnp.zeros((2, dataset.config.tokenizer_max_length))
        rngs = next_rng(keys=model.rng_keys())
        params = model.init(
            rngs, image, text, text_padding_mask, deterministic=False
        )

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

    def generate_batch():
        def infinite_iterator(iterator):
            while True:
                for batch in iterator:
                    yield tuple(
                        x.numpy().reshape(
                            n_devices, -1, *x.shape[1:]
                        ) for x in batch
                    )

        paired_iterator = infinite_iterator(dataloader)

        if FLAGS.unpaired_text_loss_weight > 0.0:
            unpaired_text_iterator = infinite_iterator(unpaired_text_dataloader)

        while True:
            batch = {}
            image, text, text_padding_mask = next(paired_iterator)
            batch['image'] = image.astype(np.float32)
            batch['text'] = text.astype(np.int32)
            batch['text_padding_mask'] = text_padding_mask.astype(np.float32)

            if FLAGS.unpaired_text_loss_weight > 0.0:
                unpaired_text, unpaired_text_padding_mask = next(unpaired_text_iterator)
                batch['unpaired_text'] = unpaired_text.astype(np.int32)
                batch['unpaired_text_padding_mask'] = unpaired_text_padding_mask.astype(np.float32)

            yield batch

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

    data_iterator = prefetch_to_device(generate_batch(), 2, jax_devices)
    step_counter = trange(start_step, total_steps, ncols=0)

    for step, batch in zip(step_counter, data_iterator):
        epoch = int(step * jax_process_count / len(dataloader))
        state, metrics, sharded_rng, accumulated_grads, accumulated_steps = train_step_fn(
            state, sharded_rng, accumulated_grads, accumulated_steps, batch
        )
        if step % FLAGS.log_freq == 0:
            log_metrics = {"step": step, "epoch": epoch}
            log_metrics.update(get_metrics(metrics, unreplicate=True))
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.plot_freq > 0 and step % FLAGS.plot_freq == 0:
            log_image = create_log_images(
                jax.device_get(patch_predict_fn(state, sharded_rng, batch)),
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
