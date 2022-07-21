# This file is taken from https://github.com/google-research/maskgit with modifications.
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import io
import math
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from PIL import Image, ImageFilter
import requests
import os

from vqgan_jax.modeling_flax_vqgan import VQModel


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
    """Avg pooling as done by TF (Flax layer gives different results).
    To be specific, Flax includes padding cells when taking the average,
    while TF does not.
    Args:
      x: Input tensor
      window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
        2-dim tuple one gets 2d pooling.
      strides: Must have the same dimension as the window_shape.
      padding: Either 'SAME' or 'VALID' to indicate pooling method.
    Returns:
      pooled: Tensor after applying pooling.
    """
    pool_sum = jax.lax.reduce_window(
        x, 0.0, jax.lax.add, (1,) + window_shape + (1,), (1,) + strides + (1,), padding
    )
    pool_denom = jax.lax.reduce_window(
        jnp.ones_like(x),
        0.0,
        jax.lax.add,
        (1,) + window_shape + (1,),
        (1,) + strides + (1,),
        padding,
    )
    return pool_sum / pool_denom


def upsample(x, factor=2):
    n, h, w, c = x.shape
    x = jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")
    return x


def dsample(x):
    return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding="same")


def get_norm_layer(train, dtype, norm_type="BN"):
    """Normalization layer."""
    if norm_type == "BN":
        norm_fn = functools.partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=None,
            axis_index_groups=None,
            dtype=jnp.float32,
        )
    elif norm_type == "LN":
        norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
    elif norm_type == "GN":
        norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
    else:
        raise NotImplementedError
    return norm_fn


def squared_euclidean_distance(
    a: jnp.ndarray, b: jnp.ndarray, b2: jnp.ndarray = None, precision: Any = None
) -> jnp.ndarray:
    """Computes the pairwise squared Euclidean distance.
    Args:
      a: float32: (n, d): An array of points.
      b: float32: (m, d): An array of points.
      b2: float32: (d, m): b square transpose.
      precision: use DEFAULT precision by default
    Returns:
      d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
      a[i] and b[j].
    """
    if b2 is None:
        b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
    a2 = jnp.sum(a**2, axis=1, keepdims=True)
    ab = jnp.matmul(a, b.T, precision=precision)
    d = a2 - 2 * ab + b2
    return d


def entropy_loss_fn(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = jnp.argmax(flat_affinity, axis=-1)
        onehots = jax.nn.one_hot(
            codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype
        )
        onehots = probs - jax.lax.stop_gradient(probs - onehots)
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = jnp.mean(target_probs, axis=0)
    avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
    sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class ResBlock(nn.Module):
    """Basic Residual Block."""

    filters: int
    norm_fn: Any
    conv_fn: Any
    dtype: int = jnp.float32
    activation_fn: Any = nn.relu
    use_conv_shortcut: bool = False

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            if self.use_conv_shortcut:
                residual = self.conv_fn(
                    self.filters, kernel_size=(3, 3), use_bias=False
                )(x)
            else:
                residual = self.conv_fn(
                    self.filters, kernel_size=(1, 1), use_bias=False
                )(x)
        return x + residual


class Encoder(nn.Module):
    """Encoder Blocks."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32

    def setup(self):
        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.embedding_dim = self.config.vqvae.embedding_dim
        self.conv_downsample = self.config.vqvae.conv_downsample
        self.norm_type = self.config.vqvae.norm_type
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = nn.relu
        elif self.config.vqvae.activation_fn == "swish":
            self.activation_fn = nn.swish
        else:
            raise NotImplementedError

    @nn.compact
    def __call__(self, x):
        conv_fn = nn.Conv
        norm_fn = get_norm_layer(
            train=self.train, dtype=self.dtype, norm_type=self.norm_type
        )
        block_args = dict(
            norm_fn=norm_fn,
            conv_fn=conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )
        x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i < num_blocks - 1:
                if self.conv_downsample:
                    x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
                else:
                    x = dsample(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""

    config: ml_collections.ConfigDict
    train: bool
    output_dim: int = 3
    dtype: Any = jnp.float32

    def setup(self):
        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.norm_type = self.config.vqvae.norm_type
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = nn.relu
        elif self.config.vqvae.activation_fn == "swish":
            self.activation_fn = nn.swish
        else:
            raise NotImplementedError

    @nn.compact
    def __call__(self, x):
        conv_fn = nn.Conv
        norm_fn = get_norm_layer(
            train=self.train, dtype=self.dtype, norm_type=self.norm_type
        )
        block_args = dict(
            norm_fn=norm_fn,
            conv_fn=conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )
        num_blocks = len(self.channel_multipliers)
        filters = self.filters * self.channel_multipliers[-1]
        x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i > 0:
                x = upsample(x, 2)
                x = conv_fn(filters, kernel_size=(3, 3))(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
        return x


class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32

    @nn.compact
    def __call__(self, x, **kwargs):
        codebook_size = self.config.vqvae.codebook_size
        codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            (codebook_size, x.shape[-1]),
        )
        codebook = jnp.asarray(codebook, dtype=self.dtype)
        distances = jnp.reshape(
            squared_euclidean_distance(jnp.reshape(x, (-1, x.shape[-1])), codebook),
            x.shape[:-1] + (codebook_size,),
        )
        encoding_indices = jnp.argmin(distances, axis=-1)
        encodings = jax.nn.one_hot(encoding_indices, codebook_size, dtype=self.dtype)
        quantized = self.quantize(encodings)
        result_dict = dict()
        if self.train:
            e_latent_loss = (
                jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
                * self.config.vqvae.commitment_cost
            )
            q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
            entropy_loss = 0.0
            if self.config.vqvae.entropy_loss_ratio != 0:
                entropy_loss = (
                    entropy_loss_fn(
                        -distances,
                        loss_type=self.config.vqvae.entropy_loss_type,
                        temperature=self.config.vqvae.entropy_temperature,
                    )
                    * self.config.vqvae.entropy_loss_ratio
                )
            e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
            q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
            entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
            loss = e_latent_loss + q_latent_loss + entropy_loss
            result_dict = dict(
                quantizer_loss=loss,
                e_latent_loss=e_latent_loss,
                q_latent_loss=q_latent_loss,
                entropy_loss=entropy_loss,
            )
            quantized = x + jax.lax.stop_gradient(quantized - x)

        result_dict.update(
            {
                "encodings": encodings,
                "encoding_indices": encoding_indices,
                "raw": x,
            }
        )
        return quantized, result_dict

    def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
        codebook = jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)
        return jnp.dot(z, codebook)

    def get_codebook(self) -> jnp.ndarray:
        return jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        codebook = self.variables["params"]["codebook"]
        return jnp.take(codebook, ids, axis=0)


class GumbelVQ(nn.Module):
    """Gumbel VQ."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32

    @nn.compact
    def __call__(self, x, *, tau=1.0):
        codebook_size = self.config.vqvae.codebook_size
        codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="uniform"
            ),
            (codebook_size, x.shape[-1]),
        )
        codebook = jnp.asarray(codebook, dtype=self.dtype)
        distances = jnp.reshape(
            squared_euclidean_distance(jnp.reshape(x, (-1, x.shape[-1])), codebook),
            x.shape[:-1] + (codebook_size,),
        )
        result_dict = dict()
        encoding_indices = jnp.argmin(distances, axis=-1)
        if self.train:
            noise = jax.random.gumbel(
                self.make_rng("rng"), distances.shape, dtype=self.dtype
            )
            encodings = jax.nn.softmax((-distances + noise) / tau, axis=-1)
            quantized = self.quantize(encodings)
        else:
            encodings = jax.nn.one_hot(
                encoding_indices, codebook_size, dtype=self.dtype
            )
            quantized = self.quantize(encodings)
        result_dict.update(
            {
                "quantizer_loss": 0.0,
                "encodings": encodings,
                "encoding_indices": encoding_indices,
            }
        )
        return quantized, result_dict

    def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
        codebook = jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)
        return jnp.dot(z, codebook)

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.variables["params"]["codebook"], ids, axis=0)


class VQVAE(nn.Module):
    """VQVAE model."""

    config: ml_collections.ConfigDict
    train: bool
    dtype: int = jnp.float32
    activation_fn: Any = nn.relu

    def setup(self):
        """VQVAE setup."""
        if self.config.vqvae.quantizer == "gumbel":
            self.quantizer = GumbelVQ(
                config=self.config, train=self.train, dtype=self.dtype
            )
        elif self.config.vqvae.quantizer == "vq":
            self.quantizer = VectorQuantizer(
                config=self.config, train=self.train, dtype=self.dtype
            )
        else:
            raise NotImplementedError
        output_dim = 3
        self.encoder = Encoder(config=self.config, train=self.train, dtype=self.dtype)
        self.decoder = Decoder(
            config=self.config,
            train=self.train,
            output_dim=output_dim,
            dtype=self.dtype,
        )

    def encode(self, input_dict):
        image = input_dict["image"]
        encoded_feature = self.encoder(image)
        if self.config.vqvae.quantizer == "gumbel" and self.train:
            quantized, result_dict = self.quantizer(
                encoded_feature, tau=input_dict["tau"]
            )
        else:
            quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        reconstructed = self.decoder(x)
        return reconstructed

    def get_codebook_funct(self):
        return self.quantizer.get_codebook()

    def decode_from_indices(self, inputs):
        if isinstance(inputs, dict):
            ids = inputs["encoding_indices"]
        else:
            ids = inputs
        features = self.quantizer.decode_ids(ids)
        reconstructed_image = self.decode(features)
        return reconstructed_image

    def encode_to_indices(self, inputs):
        if isinstance(inputs, dict):
            image = inputs["image"]
        else:
            image = inputs
        encoded_feature = self.encoder(image)
        _, result_dict = self.quantizer(encoded_feature)
        ids = result_dict["encoding_indices"]
        return ids

    def __call__(self, input_dict):
        quantized = self.encode(input_dict)
        outputs = self.decoder(quantized)
        return outputs


def restore_from_path(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        state = flax.serialization.from_bytes(None, f.read())
    return state


def get_config():
    config = ml_collections.ConfigDict()

    # vq checkpoint
    config.vq_checkpoint_url = "https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet256_checkpoint"

    # config of vqgan
    config.vqgan = ml_collections.ConfigDict()
    config.vqgan.loss_type = "non-saturating"
    config.vqgan.g_adversarial_loss_weight = 0.1
    config.vqgan.gradient_penalty = "r1"
    config.vqgan.grad_penalty_cost = 10.0

    # config of vqvae
    config.vqvae = ml_collections.ConfigDict()
    config.vqvae.quantizer = "vq"
    config.vqvae.codebook_size = 1024

    config.vqvae.entropy_loss_ratio = 0.1
    config.vqvae.entropy_temperature = 0.01
    config.vqvae.entropy_loss_type = "softmax"
    config.vqvae.commitment_cost = 0.25

    config.vqvae.filters = 128
    config.vqvae.num_res_blocks = 2
    config.vqvae.channel_multipliers = [1, 1, 2, 2, 4]
    config.vqvae.embedding_dim = 256
    config.vqvae.conv_downsample = False
    config.vqvae.activation_fn = "swish"
    config.vqvae.norm_type = "GN"

    return config


def get_image_tokenizer(image_tokenizer_type='maskgit'):
    if image_tokenizer_type == 'dalle':
        model = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
        params = {'params': model.params}
        params = flax.core.frozen_dict.freeze(params)

        def encode_fn(params, image):
            return model.module.apply(params, image, method=model.module.encode)[1]

        def decode_fn(params, code):
            return jnp.clip(
                model.module.apply(params, code, method=model.module.decode_code),
                0.0,
                1.0
            )

        return params, encode_fn, decode_fn, 16384
    elif image_tokenizer_type == 'maskgit':
        config = get_config()
        model = VQVAE(config=config, dtype=jnp.float32, train=False)
        params = flax.serialization.from_bytes(
            None, requests.get(config.vq_checkpoint_url, allow_redirects=True).content
        )
        params = flax.core.frozen_dict.freeze(params)

        def encode_fn(params, image):
            image_tokens = model.apply(
                params,
                image,
                method=model.encode_to_indices,
                mutable=False,
            )
            return image_tokens.reshape(image_tokens.shape[0], -1)

        def decode_fn(params, code):
            decoded = model.apply(
                params,
                code.reshape(-1, 16, 16),
                method=model.decode_from_indices,
                mutable=False,
            )
            return jnp.clip(decoded, 0.0, 1.0)

        return params, encode_fn, decode_fn, config.vqvae.codebook_size
    else:
        raise ValueError('Unsupported tokenizer model!')
