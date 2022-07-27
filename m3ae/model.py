from typing import Callable, Optional, Any

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from functools import partial


def mask_union(mask1, mask2):
    return jnp.logical_or(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_intersection(mask1, mask2):
    return jnp.logical_and(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_not(mask):
    return 1.0 - mask


def mask_select(mask, this, other=None):
    if other is None:
        other = jnp.array(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask == 0.0, this, other)


def no_mask(x):
    return jnp.zeros(x.shape[:2])


def all_mask(x):
    return jnp.ones(x.shape[:2])


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = all_mask(tokens)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-5)

    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = jnp.sum(valid, axis=-1) / valid.shape[-1]
    return jnp.mean(
        jnp.mean(
            jnp.where(
                valid > 0.0,
                jnp.mean(jnp.square(patch_target - patch_output), axis=-1),
                jnp.array(0.0),
            ),
            axis=-1,
        ) / valid_ratio
    )


def extract_patches(inputs, patch_size):
    batch, height, width, channels = inputs.shape
    height, width = height // patch_size, width // patch_size
    x = jnp.reshape(inputs, (batch, height, patch_size, width, patch_size, channels))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * width, patch_size**2 * channels))
    return x


def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
    return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


def index_sequence(x, ids):
    return x[:, ids, ...]


def random_masking(x, rng, keep_len, padding_mask=None):
    batch, length, dim = x.shape
    noise = jax.random.uniform(rng, (length,), dtype=jnp.float32)
    ids_shuffle = jnp.argsort(noise, axis=0)
    ids_restore = jnp.argsort(ids_shuffle, axis=0)
    kept = index_sequence(x, ids_shuffle[:keep_len])
    mask = jnp.ones([batch, length], dtype=jnp.float32)
    mask = mask.at[:, :keep_len].set(0.0)
    mask = index_sequence(mask, ids_restore)

    if padding_mask is None:
        return kept, mask, ids_restore

    padding_mask_kept = index_sequence(padding_mask, ids_shuffle[:keep_len])
    return kept, mask, ids_restore, padding_mask_kept


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int
    input_norm: bool = True

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        if self.input_norm:
            x = nn.LayerNorm()(x)

        for i in range(self.depth):
            y = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            y = nn.gelu(y)
            y = nn.LayerNorm()(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class DropPath(nn.Module):
    dropout_prob: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=jnp.float32)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = nn.Dense(
            self.dim, kernel_init=self.kernel_init, name="fc1"
        )(inputs)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(
            self.out_dim, kernel_init=self.kernel_init, name="fc2"
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    """Modified from flax_models to support mask"""

    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None, padding_mask=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
        )(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale

        if padding_mask is not None:
            padding_mask = jnp.expand_dims(jnp.expand_dims(padding_mask, 1), 1)
            padding_mask = jnp.broadcast_to(padding_mask, attention.shape)
            attention = jnp.where(padding_mask > 0, jnp.array(-1e7), attention)

        attention = nn.softmax(attention, axis=-1)
        self.sow('intermediates', 'attention', attention)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(
            self.dim, kernel_init=nn.initializers.xavier_uniform()
        )(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    emb_dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, inputs, deterministic=False, padding_mask=None):
        x = nn.LayerNorm()(inputs)
        x = Attention(
            self.emb_dim, self.num_heads, True, self.att_drop, self.drop
        )(x, deterministic, padding_mask)
        x = DropPath(self.drop_path)(x, deterministic)
        inputs = inputs + x

        x = nn.LayerNorm()(inputs)
        x = TransformerMLP(
            self.emb_dim * self.mlp_ratio, self.emb_dim, self.drop,
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        return inputs + x


class Transformer(nn.Module):
    emb_dim: int = 1024
    depth: int = 24
    att_drop: float = 0
    drop: float = 0
    drop_path: float = 0
    num_heads: int = 16
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, deterministic=False, padding_mask=None):
        for _ in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.drop_path,
            )(x, deterministic, padding_mask)

        x = nn.LayerNorm()(x)
        return x


class MaskedMultimodalAutoencoder(nn.Module):
    config_updates: ... = None
    text_vocab_size: int = -1
    image_output_dim: int = 768

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.model_type = config_dict.placeholder(str)
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        # Dropout not applied in original MAE implementation.
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        # Tuned default mask ratio
        config.image_mask_ratio = 0.75
        config.text_mask_ratio = 0.75

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise', 'drop_path', 'dropout')

    @nn.nowrap
    def no_decay_list(self):
        # model specific no decay list
        no_decay = [
            'cls_token', 'encoder_image_type_embedding', 'encoder_text_type_embedding',
            'decoder_image_type_embedding', 'decoder_text_type_embedding',
            'image_mask_embedding', 'text_mask_embedding', 'text_embedding',
            'bias', 'embedding',
        ]
        return no_decay

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        assert self.text_vocab_size > 0

        self.text_embedding = nn.Embed(
            self.text_vocab_size, self.config.emb_dim,
            embedding_init=jax.nn.initializers.normal(stddev=1.0)
        )
        self.image_embedding = nn.Dense(
            self.config.emb_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )

        # Type embeddings
        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = self.param(
                "encoder_image_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.emb_dim),
            )
            self.decoder_image_type_embedding = self.param(
                "decoder_image_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.dec_emb_dim),
            )
            self.encoder_text_type_embedding = self.param(
                "encoder_text_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.emb_dim),
            )
            self.decoder_text_type_embedding = self.param(
                "decoder_text_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.dec_emb_dim),
            )

        # CLS and masks
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.config.emb_dim),
        )
        self.image_mask_embedding = self.param(
            "image_mask_embedding",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.config.dec_emb_dim),
        )
        self.text_mask_embedding = self.param(
            "text_mask_embedding",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.config.dec_emb_dim),
        )

        self.encoder = Transformer(
            emb_dim=self.config.emb_dim,
            depth=self.config.depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder = Transformer(
            emb_dim=self.config.dec_emb_dim,
            depth=self.config.dec_depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.dec_num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder_input_projection = nn.Dense(
            self.config.dec_emb_dim,
            kernel_init=nn.initializers.xavier_uniform(),
        )

        self.decoder_image_output = MLP(
            self.config.dec_emb_dim,
            self.image_output_dim,
            self.config.output_head_depth,
            input_norm=self.config.output_head_depth > 0,
        )

        self.decoder_text_output = MLP(
            self.config.dec_emb_dim,
            self.text_vocab_size,
            self.config.output_head_depth,
            input_norm=self.config.output_head_depth > 0,
        )

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'encoder_text_type_embedding': self.encoder_text_type_embedding,
                'decoder_image_type_embedding': self.decoder_image_type_embedding,
                'decoder_text_type_embedding': self.decoder_text_type_embedding,
            }[name]
        else:
            return 0.0

    def forward_representation(self, image, text, text_padding_mask, deterministic=False):
        batch_size = image.shape[0]
        cls_token = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.emb_dim))
        input_tensors = [cls_token]
        padding_masks = [jnp.zeros((batch_size,  1), dtype=jnp.float32)]
        if image is not None:
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(jnp.zeros((batch_size, image.shape[1]), dtype=jnp.float32))

        if text is not None:
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)

        x = jnp.concatenate(input_tensors, axis=1)
        padding_mask = jnp.concatenate(padding_masks, axis=1)
        x = self.encoder(x, deterministic, padding_mask)
        return x

    def forward_encoder(self, image, text, text_padding_mask, deterministic=False):
        if image is not None:
            batch_size = image.shape[0]
        else:
            batch_size = text.shape[0]
        cls_token = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.emb_dim))
        input_tensors = [cls_token]
        padding_masks = [jnp.zeros((batch_size, 1), dtype=jnp.float32)]
        if image is not None:
            image_keep_length = int(
                image.shape[1] * (1.0 - self.config.image_mask_ratio)
            )
            image_x = (
                self.image_embedding(image)
                + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
                + self.get_type_embedding('encoder_image_type_embedding')
            )
            image_x, image_mask, image_ids_restore = random_masking(
                image_x, self.make_rng("noise"), image_keep_length
            )
            input_tensors.append(image_x)
            padding_masks.append(jnp.zeros((batch_size, image_keep_length), dtype=jnp.float32))
        else:
            image_mask = image_ids_restore = None

        if text is not None:
            text_keep_length = int(
                text.shape[1] * (1.0 - self.config.text_mask_ratio)
            )
            text_x = (
                self.text_embedding(text)
                + get_1d_sincos_pos_embed(self.config.emb_dim, text.shape[1])
                + self.get_type_embedding('encoder_text_type_embedding')
            )
            text_x, text_mask, text_ids_restore, text_padding_mask = random_masking(
                text_x,
                self.make_rng("noise"),
                text_keep_length,
                text_padding_mask,
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)
        else:
            text_mask = text_ids_restore = text_padding_mask = None

        x = jnp.concatenate(input_tensors, axis=1)
        padding_mask = jnp.concatenate(padding_masks, axis=1)

        x = self.encoder(x, deterministic, padding_mask)

        cls_x = x[:, :1, :]
        if image is None:
            image_x = None
            text_x = x[:, 1:, :]
        elif text is None:
            image_x = x[:, 1:, :]
            text_x = None
        else:
            image_x = x[:, 1:image_keep_length + 1, :]
            text_x = x[:, image_keep_length + 1:, :]

        return cls_x, image_x, text_x, image_mask, text_mask, image_ids_restore, text_ids_restore

    def forward_decoder(
        self,
        cls_x,
        image_x,
        text_x,
        image_ids_restore,
        text_ids_restore,
        text_padding_mask,
        deterministic=False,
    ):
        batch_size = cls_x.shape[0]
        input_tensors = [self.decoder_input_projection(cls_x)]
        padding_masks = [jnp.zeros((batch_size,  1), dtype=jnp.float32)]

        if image_x is not None:
            image_keep_length = int(
                image_ids_restore.shape[0] * (1.0 - self.config.image_mask_ratio)
            )
            image_x = self.decoder_input_projection(image_x)
            masked_image_x = jnp.broadcast_to(
                self.image_mask_embedding,
                (
                    batch_size,
                    image_ids_restore.shape[0] - image_keep_length,
                    self.config.dec_emb_dim,
                ),
            )
            image_x = index_sequence(
                jnp.concatenate([image_x, masked_image_x], axis=1), image_ids_restore
            )
            image_x = (
                image_x
                + get_2d_sincos_pos_embed(self.config.dec_emb_dim, image_ids_restore.shape[0])
                + self.get_type_embedding('decoder_image_type_embedding')
            )
            input_tensors.append(image_x)
            padding_masks.append(jnp.zeros((batch_size, image_ids_restore.shape[0]), dtype=jnp.float32))

        if text_x is not None:
            text_keep_length = int(
                text_ids_restore.shape[0] * (1.0 - self.config.text_mask_ratio)
            )
            text_x = self.decoder_input_projection(text_x)
            masked_text_x = jnp.broadcast_to(
                self.text_mask_embedding,
                (
                    batch_size,
                    text_ids_restore.shape[0] - text_keep_length,
                    self.config.dec_emb_dim,
                ),
            )
            text_x = index_sequence(
                jnp.concatenate([text_x, masked_text_x], axis=1), text_ids_restore
            )
            text_x = (
                text_x
                + get_1d_sincos_pos_embed(self.config.dec_emb_dim, text_ids_restore.shape[0])
                + self.get_type_embedding('decoder_text_type_embedding')
            )
            input_tensors.append(text_x)
            padding_masks.append(text_padding_mask)

        x = jnp.concatenate(input_tensors, axis=1)
        padding_mask = jnp.concatenate(padding_masks, axis=1)
        x = self.decoder(x, deterministic, padding_mask)

        cls_x = x[:, :1, :]
        if image_x is None:
            image_output = None
            text_output = self.decoder_text_output(x[:, 1:, :])
        elif text_x is None:
            image_output = self.decoder_image_output(x[:, 1:, :])
            text_output = None
        else:
            image_output = self.decoder_image_output(x[:, 1:image_ids_restore.shape[0] + 1, :])
            text_output = self.decoder_text_output(x[:, image_ids_restore.shape[0] + 1:, :])

        return image_output, text_output

    def __call__(self, image, text, text_padding_mask, deterministic=False):
        (
            cls_x,
            image_x,
            text_x,
            image_mask,
            text_mask,
            image_ids_restore,
            text_ids_restore,
        ) = self.forward_encoder(image, text, text_padding_mask, deterministic)
        image_output, text_output = self.forward_decoder(
            cls_x,
            image_x,
            text_x,
            image_ids_restore,
            text_ids_restore,
            text_padding_mask,
            deterministic,
        )
        return image_output, text_output, image_mask, text_mask


class MaskedAutoencoder(nn.Module):
    config_updates: ... = None
    image_output_dim: int = 768

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.model_type = config_dict.placeholder(str)
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        # Dropout not applied in original MAE implementation.
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        # Tuned default mask ratio
        config.image_mask_ratio = 0.75

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise', 'drop_path', 'dropout')

    @nn.nowrap
    def no_decay_list(self):
        # model specific no decay list
        no_decay = [
            'cls_token', 'encoder_image_type_embedding', 'image_mask_embedding',
            'bias',
        ]
        return no_decay

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

        self.image_embedding = nn.Dense(
            self.config.emb_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )
        # Type embeddings
        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = self.param(
                "encoder_image_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.emb_dim),
            )
            self.decoder_image_type_embedding = self.param(
                "decoder_image_type_embedding",
                nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
                (1, 1, self.config.dec_emb_dim),
            )

        # CLS and masks
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.config.emb_dim),
        )
        self.image_mask_embedding = self.param(
            "image_mask_embedding",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.config.dec_emb_dim),
        )

        self.encoder = Transformer(
            emb_dim=self.config.emb_dim,
            depth=self.config.depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder = Transformer(
            emb_dim=self.config.dec_emb_dim,
            depth=self.config.dec_depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.dec_num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.decoder_input_projection = nn.Dense(
            self.config.dec_emb_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )

        self.decoder_image_output = MLP(
            self.config.dec_emb_dim,
            self.image_output_dim,
            self.config.output_head_depth,
            input_norm=self.config.output_head_depth > 0,
        )

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'decoder_image_type_embedding': self.decoder_image_type_embedding,
            }[name]
        else:
            return 0.0

    def forward_representation(self, image, deterministic=False):
        batch_size = image.shape[0]
        image_x = self.image_embedding(image)
        image_x = (
            image_x
            + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
            + self.get_type_embedding('encoder_image_type_embedding')
        )
        cls_token = jnp.broadcast_to(
            self.cls_token, (batch_size, 1, self.config.emb_dim)
        )
        x = jnp.concatenate([cls_token, image_x], axis=1)
        x = self.encoder(x, deterministic)
        return x

    def forward_encoder(self, image, deterministic=False):
        batch_size = image.shape[0]
        image_keep_length = int(image.shape[1] * (1.0 - self.config.image_mask_ratio))
        image_x = self.image_embedding(image)
        image_x = (
            image_x
            + get_2d_sincos_pos_embed(self.config.emb_dim, image.shape[1])
            + self.get_type_embedding('encoder_image_type_embedding')
        )
        image_x, image_mask, image_ids_restore = random_masking(
            image_x, self.make_rng("noise"), image_keep_length
        )
        cls_token = jnp.broadcast_to(
            self.cls_token, (batch_size, 1, self.config.emb_dim)
        )
        x = jnp.concatenate([cls_token, image_x], axis=1)
        x = self.encoder(x, deterministic)

        return x, image_mask, image_ids_restore

    def forward_decoder(self, x, image_ids_restore, deterministic=False):
        batch_size = x.shape[0]
        image_keep_length = int(image_ids_restore.shape[0] * (1.0 - self.config.image_mask_ratio))
        x = self.decoder_input_projection(x)
        encoder_cls = x[:, :1, :]
        image_x = x[:, 1:, :]

        masked_image_x = jnp.broadcast_to(
            self.image_mask_embedding,
            (
                batch_size,
                image_ids_restore.shape[0] - image_keep_length,
                self.config.dec_emb_dim,
            ),
        )

        image_x = index_sequence(
            jnp.concatenate([image_x, masked_image_x], axis=1), image_ids_restore
        )

        image_x = (
            image_x
            + get_2d_sincos_pos_embed(self.config.dec_emb_dim, image_ids_restore.shape[0])
            + self.get_type_embedding('decoder_image_type_embedding')
        )

        x = jnp.concatenate([encoder_cls, image_x], axis=1)
        x = self.decoder(x, deterministic)
        image_x = x[:, 1:, :]
        image_output = self.decoder_image_output(image_x)

        return image_output

    def __call__(self, image, deterministic=False):
        x, image_mask, image_ids_restore = self.forward_encoder(image, deterministic)
        image_output = self.forward_decoder(x, image_ids_restore, deterministic)
        return image_output, image_mask, x


class M3AETrainState(TrainState):
    tokenizer_params: Optional[flax.core.FrozenDict[str, Any]] = None


class LinearCLS(nn.Module):
    num_classes: int = 1000
    pool: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        if self.pool:
            x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
        else:
            x = x[:, 0]
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            use_scale=False,
            use_bias=False,
        )
        x = norm(name="bn")(x)
        logits = nn.Dense(self.num_classes)(x)
        return logits


class ViTClassifier(nn.Module):
    base_model: nn.Module
    num_classes: int
    global_pool: bool = False
    stop_gradient: bool = False

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise', 'drop_path')

    @nn.compact
    def __call__(self, x, deterministic=False, features=False):
        x = self.base_model.forward_representation(x, deterministic=deterministic)
        if self.global_pool:
            x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
        else:
            x = x[:, 0]

        z = x

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_classes)(x)
        logits = x
        log_probs = nn.log_softmax(x, axis=1)

        if features:
            return log_probs, logits, z
        else:
            return logits


def get_transformer_by_config(model_type, config):
    if model_type == 'small':
        config.emb_dim = 384
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 6
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'base':
        config.emb_dim = 768
        config.dec_emb_dim = 512
        config.depth = 12
        config.dec_depth = 8
        config.num_heads = 12
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'large':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'huge':
        config.emb_dim = 1280
        config.dec_emb_dim = 512
        config.depth = 32
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    elif model_type == 'debug':
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 2
        config.dec_depth = 2
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4
    else:
        raise ValueError('Unsupported model type!')
