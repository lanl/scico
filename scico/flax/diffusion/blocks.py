# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementation of different neural network blocks for
   diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import math
from typing import Callable, Optional

import jax
from jax.typing import ArrayLike

from einops import rearrange
from einops.layers.flax import Rearrange

import flax.linen as nn
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from scico.flax.diffusion.helpers import default, exists

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


class Residual(nn.Module):
    """Residual block.

    Args:
        fn: Given processing block.
    """

    fn: Callable

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Apply residual block, i.e. add input to block output.
        Args:
            x: The array to be transformed.
            args: Arguments of given processing block.
            kwargs: Keyword arguments of given processing block.
        """
        return self.fn(x, *args, **kwargs) + x


class Upsample(nn.Module):
    """Upsample Flax block."""

    dim: int
    dim_out: Optional[int] = None

    @nn.compact
    def __call__(self, x: ArrayLike):
        """Apply upsample."""
        factor = 2
        B, H, W, C = x.shape
        out = jax.image.resize(x, shape=(B, H * factor, W * factor, C), method="bilinear")
        out = nn.Conv(default(self.dim_out, self.dim), kernel_size=(3, 3), padding=1)(out)
        return out


class Downsample(nn.Module):
    """Downsample Flax block."""

    dim: int
    dim_out: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        """Apply downsample."""
        return nn.Sequential(
            [
                Rearrange("b (h p1) (w p2) c -> b h w (c p1 p2)", {"p1": 2, "p2": 2}),
                nn.Conv(default(self.dim_out, self.dim), kernel_size=(1, 1)),
            ]
        )(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """Definition of sinusoilda positional embeddings."""

    dim: int

    @nn.compact
    def __call__(self, time):
        """Compute embeddings."""
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -embeddings)
        # Next, alternatively
        embeddings = jnp.asarray(time, dtype=jnp.float32) * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings


class ConvGroupNBlock(nn.Module):
    """Group normalization for convolution layer."""

    dim_out: int
    groups: int = 8
    act: Callable[..., ArrayLike] = nn.silu

    @nn.compact
    def __call__(self, x, scale_shift=None):
        """Apply group normalization."""
        x = nn.Conv(self.dim_out, kernel_size=(3, 3), padding=1)(x)
        x = nn.GroupNorm(num_groups=self.groups)(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# https://arxiv.org/abs/1512.03385
class ResnetBlock(nn.Module):
    """Definition of Resnet block."""

    dim: int
    dim_out: int
    time_emb_dim: Optional[int] = None
    groups: int = 8

    @nn.compact
    def __call__(self, x, time_emb=None):
        """Apply block."""
        scale_shift = None
        if exists(self.time_emb_dim) and exists(time_emb):
            time_emb = nn.Sequential([nn.silu, nn.Dense(self.dim_out * 2)])(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")  # channel last in flax
            scale_shift = jnp.split(time_emb, 2, axis=-1)

        h = ConvGroupNBlock(self.dim_out, groups=self.groups)(x, scale_shift=scale_shift)
        h = ConvGroupNBlock(self.dim_out, groups=self.groups)(h)
        if self.dim != self.dim_out:
            return h + nn.Conv(self.dim_out, kernel_size=(1, 1))(x)
        return h + x


class Attention(nn.Module):
    """Definition of attention block."""

    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        """Apply attention block."""
        scale = self.dim_head**-0.5
        hidden_dim = self.dim_head * self.heads

        b, h, w, c = x.shape  # channel last in flax
        qkv_ = nn.Conv(hidden_dim * 3, kernel_size=(1, 1), use_bias=False)(x)
        qkv = jnp.split(qkv_, 3, axis=-1)  # channel last in flax
        q, k, v = map(lambda t: rearrange(t, "b x y (h c)  -> b (x y) h c", h=self.heads), qkv)
        q = q * scale

        sim = jnp.einsum("b d h i, b d h j -> b i h j", q, k)
        sim = sim - jnp.amax(sim, axis=-1, keepdims=True)
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i h j, b d h j -> b d h i", attn, v)
        out = rearrange(out, "b (x y) h c  -> b x y (h c)", x=h, y=w)
        return nn.Conv(self.dim, kernel_size=(1, 1))(out)


class LinearAttention(nn.Module):
    """Definition of linear attention block."""

    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        """Apply linear attention block."""
        scale = self.dim_head**-0.5
        hidden_dim = self.dim_head * self.heads

        b, h, w, c = x.shape  # channel last in flax
        qkv_ = nn.Conv(hidden_dim * 3, kernel_size=(1, 1), use_bias=False)(x)
        qkv = jnp.split(qkv_, 3, axis=-1)  # channel last in flax
        q, k, v = map(lambda t: rearrange(t, "b x y (h c)  -> b (x y) h c", h=self.heads), qkv)
        q = nn.softmax(q, axis=-2)
        k = nn.softmax(k, axis=-1)

        q = q * scale
        context = jnp.einsum("b n h d, b n h e -> b d h e", k, v)
        out = jnp.einsum("b d h e, b n h d -> b n h e", context, q)
        out = rearrange(out, "b (x y) h c  -> b x y (h c)", h=self.heads, x=h, y=w)

        return nn.Sequential([nn.Conv(self.dim, kernel_size=(1, 1)), nn.GroupNorm(self.dim)])(out)


class PreNorm(nn.Module):
    """Pre-normalization block."""

    # (pre or post in transformers is still in debate)
    dim: int
    fn: Callable

    @nn.compact
    def __call__(self, x):
        """Apply group norm before block."""
        x = nn.GroupNorm(self.dim)(x)
        return self.fn(x)
