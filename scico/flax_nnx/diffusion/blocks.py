# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax NNX implementation of different neural network blocks for
diffusion generative models.

The form of these blocks is based on `PyTorch code for diffusion models
<https://huggingface.co/blog/annotated-diffusion>`_, but with a number of
differences:

- Modules have been edited to assume channel last which is the Flax
  convention.
- The :class:`Upsample` and :class:`Downsample` blocks use a different
  interface that separates factors for channels than factors for resizing,
  and provide a parameter for specifying resizing mode. This interface
  affords more flexibility.
- Some block names have been changed to provide a more specific
  description (e.g. :class:`ConvGroupNBlock` instead of `Block`).
- Block :class:`WeightStandardizedConv2d` has not been implemented.
- The UNet specification has changed to use the custom :class:`Upsample`
  and :class:`Downsample` blocks.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import math
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from einops import rearrange

from flax import nnx
from flax.core import Scope  # noqa
from scico.flax.diffusion.helpers import default, exists

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


def get_timestep_embedding(timesteps: ArrayLike, embedding_dim: int = 128):
    """Construct an embedding for a sequence of time steps.

    Args:
        timesteps: Sequence of time steps to embed.
        embedding_dim: Embedding dimension.

    Returns:
        Time steps as an embedded sequence with specified dimension.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)

    emb = jnp.asarray(timesteps, dtype=jnp.float32) * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [0, 1], mode="constant")

    return emb


class SinusoidalPositionEmbeddings(nnx.Module):
    """Define sinusoidal positional embeddings class."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings class.

        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        """Compute embeddings."""
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -embeddings)
        embeddings = jnp.asarray(time, dtype=jnp.float32) * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings


class Residual(nnx.Module):
    """Define residual block."""

    def __init__(self, fn: Callable):
        """Initialize residual block.

        Args:
            fn: Given processing block.
        """
        super().__init__()
        self.fn = fn

    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Apply residual block, i.e. add input to block output.

        Args:
            x: The array to be transformed.
            args: Arguments of given processing block.
            kwargs: Keyword arguments of given processing block.
        """
        return self.fn(x, *args, **kwargs) + x


class Upsample(nnx.Module):
    """Define upsample Flax block."""

    def __init__(
        self,
        ftrs: int,
        ftrs_out: Optional[int] = None,
        factor: float = 2,
        shp_out: Optional[Tuple[int, int]] = None,
        method: str = "bilinear",
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize upsample Flax block.

        Args:
            ftrs: Number of features (a.k.a. channels).
            ftrs_out: Optional number of output features (a.k.a. channels).
            factor: Factor to use in the spatial upsample.
            shp_out: Shape of output signal. If given, it is prioritized
                over the factor argument.
            method: Method for upsampling. Options (strings): "nearest",
                "linear", "bilinear", "trilinear", "triangle", "cubic",
                "bicubic", "tricubic", "lanczos3", "lanczos5".
            rngs: Random generation key.
        """

        super().__init__()
        self.factor = factor
        self.shp_out = shp_out
        self.method = method
        self.out_ = nnx.Conv(
            ftrs, default(ftrs_out, ftrs), kernel_size=(3, 3), padding=1, rngs=rngs
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply upsample."""
        B, H, W, C = x.shape
        if self.shp_out is not None:
            hnew, wnew = self.shp_out
        else:
            hnew = int(H * self.factor)
            wnew = int(W * self.factor)
        out = jax.image.resize(x, shape=(B, hnew, wnew, C), method=self.method)
        out = self.out_(out)
        return out


class Downsample(nnx.Module):
    """Define downsample Flax block."""

    def __init__(
        self,
        ftrs: int,
        ftrs_out: Optional[int] = None,
        factor: float = 2,
        shp_out: Optional[Tuple[int, int]] = None,
        method="bilinear",
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize downsample Flax block.

        Args:
            ftrs: Number of features (a.k.a. channels).
            ftrs_out: Optional number of output features (a.k.a. channels).
            factor: Factor to use in the spatial downsample.
            shp_out: Shape of output signal. If given, it is prioritized
                over the factor argument.
            method: Method for downsampling. Options (strings): "nearest",
                "linear", "bilinear", "trilinear", "triangle", "cubic",
                "bicubic", "tricubic", "lanczos3", "lanczos5".
            rngs: Random generation key.
        """

        super().__init__()
        self.factor = factor
        self.shp_out = shp_out
        self.method = method
        in_features = int(ftrs * factor * factor)
        self.out_ = nnx.Conv(in_features, default(ftrs_out, ftrs), kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        """Apply downsample."""
        B, H, W, C = x.shape
        if self.shp_out is not None:
            hnew, wnew = self.shp_out
        else:
            hnew = int(H // self.factor)
            wnew = int(W // self.factor)
        cnew = int(C * self.factor * self.factor)
        out = jax.image.resize(x, shape=(B, hnew, wnew, cnew), method=self.method)
        return self.out_(out)


class ConvGroupNBlock(nnx.Module):
    """Define group normalization for convolution layer."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        act: Callable[..., ArrayLike] = nnx.silu,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize group normalization for convolution block.

        Args:
            dim: Dimensionality of input signal.
            dim_out: Dimensionality of output signal.
            groups: Number of groups.
            kernel_size: Size of convolutional filter.
            act: Activation function.
            rngs: Random generation key.
        """
        super().__init__()
        padding = int(kernel_size[0] // 2)
        self.proj = nnx.Conv(dim, dim_out, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.norm = nnx.GroupNorm(num_features=dim_out, num_groups=groups, rngs=rngs)
        self.act = act

    def __call__(self, x, scale_shift=None):
        """Apply group normalization."""
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nnx.Module):
    """Define ResNet :cite:`he-2016-deep` block."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: Optional[int] = None,
        groups: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        act: Callable[..., ArrayLike] = nnx.silu,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize ResNet block.

        Args:
            dim: Dimensionality of input signal.
            dim_out: Dimensionality of output signal.
            time_emb_dim: Dimensionality of time embedding.
            groups: Number of groups.
            kernel_size: Size of convolutional filter.
            act: Activation function.
            rngs: Random generation key.
        """
        super().__init__()

        self.mlp = (
            nnx.Sequential(*[nnx.silu, nnx.Linear(time_emb_dim, dim_out * 2, rngs=rngs)])
            if exists(time_emb_dim)
            else None
        )

        self.block1 = ConvGroupNBlock(
            dim, dim_out, groups=groups, kernel_size=kernel_size, act=act, rngs=rngs
        )
        self.block2 = ConvGroupNBlock(
            dim_out, dim_out, groups=groups, kernel_size=kernel_size, act=act, rngs=rngs
        )
        self.res_conv = (
            nnx.Conv(dim, dim_out, kernel_size=(1, 1), rngs=rngs)
            if dim != dim_out
            else nnx.nn.activations.identity
        )

    def __call__(self, x: ArrayLike, time_emb=None) -> ArrayLike:
        """Apply block."""
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")  # channel last in flax
            scale_shift = jnp.split(time_emb, 2, axis=-1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nnx.Module):
    """Define attention block."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize attention block.

        Args:
            dim: Dimensionality of signal.
            heads: Number of heads for attention layer.
            dim_head: Dimension per head.
            rngs: Random generation key.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * self.heads

        self.qkv_ = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.out_ = nnx.Conv(hidden_dim, dim, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply attention block."""
        b, h, w, c = x.shape  # channel last in flax
        qkv = self.qkv_(x)
        qkv = jnp.split(qkv, 3, axis=-1)  # channel last in flax
        q, k, v = map(lambda t: rearrange(t, "b x y (h c)  -> b (x y) h c", h=self.heads), qkv)
        q = q * self.scale

        sim = jnp.einsum("b d h i, b d h j -> b i h j", q, k)
        sim = sim - jnp.amax(sim, axis=-1, keepdims=True)
        attn = nnx.softmax(sim, axis=-1)
        out = jnp.einsum("b i h j, b d h j -> b d h i", attn, v)
        out = rearrange(out, "b (x y) h c  -> b x y (h c)", x=h, y=w)
        return self.out_(out)


class LinearAttention(nnx.Module):
    """Define linear attention block."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize linear attention block.

        Args:
            dim: Dimensionality of signal.
            heads: Number of heads for attention layer.
            dim_head: Dimension per head.
            rngs: Random generation key.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * self.heads

        self.qkv_ = nnx.Conv(dim, hidden_dim * 3, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.out_ = nnx.Sequential(
            *[
                nnx.Conv(hidden_dim, dim, kernel_size=(1, 1), rngs=rngs),
                nnx.GroupNorm(num_features=dim, num_groups=1, rngs=rngs),
            ]
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply linear attention block."""

        b, h, w, c = x.shape  # channel last in Flax
        qkv = self.qkv_(x)
        qkv = jnp.split(qkv, 3, axis=-1)  # channel last in Flax
        q, k, v = map(lambda t: rearrange(t, "b x y (h c)  -> b (x y) h c", h=self.heads), qkv)
        q = nnx.softmax(q, axis=-2)
        k = nnx.softmax(k, axis=-1)

        q = q * self.scale
        context = jnp.einsum("b n h d, b n h e -> b d h e", k, v)
        out = jnp.einsum("b d h e, b n h d -> b n h e", context, q)
        out = rearrange(out, "b (x y) h c  -> b x y (h c)", h=self.heads, x=h, y=w)

        return self.out_(out)


class PreNorm(nnx.Module):
    """Define pre-normalization block."""

    def __init__(self, dim: int, fn: Callable, rngs: nnx.Rngs = nnx.Rngs(0)):
        """Initialize pre-normalization block.

        Args:
            dim: Dimensionality for group normalization.
            fn: Given processing block.
            rngs: Random generation key.
        """
        # (pre or post in transformers is still in debate)
        super().__init__()
        self.fn = fn
        self.norm = nnx.GroupNorm(num_features=dim, num_groups=1, rngs=rngs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply group norm before block."""
        x = self.norm(x)
        return self.fn(x)
