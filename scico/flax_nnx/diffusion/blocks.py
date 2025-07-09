# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax NNX implementation of different neural network blocks for
   diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial

import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.nn.initializers import variance_scaling

from flax import nnx
from flax.nnx import initializers

def init_variance_scaling():
    return partial(variance_scaling, mode="fan_in", distribution="uniform")

def get_sinusoidal_positional_embedding(timesteps: ArrayLike, embedding_dim: int = 128):
    """Construct a sinusoidal position embedding for a sequence of time steps.

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


class TimestepEmbedding(nnx.Module):
    """Class to construct a timestep embedding.
    
    Modified from: https://github.com/annegnx/PnP-Flow/blob/main/pnpflow/models.py
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, act_fun: Callable = nnx.swish, rngs: nnx.Rngs = nnx.Rngs(0),):
        """Initialize timestep embedding class.

        Args:
            embedding_dim: Embedding dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output layer dimension.
            act_fun: Activation function.
            rngs: Random generation key.
        """
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        linear1 = nnx.Linear(in_features = embedding_dim,
                             out_features = hidden_dim,
                             kernel_init = init_variance_scaling(scale=1.),
                             bias_init = initializers.zeros_init(),
                             rngs=rngs
                             )

        linear2 = nnx.Linear(in_features = hidden_dim,
                             out_features = output_dim,
                             kernel_init = init_variance_scaling(scale=1.),
                             bias_init = initializers.zeros_init(),
                             rngs=rngs
                             )

        self.block = nnx.Sequential([linear1, act_fun, linear2])
        
    
    def __call__(self, tstep: ArrayLike) -> ArrayLike:
        """Compute time step embedding for a sequence of timesteps.

        Args:
            tstep: Timesteps to embed.
            
        Returns:
            Embeded sequence of timesteps.
        """
        temb = get_sinusoidal_positional_embedding(tstep, self.embedding_dim)
        return self.block(temb)


class IdentityLayer(nnx.Module):
    """Class to apply an identity transform."""
    def __init__(self,):
        """Initialize identity."""
        super().__init__()
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return x


class ResidualBlock(nnx.Module):
    def __init__(self,
                 in_ch: int,
                 temb_ch: int,
                 out_ch: Optional[int] = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.,
                 normalize: Callable = nnx.GroupNorm,
                 act_fun: Callable = nnx.swish,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 ):
        """Initialization of residual block.

        Args:
            in_ch: Number of input channels.
            temb_ch: Number of timestep embedding channels.
            out_ch: Number of output channels.
            conv_shortcut: Flag to indicate if a residual connection passed
                through a convolution layer is to be used (``True``) or not (``False``).
            dropout: Dropout factor to apply per layer (if any).
            normalize: Normalization function.
            act_fun: Activation function.
            rngs: Random generation key.
        """
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act

        self.temb_proj = nnx.Linear(in_features = temb_ch,
                             out_features = out_ch,
                             kernel_init = init_variance_scaling(scale=1.),
                             bias_init = initializers.zeros_init(),
                             rngs=rngs
                             )
                                     
        self.norm1 = IdentityLayer()
        self.norm2 = IdentityLayer()
        if isinstance(normalize, nnx.GroupNorm):
            self.norm1 = nnx.GroupNorm(num_features=in_ch, num_groups=32, rngs=rngs)
            self.norm2 = nnx.GroupNorm(num_features=out_ch, num_groups=32, rngs=rngs)
            
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding=1, kernel_init=                   init_variance_scaling(scale=1.), bias_init =
                             initializers.zeros_init(), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding=1, kernel_init=                   init_variance_scaling(scale=0.), bias_init =
                              initializers.zeros_init(), rngs=rngs)
        
        
        if dropout > 0.:
            self.dropout = nnx.Dropout(dropout, rngs=rngs)
        else:
            self.dropout = IdentityLayer()

        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding=1,
                                         kernel_init=init_variance_scaling(scale=1.),
                                         bias_init=initializers.zeros_init(), rngs=rngs)
            else:
                self.shortcut = nnx.Conv(in_ch, out_ch, kernel_size=(1, 1), padding=0,
                                         kernel_init=init_variance_scaling(scale=1.),
                                         bias_init=initializers.zeros_init(), rngs=rngs)
        else:
            self.shortcut = IdentityLayer()
            
    def __call__(self, x: ArrayLike, temb: ArrayLike) -> ArrayLike:
        # call conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # add in timestep embedding
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        # call conv2
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # apply shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h


class SelfAttention(nnx.Module):
    """Implementation of self attention layer.
    
    Copied and modified from: https://github.com/annegnx/PnP-Flow/blob/main/pnpflow/models.py
    """

    def __init__(self, in_channels: int, normalize: Callable = nnx.GroupNorm, rngs: nnx.Rngs = nnx.Rngs(0),):
        """Initialization of self attention block.

        Args:
            in_channels: Number of input channels.
            normalize: Normalization function.
            rngs: Random generation key.
        """
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding=0,
                               kernel_init = init_variance_scaling(scale=1.),
                               bias_init = initializers.zeros_init(), rngs=rngs)
        self.attn_k = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding=0,
                               kernel_init = init_variance_scaling(scale=1.),
                               bias_init = initializers.zeros_init(), rngs=rngs)
        self.attn_v = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding=0,
                               kernel_init = init_variance_scaling(scale=1.),
                               bias_init = initializers.zeros_init(), rngs=rngs)
        
        self.proj_out = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding=0,
                                 kernel_init = init_variance_scaling(scale=0.),
                                 bias_init = initializers.zeros_init(), rngs=rngs)
        
        self.softmax = nnx.Softmax(axis=-1)
        if normalize is not None:
            self.norm = normalize(num_features=in_channels, num_groups=32, rngs=rngs)
        else:
            self.norm = IdentityLayer()

    def __call__(self, x: ArrayLike, temp=None) -> ArrayLike:
        """Apply self attention block.
        
        Args:
            x: Array for self attention.
            temp: not used. 
            
        Returns:
            Processed array.
        """
        _, H, W, C = x.shape # Flax is channel last

        h = self.norm(x)
        q = self.attn_q(h).reshape((-1, H * W, C))
        k = self.attn_k(h).reshape((-1, H * W, C))
        v = self.attn_v(h).reshape((-1, H * W, C))

        #attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = jnp.einsum("b j i, b j k -> b i k", q, k)
        attn = self.softmax(attn)

        #h = torch.bmm(v, attn.permute(0, 2, 1))
        h = jno.einsum("b i j, b k j -> b i k", v, attn)
        h = h.reshape((-1, H, W, C))
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


