# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural networks for diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx
from scico.flax_nnx.diffusion.blocks import (
    Attention,
    Downsample,
    LinearAttention,
    PreNorm,
    Residual,
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    Upsample,
)
from scico.flax.diffusion.helpers import default

class ConditionalUNet(nnx.Module):
    """Define Flax conditional U-Net model."""
    def __init__(self, dim: int, init_dim: Optional[int] = None, out_dim: Optional[int] = None,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8), channels: int = 3,
                 self_condition: bool = False, resnet_block_groups: int = 4,
                 kernel_size: Tuple[int, int] = (7, 7), padding: int = 3,
                 time_embed: bool = True, dtype: Any = jnp.float32,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        """Initialize Flax conditional U-Net model.

        Args:
            dim: Dimension of signal.
            init_dim: Optional dimension of first convolution layer.
            out_dim: Optional dimension of output convolution layer.
            dim_mults: Dimension multipliers at each level of the Unet.
            channels: Number of channels of signal to process.
            self_condition: Flag to include additional processing channel
                if building conditional model.
            resnet_block_groups: Number of groups in the residual network
                blocks.
            kernel_size: A shape tuple defining the size of the
                convolution filters.
            padding: An integer defining the size of the padding for the
                convolution filters.
            time_embed: Flag to indicate that the model uses a time embedding
                component. This is used when initializing model parameters
                and should not be changed.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """

        super().__init__()
        
        self.dtype = dtype
        self.time_embed = time_embed

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nnx.Conv(input_channels, init_dim, kernel_size=(1, 1), padding=0, rngs=rngs)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        print(f"dims: {dims}")
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"in_out: {in_out}")

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nnx.Sequential(
            *[
                SinusoidalPositionEmbeddings(dim),
                nnx.Linear(dim, time_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(time_dim, time_dim, rngs=rngs),
            ]
        )
        
        # layers
        downs = []
        ups = []
        num_resolutions = len(in_out)

        # Configure down path of Unet
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            downs.append(
                [
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, rngs=rngs),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, rngs=rngs),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in, rngs=rngs), rngs=rngs)),
                    (
                        Downsample(dim_in, dim_out, rngs=rngs)
                        if not is_last
                        else nnx.Conv(dim_in, dim_out, kernel_size=(3, 3), padding=1, rngs=rngs)
                    ),
                ]
            )

        # Configure bottleneck of Unet
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, rngs=rngs)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, rngs=rngs), rngs=rngs))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, rngs=rngs)

        # Configure up path of Unet
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            ups.append(
                [
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, rngs=rngs),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, rngs=rngs),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out, rngs=rngs), rngs=rngs)),
                    (
                        Upsample(dim_out, dim_in, rngs=rngs)
                        if not is_last
                        else nnx.Conv(dim_out, dim_in, kernel_size=(3, 3), padding=1, rngs=rngs)
                    ),
                ]
            )

        self.downs = downs
        self.ups = ups

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, rngs=rngs)
        self.final_conv = nnx.Conv(dim, self.out_dim, kernel_size=(1, 1), rngs=rngs)


    def __call__(self, x: ArrayLike, time: ArrayLike, x_self_cond: ArrayLike = None) -> ArrayLike:
        """Apply conditional Unet model.

        Args:
            x: The array to process.
            time: The array with the time embedding component.
            x_self_cond: The array for conditional processing.

        Returns:
            The processed array.
        """

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: jnp.zeros_like(x))
            x = jnp.concatenate([x_self_cond, x], axis=-1)

        x = self.init_conv(x)
        r = x.copy()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            print(f"Shape before downsampling: {x.shape}")
            x = downsample(x)
            print(f"Shape after downsampling: {x.shape}")

        print("Shapes stored in h")
        for hi in h:
            print(f"{hi.shape}")
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block1(x, t)

            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block2(x, t)
            x = attn(x)

            print(f"Shape before upsampling: {x.shape}")
            x = upsample(x)
            print(f"Shape after upsampling: {x.shape}")

        x = jnp.concatenate([x, r], axis=-1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
