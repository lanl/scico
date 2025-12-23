# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural networks for diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx
from scico.flax.diffusion.helpers import default
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


class ConditionalUNet(nnx.Module):
    """Define Flax conditional U-Net model."""

    def __init__(
        self,
        shape: Tuple[int, int],
        channels: int = 3,
        init_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        self_condition: bool = False,
        resnet_block_groups: int = 4,
        kernel_size: Tuple[int, int] = (7, 7),
        time_embed: bool = True,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize Flax conditional U-Net model.

        Args:
            shape: Shape of signal.
            channels: Number of channels of signal to process.
            init_channels: Optional features (a.k.a. output channels) of
                first convolution layer.
            out_channels: Optional features (a.k.a. output channels) of
                output convolution layer.
            dim_mults: Dimension multipliers at each level of the Unet.
            self_condition: Flag to include additional processing channel
                if building conditional model.
            resnet_block_groups: Number of groups in the residual network
                blocks.
            kernel_size: A shape tuple defining the size of the
                convolution filters.
            time_embed: Flag to indicate that the model uses a time
                embedding component. This is used when initializing model
                parameters and should not be changed.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """

        super().__init__()

        self.dtype = dtype
        self.time_embed = time_embed

        # Determine feature dimensions.
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        padding = int(kernel_size[0] // 2)  # padding for the convolution filters

        init_channels = default(init_channels, channels)
        self.init_conv = nnx.Conv(
            input_channels, init_channels, kernel_size=kernel_size, padding=padding, rngs=rngs
        )

        self.dim_mults = dim_mults

        features = [init_channels, *map(lambda m: int(init_channels * m), self.dim_mults)]
        in_out_f = list(zip(features[:-1], features[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups, kernel_size=kernel_size)

        # Define time embeddings.
        dim = (max(shape) // 2) * 2
        time_dim = dim * 4

        self.time_mlp = nnx.Sequential(
            *[
                SinusoidalPositionEmbeddings(dim),
                nnx.Linear(dim, time_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(time_dim, time_dim, rngs=rngs),
            ]
        )

        # Define layer storage.
        downs = nnx.List([])
        ups = nnx.List([])
        num_resolutions = len(in_out_f)
        shps = nnx.List([shape])

        # Configure down path of Unet.
        for ind, (feat_in, feat_out) in enumerate(in_out_f):
            shps.append(
                (int(shape[0] // self.dim_mults[ind]), int(shape[1] // self.dim_mults[ind]))
            )

            downs.append(
                [
                    block_klass(feat_in, feat_in, time_emb_dim=time_dim, rngs=rngs),
                    block_klass(feat_in, feat_in, time_emb_dim=time_dim, rngs=rngs),
                    Residual(PreNorm(feat_in, LinearAttention(feat_in, rngs=rngs), rngs=rngs)),
                    (
                        Downsample(
                            feat_in,
                            feat_out,
                            factor=self.dim_mults[ind],
                            shp_out=shps[-1],
                            rngs=rngs,
                        )
                    ),
                ]
            )

        shps = shps[:-1]
        # Configure bottleneck of Unet.
        mid_feat = features[-1]
        self.mid_block1 = block_klass(mid_feat, mid_feat, time_emb_dim=time_dim, rngs=rngs)
        self.mid_attn = Residual(PreNorm(mid_feat, Attention(mid_feat, rngs=rngs), rngs=rngs))
        self.mid_block2 = block_klass(mid_feat, mid_feat, time_emb_dim=time_dim, rngs=rngs)

        # Configure up path of Unet.
        for ind, (feat_in, feat_out) in enumerate(reversed(in_out_f)):

            ups.append(
                [
                    block_klass(2 * feat_in, feat_in, time_emb_dim=time_dim, rngs=rngs),
                    block_klass(2 * feat_in, feat_in, time_emb_dim=time_dim, rngs=rngs),
                    Residual(PreNorm(feat_in, LinearAttention(feat_in, rngs=rngs), rngs=rngs)),
                    (
                        Upsample(
                            feat_out,
                            feat_in,
                            factor=self.dim_mults[-(ind + 1)],
                            shp_out=shps[-(ind + 1)],
                            rngs=rngs,
                        )
                    ),
                ]
            )

        self.downs = downs
        self.ups = ups

        self.out_channels = default(out_channels, channels)

        self.final_res_block = block_klass(
            features[0] * 2, features[0], time_emb_dim=time_dim, rngs=rngs
        )
        self.final_conv = nnx.Conv(features[0], self.out_channels, kernel_size=(1, 1), rngs=rngs)

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

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:

            x = upsample(x)
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block1(x, t)

            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block2(x, t)
            x = attn(x)

        x = jnp.concatenate([x, r], axis=-1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
