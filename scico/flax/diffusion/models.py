# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural networks for diffusion generative models."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from typing import Optional, Tuple

from jax.typing import ArrayLike

import flax.linen as nn
from scico.flax.diffusion.blocks import (
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


class ConditionalUnet(nn.Module):
    """Conditional U Net model.

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
    """

    dim: int
    init_dim: Optional[int] = None
    out_dim: Optional[int] = None
    dim_mults: Tuple[int] = (1, 2, 4, 8)
    channels: int = 3
    self_condition: bool = False
    resnet_block_groups: int = 4
    kernel_size: Tuple[int, int] = (7, 7)
    padding: int = 3

    def setup(self):
        """Setup of layers in conditional Unet model."""
        super().__init__()

        # determine dimensions
        input_channels = self.channels * (2 if self.self_condition else 1)

        init_dim = default(self.init_dim, self.dim)
        self.init_conv = nn.Conv(init_dim, kernel_size=self.kernel_size, padding=self.padding)

        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)

        # time embeddings
        time_dim = self.dim * 4

        self.time_mlp = nn.Sequential(
            [
                SinusoidalPositionEmbeddings(self.dim),
                nn.Dense(time_dim),
                nn.gelu,
                nn.Dense(time_dim),
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
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv(dim_out, kernel_size=(3, 3), padding=1),
                ]
            )

        # Configure bottleneck of Unet
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Configure up path of Unet
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            ups.append(
                [
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv(dim_in, kernel_size=(3, 3), padding=1),
                ]
            )

        self.downs = downs
        self.ups = ups

        out_dim = default(self.out_dim, self.channels)

        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv(out_dim, kernel_size=(1, 1))

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
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block1(x, t)

            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = jnp.concatenate([x, r], axis=-1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
