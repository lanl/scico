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

from scico.flax_nnx.autoencoders.blocks import MLP
from scico.flax_nnx.diffusion.blocks import (
    TimestepEmbedding,
)


class AttnUnet(nnx.Module):
    """Unet model with attention layers."""
    
    def __init__(self,
                 input_channels: int,
                 input_dim: int,
                 channels: int,
                 output_channels: Optional[int] = None,
                 ch_mult: Tuple[int] = (1, 2, 4, 8),
                 kernel_size: Tuple[int, int] = (3, 3),
                 num_res_blocks: int = 2,
                 attn_resolutions: Tuple[int] = (16,),
                 dropout: float = 0.,
                 normalize: Callable = group_norm,
                 act_fun: Callable = nnx.swish,
                 resample_with_conv: bool = True,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 ):
        """Initialization of Unet model with attention.

        Args:
            input_channels: Number of channels of signal to process.
            input_dim: Dimension of signal.
            channels: Number of channels of signal to process.
            output_channels: Number of channels of output signal.
            ch_mults: Channel multipliers at each level of the Unet.
            kernel_size: A shape tuple defining the size of the
                convolution filters.
            num_res_blocks: Number of residual blocks.
            attn_resolutions: Number of levels in Unet.
            dropout: Dropout factor to apply per layer (if any).
            normalize: Normalization function.
            act_fun: Activation function.
            resample_with_conv: Perform resampling with convolution layer.
            rngs: Random generation key.
        """
        super().__init__()
        # store model parameters
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.channels = channels
        self.output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.act_fun = act_fun
        self.normalize = normalize

        # Initialize
        self.num_resolutions = len(ch_mult)
        in_dim = input_dim
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_dim % 2 ** (self.num_resolutions - 1) == 0, "input_height doesn't satisfy the condition"
        padding = kernel_size[0] // 2
        
        # Timestep embedding
        self.temb_block = TimestepEmbedding(
            embedding_dim = ch,
            hidden_dim = temb_ch,
            output_dim = temb_ch,
            act_fun = act_fun,
        )

        self.init_conv = nnx.Conv(input_channels, channels, kernel_size=kernel_size, padding=padding, rngs=rngs)
        unet_chs = [ch]
        in_ch = ch
        down_modules = []
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                )
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(
                        out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = downsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)




        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)
