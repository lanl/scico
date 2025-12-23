# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax nnx implementation of different convolutional nets."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Callable, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx
from flax.core import Scope  # noqa

from .blocks import (
    ConvBNBlock,
    ConvBNMultiBlock,
    ConvBNPoolBlock,
    ConvBNUpsampleBlock,
    upscale_nn,
)

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


class DnCNNNet(nnx.Module):
    r"""Flax nnx implementation of DnCNN :cite:`zhang-2017-dncnn`.

    Flax nnx implementation of the convolutional neural network (CNN)
    architecture for denoising described in :cite:`zhang-2017-dncnn`.
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        num_filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        act: Callable[..., ArrayLike] = nnx.relu,
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize DnCNN model.

        Attributes:
            depth: Number of layers in the neural network.
            channels: Number of channels of input tensor.
            num_filters: Number of filters in the convolutional layers.
            kernel_size: Size of the convolution filters.
            strides: Convolution strides.
            act: Class of activation function to apply. Default:
                :func:`~flax.nnx.relu`.
            kernel_init: Flax function for initializing the convolution kernels. Default:
                :func:`~flax.nnx.initializers.kaiming_normal`.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()

        self.blocks = nnx.Sequential(
            nnx.Conv(
                channels,
                num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="CIRCULAR",
                use_bias=False,
                dtype=dtype,
                kernel_init=kernel_init(),
                rngs=rngs,
            ),
            act,
            *[
                nnx.Sequential(
                    ConvBNBlock(
                        num_filters,
                        num_filters,
                        act,
                        kernel_size,
                        strides,
                        kernel_init,
                        dtype,
                        rngs=rngs,
                    ),
                )
                for _ in range(depth - 2)
            ],
            nnx.Conv(
                num_filters,
                channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="CIRCULAR",
                use_bias=False,
                dtype=dtype,
                kernel_init=kernel_init(),
                rngs=rngs,
            ),
        )

    def __call__(self, inputs: ArrayLike) -> ArrayLike:
        """Apply DnCNN denoiser.

        Args:
            inputs: The array to be transformed.

        Returns:
            The denoised input.
        """
        # Application of DnCNN model.
        base = inputs
        outputs = self.blocks(inputs)
        return base - outputs  # residual-like network


class ResNet(nnx.Module):
    """Flax nnx implementation of convolutional network with residual connection.

    Net constructed from sucessive applications of convolution plus batch
    normalization blocks and ending with residual connection (i.e. adding
    the input to the output of the block)."""

    def __init__(
        self,
        depth: int,
        channels: int,
        num_filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        act: Callable[..., ArrayLike] = nnx.relu,
        kernel_init: Callable = nnx.initializers.xavier_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize ResNet model.

        Args:
            depth: Depth of residual net.
            channels: Number of channels of input tensor.
            num_filters: Number of filters in the layers of the block.
                Corresponds to the number of channels in the network
                processing.
            kernel_size: Size of the convolution filters.
            strides: Convolution strides.
            act: Class of activation function to apply. Default:
                :func:`~flax.nnx.relu`.
            kernel_init: Flax function for initializing the convolution kernels. Default:
                :func:`~flax.nnx.initializers.xavier_normal`.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()

        self.blocks = nnx.Sequential(
            ConvBNBlock(
                channels, num_filters, act, kernel_size, strides, kernel_init, dtype, rngs=rngs
            ),
            *[
                nnx.Sequential(
                    ConvBNBlock(
                        num_filters,
                        num_filters,
                        act,
                        kernel_size,
                        strides,
                        kernel_init,
                        dtype,
                        rngs=rngs,
                    ),
                )
                for _ in range(depth - 2)
            ],
            ConvBNBlock(
                num_filters,
                channels,
                nnx.identity,
                kernel_size,
                strides,
                kernel_init,
                dtype,
                rngs=rngs,
            ),
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply ResNet.

        Args:
            x: The array to be transformed.

        Returns:
            The ResNet result.
        """
        residual = x
        outputs = self.blocks(x)
        return outputs + residual


class ConvBNNet(nnx.Module):
    """Convolution and batch normalization net.

    Net constructed from sucessive applications of convolution plus batch
    normalization and activation blocks. No residual connection."""

    def __init__(
        self,
        depth: int,
        channels: int,
        num_filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        act: Callable[..., ArrayLike] = nnx.relu,
        kernel_init: Callable = nnx.initializers.xavier_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize ConvBNNet model.

        Args:
            depth: Depth of net.
            channels: Number of channels of input tensor.
            num_filters: Number of filters in the layers of the block.
                Corresponds to the number of channels in the network
                processing.
            kernel_size: Size of the convolution filters.
            strides: Convolution strides.
            act: Class of activation function to apply. Default:
                :func:`~flax.nnx.relu`.
            kernel_init: Flax function for initializing the convolution kernels. Default:
                :func:`~flax.nnx.initializers.xavier_normal`.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()

        self.blocks = nnx.Sequential(
            ConvBNBlock(
                channels, num_filters, act, kernel_size, strides, kernel_init, dtype, rngs=rngs
            ),
            *[
                nnx.Sequential(
                    ConvBNBlock(
                        num_filters,
                        num_filters,
                        act,
                        kernel_size,
                        strides,
                        kernel_init,
                        dtype,
                        rngs=rngs,
                    ),
                )
                for _ in range(depth - 2)
            ],
            ConvBNBlock(
                num_filters,
                channels,
                nnx.identity,
                kernel_size,
                strides,
                kernel_init,
                dtype,
                rngs=rngs,
            ),
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply ConvBNNet.

        Args:
            x: The array to be transformed.

        Returns:
            The ConvBNNet result.
        """
        return self.blocks(x)


class UNet(nnx.Module):
    """Flax nnx implementation of U-Net model :cite:`ronneberger-2015-unet`."""

    def __init__(
        self,
        depth: int,
        channels: int,
        num_filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        block_depth: int = 2,
        window_shape: Tuple[int, int] = (2, 2),
        upsampling: int = 2,
        act: Callable[..., ArrayLike] = nnx.relu,
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize U-Net model.

        Args:
            depth: Depth of U-Net.
            channels: Number of channels of input tensor.
            num_filters: Number of filters in the convolutional layer of the
                block. Corresponds to the number of channels in the network
                processing.
            kernel_size: Size of the convolution filters.
            strides: Convolution strides.
            block_depth: Number of processing layers per block.
            window_shape: Window for reduction for pooling and downsampling.
            upsampling: Factor for expanding.
            act: Class of activation function to apply. Default:
                :func:`~flax.nnx.relu`.
            kernel_init: Flax function for initializing the convolution kernels. Default:
                :func:`~flax.nnx.initializers.kaiming_normal`.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()

        # Processing at input resolution
        self.start = ConvBNMultiBlock(
            channels,
            block_depth,
            num_filters,
            act,
            kernel_size,
            strides,
            kernel_init,
            dtype,
            rngs=rngs,
        )

        # Downsampling path
        self.down = nnx.List([])
        for j in range(depth - 1):
            self.down.append(
                nnx.Sequential(
                    *[
                        ConvBNPoolBlock(
                            2**j * num_filters,
                            2 * 2**j * num_filters,
                            act,
                            nnx.max_pool,
                            kernel_size,
                            strides,
                            window_shape,
                            kernel_init,
                            dtype,
                            rngs=rngs,
                        ),
                        ConvBNMultiBlock(
                            2 * 2**j * num_filters,
                            block_depth,
                            2 * 2**j * num_filters,
                            act,
                            kernel_size,
                            strides,
                            kernel_init,
                            dtype,
                            rngs=rngs,
                        ),
                    ]
                )
            )

        # Definition of upscaling function
        upfn = partial(upscale_nn, scale=upsampling)

        # Upsampling path
        self.up = nnx.List([])
        for j in reversed(range(depth - 1)):
            self.up.append(
                nnx.List(
                    [
                        ConvBNUpsampleBlock(
                            2 * 2**j * num_filters,
                            2**j * num_filters,
                            act,
                            upfn,
                            kernel_size,
                            strides,
                            kernel_init,
                            dtype,
                            rngs=rngs,
                        ),
                        ConvBNMultiBlock(
                            2 * 2**j * num_filters,
                            block_depth,
                            2**j * num_filters,
                            act,
                            kernel_size,
                            strides,
                            kernel_init,
                            dtype,
                            rngs=rngs,
                        ),
                    ]
                )
            )

        # Final conv1x1
        ksz_out = (1, 1)
        self.end = nnx.Conv(
            num_filters,
            channels,
            kernel_size=ksz_out,
            strides=strides,
            padding="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            kernel_init=kernel_init(),
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply U-Net.

        Args:
            x: The array to be transformed.

        Returns:
            The U-Net result.
        """
        out = self.start(x)

        # going down
        residual = []
        for block in self.down:
            residual.append(out)  # for skip connections
            out = block(out)  # Pooling and multiconvbn block

        # going up
        for block in self.up:
            out = block[0](out)  # Upsample block
            # skip connection
            out = jnp.concatenate((residual.pop(), out), axis=-1)  # channel last
            out = block[1](out)

        return self.end(out)
