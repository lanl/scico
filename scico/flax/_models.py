# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementation of different convolutional nets."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp

from flax.core import Scope  # noqa
from flax.linen import BatchNorm, Conv, max_pool, relu
from flax.linen.initializers import kaiming_normal, xavier_normal
from flax.linen.module import _Sentinel  # noqa
from flax.linen.module import Module, compact
from scico.flax.blocks import (
    ConvBNBlock,
    ConvBNMultiBlock,
    ConvBNPoolBlock,
    ConvBNUpsampleBlock,
    upscale_nn,
)
from scico.numpy import Array

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


ModuleDef = Any


class DnCNNNet(Module):
    r"""Flax implementation of DnCNN :cite:`zhang-2017-dncnn`.

    Flax implementation of the convolutional neural network (CNN)
    architecture for denoising described in :cite:`zhang-2017-dncnn`.

    Attributes:
        depth: Number of layers in the neural network.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layers.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
        act: Class of activation function to apply. Default: `nn.relu`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32
    act: Callable = relu

    @compact
    def __call__(
        self,
        inputs: Array,
        train: bool = True,
    ) -> Array:
        """Apply DnCNN denoiser.

        Args:
            inputs: The array to be transformed.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The denoised input.
        """
        # Definition using arguments common to all convolutions.
        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=kaiming_normal()
        )
        # Definition using arguments common to all batch normalizations.
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        # Definition and application of DnCNN model.
        base = inputs
        y = conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
            name="conv_start",
        )(inputs)
        y = self.act(y)
        for _ in range(self.depth - 2):
            y = ConvBNBlock(
                self.num_filters,
                conv=conv,
                norm=norm,
                act=self.act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(y)
        y = conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
            name="conv_end",
        )(y)
        return base - y  # residual-like network


class ResNet(Module):
    """Flax implementation of convolutional network with residual connection.

    Net constructed from sucessive applications of convolution plus batch
    normalization blocks and ending with residual connection (i.e. adding
    the input to the output of the block).

    Args:
        depth: Depth of residual net.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the layers of the block.
            Corresponds to the number of channels in the network
            processing.
        kernel_size: Size of the convolution filters. Default: 3x3.
        strides: Convolution strides. Default: 1x1.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Apply ResNet.

        Args:
            x: The array to be transformed.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The ResNet result.
        """

        residual = x

        # Definition using arguments common to all convolutions.
        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=xavier_normal()
        )

        # Definition using arguments common to all batch normalizations.
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        act = relu

        # Definition and application of ResNet.
        for _ in range(self.depth - 1):
            x = ConvBNBlock(
                self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)

        x = conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
        )(x)
        x = norm()(x)

        return x + residual


class ConvBNNet(Module):
    """Convolution and batch normalization net.

    Net constructed from sucessive applications of convolution plus batch
    normalization blocks. No residual connection.

    Args:
        depth: Depth of net.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the layers of the block.
            Corresponds to the number of channels in the network
            processing.
        kernel_size: Size of the convolution filters. Default: 3x3.
        strides: Convolution strides. Default: 1x1.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Apply ConvBNNet.

        Args:
            x: The array to be transformed.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The ConvBNNet result.
        """
        # Definition using arguments common to all convolutions.
        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=xavier_normal()
        )

        # Definition using arguments common to all batch normalizations.
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        act = relu

        # Definition and application of ConvBNNet.
        for _ in range(self.depth - 1):
            x = ConvBNBlock(
                self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)

        x = conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
        )(x)
        x = norm()(x)

        return x


class UNet(Module):
    """Flax implementation of U-Net model :cite:`ronneberger-2015-unet`.

    Args:
        depth: Depth of U-Net.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the network
            processing.
        kernel_size: Size of the convolution filters. Default: 3x3.
        strides: Convolution strides. Default: 1x1.
        block_depth: Number of processing layers per block. Default: 2.
        window_shape: Window for reduction for pooling and downsampling.
            Default: 2x2.
        upsampling: Factor for expanding. Default: 2.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    block_depth: int = 2
    window_shape: Tuple[int, int] = (2, 2)
    upsampling: int = 2
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Apply U-Net.

        Args:
            x: The array to be transformed.
            train: Flag to differentiate between training and testing stages.

        Returns:
            The U-Net result.
        """
        # Definition using arguments common to all convolutions.
        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=kaiming_normal()
        )

        # Definition using arguments common to all batch normalizations.
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        act = relu

        # Definition of upscaling function.
        upfn = partial(upscale_nn, scale=self.upsampling)

        # Definition and application of U-Net.
        x = ConvBNMultiBlock(
            self.block_depth,
            self.num_filters,
            conv=conv,
            norm=norm,
            act=act,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )(x)
        residual = []
        # going down
        j: int = 1
        for _ in range(self.depth - 1):
            residual.append(x)  # for skip connections
            x = ConvBNPoolBlock(
                2 * j * self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                pool=max_pool,
                kernel_size=self.kernel_size,
                strides=self.strides,
                window_shape=self.window_shape,
            )(x)
            x = ConvBNMultiBlock(
                self.block_depth,
                2 * j * self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)
            j = 2 * j

        # going up
        j = j // 2  # undo last
        res_ind = -1
        for _ in range(self.depth - 1):
            x = ConvBNUpsampleBlock(
                j * self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                upfn=upfn,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)
            # skip connection
            x = jnp.concatenate((residual[res_ind], x), axis=3)
            x = ConvBNMultiBlock(
                self.block_depth,
                j * self.num_filters,
                conv=conv,
                norm=norm,
                act=act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)
            res_ind -= 1
            j = j // 2

        # final conv1x1
        ksz_out = (1, 1)
        x = conv(self.channels, ksz_out, strides=self.strides)(x)

        return x
