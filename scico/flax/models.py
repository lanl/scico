#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax implementation of different convolutional nets.
"""

from functools import partial
from typing import Any, Callable, Tuple

from flax.linen.module import Module, compact
from flax.linen import Conv, BatchNorm, relu, max_pool
from flax.linen.initializers import kaiming_normal, xavier_normal

import jax.numpy as jnp

from scico.typing import Array

import blocks_base as blk


ModuleDef = Any


class DnCNNNet(Module):
    r"""Flax implementation of DnCNN :cite:`zhang-2017-dncnn`.

    Flax implementation of the convolutional neural network (CNN)
    architecture for denoising described in :cite:`zhang-2017-dncnn`.

    Args:
        depth : Number of layers in the neural network.
        channels : number of channels in input tensor.
        num_filters : number of filters in the convolutional layers.
        kernel_size : a shape tuple defining the size of the convolution filters. Default: (3, 3).
        strides : a shape tuple defining the size of strides in convolution. Default: (1, 1).
        act : Flax function defining the activation operation to apply. Default: relu.
        dtype : Output dtype. Default: `jnp.float32`.
    """
    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    act: Callable = relu
    dtype: Any = jnp.float32

    @compact
    def __call__(self, inputs: Array, train: bool = True) -> Array:
        """Apply DnCNN denoiser.

        Args:
            inputs: The nd-array to be transformed.

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
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        # Definition and application of DnCNN model.
        base = inputs
        y = conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(input)
        y = self.act(y)
        for _ in range(self.depth - 2):
            y = blk.ConvBNBlock(
                self.num_filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                conv=conv,
                norm=norm,
                act=self.act,
            )(y)
        y = conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
        )(y)
        return base - y  # residual-like network


class UNet(Module):
    """Flax implementation of UNet model.

    Args:
        depth : depth of U-net.
        channels : number of channels of input tensor.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the network processing.
        kernel_size : size of the convolution filters. Default: 3x3.
        strides : convolution strides. Default: 1x1.
        block_depth : number of processing layers per block. Default: 2.
        window_shape : window for reduction for pooling and downsampling. Default: 2x2.
        upsampling : factor for expanding. Default: 2.
        dtype : class of data to handle. Default: `jnp.float32`.
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

        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=kaiming_normal()
        )
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.999,
            epsilon=1e-6,
            dtype=self.dtype,
        )
        upfn = partial(upscale_nn, scale=self.upsampling)

        x = blk.ConvBNMultiBlock(
            self.block_depth, self.num_filters, self.kernel_size, self.strides, conv=conv, norm=norm
        )(x)

        residual = []

        # going down
        j: int = 1
        for _ in range(self.depth - 1):
            residual.append(x)  # for skip connections
            x = blk.ConvBNPoolBlock(
                2 * j * self.num_filters,
                self.kernel_size,
                self.strides,
                self.window_shape,
                conv=conv,
                norm=norm,
                act=relu,
                pool=max_pool,
            )(x)
            x = blk.ConvBNMultiBlock(
                self.block_depth,
                2 * j * self.num_filters,
                self.kernel_size,
                self.strides,
                conv=conv,
                norm=norm,
            )(x)
            j = 2 * j

        # going up
        j = j // 2  # undo last
        res_ind = -1
        for _ in range(self.depth - 1):
            x = blk.ConvBNUpsampleBlock(
                j * self.num_filters,
                self.kernel_size,
                self.strides,
                conv=conv,
                norm=norm,
                act=relu,
                upfn=upfn,
            )(x)
            # skip connection
            x = jnp.concatenate((residual[res_ind], x), axis=3)
            x = blk.ConvBNMultiBlock(
                self.block_depth,
                j * self.num_filters,
                self.kernel_size,
                self.strides,
                conv=conv,
                norm=norm,
            )(x)
            res_ind -= 1
            j = j // 2

        # final conv1x1
        ksz_out = (1, 1)
        x = conv(self.channels, ksz_out, strides=self.strides)(x)

        return x


class ResNet(Module):
    """Net constructed from sucessive applications of provided block and ending with residual connection (i.e. adding the input to the output of the block).

    Args:
        depth : depth of residual net.
        channels : number of channels of input tensor.
        num_filters : number of filters in the layers of the block. Corresponds to the number of channels in the network processing.
        kernel_size : size of the convolution filters. Default: 3x3.
        strides : convolution strides. Default: 1x1.
        block_cls : processing block to apply. Default: :class:`ConvBNBlock`.
        dtype : class of data to handle. Default: `jnp.float32`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    block_cls: ModuleDef = blk.ConvBNBlock
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:

        residual = x

        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=xavier_normal()
        )
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.999,
            epsilon=1e-6,
            dtype=self.dtype,
        )
        act = relu

        for _ in range(self.depth - 1):
            x = self.block_cls(
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
    """Net constructed from sucessive applications of convolutional and batch normalization blocks.

    Args:
        depth : depth of residual net.
        channels : number of channels of input tensor.
        num_filters : number of filters in the layers of the block. Corresponds to the number of channels in the network processing.
        kernel_size : size of the convolution filters. Default: 3x3.
        strides : convolution strides. Default: 1x1.
        dtype : class of data to handle. Default: `jnp.float32`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:

        residual = x

        conv = partial(
            Conv, use_bias=False, padding="CIRCULAR", dtype=self.dtype, kernel_init=xavier_normal()
        )
        norm = partial(
            BatchNorm,
            use_running_average=not train,
            momentum=0.999,
            epsilon=1e-6,
            dtype=self.dtype,
        )
        act = relu

        for _ in range(self.depth - 1):
            x = blk.ConvBNBlock(
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


class ODPDnBlock(Module):
    """Unrolled optimization with deep priors block.

    Args:
        depth : depth of block.
        channels : number of channels of input tensor.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size : size of the convolution filters. Default: 3x3.
        strides : convolution strides. Default: 1x1.
        dtype : type of signal to process. Default: jnp.float32.
        alpha_ini : initial weight of fidelity term. Default: 0.2.
    """

    depth: int
    channels: int
    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32
    alpha_ini: float = 0.2

    @compact
    def __call__(self, x: Array, y: Array, train: bool = True) -> Array:
        def alpha_init_wrap(rng, shape, dtype=self.dtype):
            return jnp.ones(shape, dtype) * self.alpha_ini

        alpha = self.param("alpha", alpha_init_wrap, (1,))

        resnet = ResNet(
            self.depth,
            self.channels,
            self.num_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
        )
        x = (resnet(x, train) + y * alpha) / (1.0 + alpha)

        return x


class ODPNet(Module):
    """Net constructed from sucessive applications of ODP blocks.

    Args:
        depth : depth of ODP net. Default = 1.
        channels : number of channels of input tensor.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        block_depth : depth of blocks.
        c0 : initial value of alpha in initial layer. Default: 0.5.
        odp_block : processing block to apply. Default :class:`ODPDnBlock`.
        dtype : type of signal to process. Default: jnp.float32.
    """

    depth: int
    channels: int
    num_filters: int
    block_depth: int
    c0: float = 0.5
    odpblock: ModuleDef = ODPDnBlock
    dtype: Any = jnp.float32

    @compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        block = partial(
            self.odpblock,
            depth=self.block_depth,
            channels=self.channels,
            num_filters=self.num_filters,
            dtype=self.dtype,
        )
        y = x
        alpha0_i = self.c0

        for i in range(self.depth):
            x = block(alpha_ini=alpha0_i)(x, y, train)
            alpha0_i /= 2.0

        return x


def upscale_nn(x: Array, scale: int = 2) -> Array:
    """Nearest neighbor upscale for image batches of shape (N, H, W, C).
    Args:
        x: input tensor of shape (N, H, W, C).
        scale: integer scaling factor.
    Returns:
        Output tensor of shape (N, H * scale, W * scale, C).
    """
    s = x.shape
    x = x.reshape((s[0],) + (s[1], 1, s[2], 1) + (s[3],))
    x = jnp.tile(x, (1, 1, scale, 1, scale, 1))
    return x.reshape((s[0],) + (scale * s[1], scale * s[2]) + (s[3],))
