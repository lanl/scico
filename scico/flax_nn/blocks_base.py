#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax implementation of different convolutional blocks.
"""

from typing import Any, Callable, Tuple

from flax import linen as nn

ModuleDef = Any
Array = Any


class ConvBNBlock(nn.Module):
    """Convolutional Batch Normalization block.
    Args:
        filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : flax module implementing convolution operation.
        norm : flax module implementing batch normalization.
        act : flax activation function.
        kernel_size : size of the convolution filters. Default: 3x3.
        strides : convolution strides. Default: 1x1.
    """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: Array,) -> Array:
        y = self.conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        return self.act(y)


class ConvBlock(nn.Module):
    """Convolutional Batch block.
    Args:
        filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size : size of the convolution filters.
        strides : convolution strides.
        conv : flax module implementing convolution operation.
        act : flax activation function.
    """
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    conv: ModuleDef
    act: Callable

    @nn.compact
    def __call__(self, x: Array,) -> Array:
        y = self.conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        return self.act(y)


class ConvBNPoolBlock(nn.Module):
    """Convolutional Batch Normalization pooling block.

    Args:
        filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size : size of the convolution filters.
        strides : convolution strides.
        window_shape : a shape tuple defining the window to reduce over.
        conv : flax module implementing convolution operation.
        norm : flax module implementing batch normalization.
        act : flax activation function.
        pool : flax pooling function.
    """
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    window_shape: Tuple[int, int]
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    pool: Callable


    @nn.compact
    def __call__(self, x: Array,) -> Array:
        y = self.conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        y = self.act(y)
        return self.pool(y, self.window_shape, strides=self.window_shape, padding='SAME')


class ConvBNUpsampleBlock(nn.Module):
    """Convolutional Batch Normalization upsample block.

    Args:
        filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size : size of the convolution filters.
        strides : convolution strides.
        conv : flax module implementing convolution operation.
        norm : flax module implementing batch normalization.
        act : flax activation function.
        upfn : upsampling function.
    """
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    upfn: Callable

    @nn.compact
    def __call__(self, x: Array,) -> Array:
        y = self.conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        y = self.act(y)
        return self.upfn(y)


class ConvBNMultiBlock(nn.Module):
    """Block constructed from multiple sucessive applications of :class:`ConvBNBlock`.

    Args:
        blocks : number of blocks to apply.
        filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        kernel_size : size of the convolution filters.
        strides : convolution strides.
        conv : flax module implementing convolution operation.
        norm : flax module implementing batch normalization.
        act : flax activation function. Default: relu.
        dtype : type of signal to process. Default: jnp.float32.
    """
    blocks: int
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x: Array,) -> Array:
        for _ in range(self.blocks):
            x = ConvBNBlock(
                self.filters,
                conv=self.conv,
                norm=self.norm,
                act=self.act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)

        return x

