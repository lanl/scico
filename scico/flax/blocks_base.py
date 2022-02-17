#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax implementation of different convolutional blocks.
"""

from typing import Any, Callable, Tuple

from flax.linen.module import Module, compact
from flax.linen import relu
from scico.typing import Array

ModuleDef = Any


class ConvBNBlock(Module):
    """Define convolution and batch normalization Flax block.
    Args:
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        norm : Flax module implementing the batch normalization layer to apply.
        act : Flax function defining the activation operation to apply.
        kernel_size : a shape tuple defining the size of the convolution filters. Default: (3, 3).
        strides : a shape tuple defining the size of strides in convolution. Default: (1, 1).
    """

    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable[[Array], Array]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        return self.act(y)


class ConvBlock(Module):
    """Define convolution Flax block.
    Args:
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        act : Flax function defining the activation operation to apply.
        kernel_size : a shape tuple defining the size of the convolution filters. Default: (3, 3).
        strides : a shape tuple defining the size of strides in convolution. Default: (1, 1).
    """

    num_filters: int
    conv: ModuleDef
    act: Callable[[Array], Array]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        return self.act(y)


class ConvBNPoolBlock(Module):
    """Define convolution, batch normalization and pooling Flax block.
    Args:
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        norm : Flax module implementing the batch normalization layer to apply.
        act : Flax function defining the activation operation to apply.
        pool : Flax function defining the pooling operation to apply.
        kernel_size : a shape tuple defining the size of the convolution filters.
        strides : a shape tuple defining the size of strides in convolution.
        window_shape : a shape tuple defining the window to reduce over in the pooling operation.
    """

    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable[[Array], Array]
    pool: Callable[[Array], Array]
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    window_shape: Tuple[int, int]

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        y = self.act(y)
        # 'SAME': pads so as to have the same output shape as input if the stride is 1.
        return self.pool(y, self.window_shape, strides=self.window_shape, padding="SAME")


class ConvBNUpsampleBlock(Module):
    """Define convolution, batch normalization and upsample Flax block.
    Args:
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        norm : Flax module implementing the batch normalization layer to apply.
        act : Flax function defining the activation operation to apply.
        upfn : Flax function defining the upsampling operation to apply.
        kernel_size : a shape tuple defining the size of the convolution filters.
        strides : a shape tuple defining the size of strides in convolution.
    """

    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable[[Array], Array]
    upfn: Callable[[Array], Array]
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(x)
        y = self.norm()(y)
        y = self.act(y)
        return self.upfn(y)


class ConvBNMultiBlock(Module):
    """Block constructed from multiple sucessive applications of :class:`ConvBNBlock`.
    Args:
        num_blocks : number of convolutional batch normalization blocks to apply. Each block has its own parameters for convolution and batch normalization.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        norm : Flax module implementing the batch normalization layer to apply.
        act : Flax function defining the activation operation to apply. Default: relu.
        kernel_size : a shape tuple defining the size of the convolution filters. Default: (3, 3).
        strides : a shape tuple defining the size of strides in convolution. Default: (1, 1).
    """

    num_blocks: int
    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable[[Array], Array] = relu
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        for _ in range(self.num_blocks):
            x = ConvBNBlock(
                self.num_filters,
                conv=self.conv,
                norm=self.norm,
                act=self.act,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(x)

        return x
