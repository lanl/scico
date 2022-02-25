#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flax implementation of different convolutional blocks.
"""

from typing import Any, Callable, Tuple

import jax.numpy as jnp

from flax.linen.module import Module, compact, _Sentinel
from flax.core import Scope  # noqa
from scico.typing import Array

# The imports of Scope and _Sentinel (above)
# are required to silence "cannot resolve forward reference"
# warnings when building sphinx api docs.

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
    act: Callable[..., Array]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        inputs: Array,
    ) -> Array:
        """Apply convolution followed by normalization and activation.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(inputs)
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
    act: Callable[..., Array]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        inputs: Array,
    ) -> Array:
        """Apply convolution followed by activation.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(inputs)
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
    act: Callable[..., Array]
    pool: Callable[..., Array]
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    window_shape: Tuple[int, int]

    @compact
    def __call__(
        self,
        inputs: Array,
    ) -> Array:
        """Apply convolution followed by normalization, activation and pooling.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(inputs)
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
    act: Callable[..., Array]
    upfn: Callable[..., Array]
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]

    @compact
    def __call__(
        self,
        inputs: Array,
    ) -> Array:
        """Apply convolution followed by normalization, activation and upsampling.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        y = self.conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
        )(inputs)
        y = self.norm()(y)
        y = self.act(y)
        return self.upfn(y)


class ConvBNMultiBlock(Module):
    """Block constructed from sucessive applications of :class:`ConvBNBlock`.

    Args:
        num_blocks : number of convolutional batch normalization blocks to apply. Each block has its own parameters for convolution and batch normalization.
        num_filters : number of filters in the convolutional layer of the block. Corresponds to the number of channels in the output tensor.
        conv : Flax module implementing the convolution layer to apply.
        norm : Flax module implementing the batch normalization layer to apply.
        act : Flax function defining the activation operation to apply.
        kernel_size : a shape tuple defining the size of the convolution filters. Default: (3, 3).
        strides : a shape tuple defining the size of strides in convolution. Default: (1, 1).
    """

    num_blocks: int
    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable[..., Array]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @compact
    def __call__(
        self,
        x: Array,
    ) -> Array:
        """Apply sucessive blocks, each one composed of convolution normalization and activation.

        Args:
            x: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
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
