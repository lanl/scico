# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax nnx implementation of different convolutional blocks."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
from typing import Callable, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx
from flax.core import Scope  # noqa

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


class ConvBNBlock(nnx.Module):
    """Define convolution, batch normalization and activation Flax nnx block."""

    def __init__(
        self,
        channels_in: int,
        num_filters: int,
        act: Callable[..., ArrayLike],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize convolution, batch normalization and activation block.

        Args:
            channels_in: Number of channels in input signal.
            num_filters: Number of filters in the convolutional layer of the block.
                Corresponds to the number of channels in the output tensor.
            act: Flax function defining the activation operation to apply.
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in
                convolution.
            kernel_init: Flax function for initializing the convolution kernels.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()
        self.act = act

        self.conv = nnx.Conv(
            channels_in,
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            kernel_init=kernel_init(),
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(num_filters, momentum=0.99, epsilon=1e-5, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        inputs: ArrayLike,
    ) -> ArrayLike:
        """Apply convolution followed by normalization and activation.

        Args:
            inputs: The array to be transformed.

        Returns:
            The transformed input.
        """
        outputs = self.norm(self.conv(inputs))
        return self.act(outputs)


class ConvBlock(nnx.Module):
    """Define Flax nnx convolution and activation block."""

    def __init__(
        self,
        channels_in: int,
        num_filters: int,
        act: Callable[..., ArrayLike],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize convolution and activation block.

        Args:
            channels_in: Number of channels in input signal.
            num_filters: Number of filters in the convolutional layer of the
                block. Corresponds to the number of channels in the output
                tensor.
            act: Flax function defining the activation operation to apply.
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in
                convolution.
            kernel_init: Flax function for initializing the convolution kernels.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()
        self.act = act

        self.conv = nnx.Conv(
            channels_in,
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            kernel_init=kernel_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        inputs: ArrayLike,
    ) -> ArrayLike:
        """Apply convolution followed by activation.

        Args:
            inputs: The array to be transformed.

        Returns:
            The transformed input.
        """
        outputs = self.conv(inputs)
        return self.act(outputs)


class ConvBNPoolBlock(nnx.Module):
    """Define convolution, batch normalization and pooling Flax nnx block."""

    def __init__(
        self,
        channels_in: int,
        num_filters: int,
        act: Callable[..., ArrayLike],
        pool: Callable[..., ArrayLike],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        window_shape: Tuple[int, int] = (1, 1),
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize convolution, batch normalization and pooling block.

        Args:
            channels_in: Number of channels in input signal.
            num_filters: Number of filters in the convolutional layer of the
                block. Corresponds to the number of channels in the output
                tensor.
            act: Flax function defining the activation operation to apply.
            pool: Flax function defining the pooling operation to apply.
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in convolution.
            window_shape: A shape tuple defining the window to reduce over in
                the pooling operation.
            kernel_init: Flax function for initializing the convolution kernels.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()
        self.act = act

        self.conv = nnx.Conv(
            channels_in,
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            kernel_init=kernel_init(),
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(num_filters, momentum=0.99, epsilon=1e-5, dtype=dtype, rngs=rngs)

        # 'SAME': pads so as to have the same output shape as input if the stride is 1.
        self.pool = partial(pool, window_shape=window_shape, strides=window_shape, padding="SAME")

    def __call__(
        self,
        inputs: ArrayLike,
    ) -> ArrayLike:
        """Apply convolution followed by normalization, activation and pooling.

        Args:
            inputs: The array to be transformed.

        Returns:
            The transformed input.
        """
        outputs = self.norm(self.conv(inputs))
        outputs = self.act(outputs)
        return self.pool(outputs)


class ConvBNUpsampleBlock(nnx.Module):
    """Define convolution, batch normalization and upsample Flax nnx block."""

    def __init__(
        self,
        channels_in: int,
        num_filters: int,
        act: Callable[..., ArrayLike],
        upfn: Callable[..., ArrayLike],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize convolution, batch normalization and upsample block.

        Args:
            channels_in: Number of channels in input signal.
            num_filters: Number of filters in the convolutional layer of the
                block. Corresponds to the number of channels in the output
                tensor.
            act: Flax function defining the activation operation to apply.
            upfn: Flax function defining the upsampling operation to apply.
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in convolution.
            kernel_init: Flax function for initializing the convolution kernels.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()
        self.act = act

        self.conv = nnx.Conv(
            channels_in,
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            kernel_init=kernel_init(),
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(num_filters, momentum=0.99, epsilon=1e-5, dtype=dtype, rngs=rngs)

        self.upfn = upfn

    def __call__(
        self,
        inputs: ArrayLike,
    ) -> ArrayLike:
        """Apply convolution followed by normalization, activation and upsampling.

        Args:
            inputs: The array to be transformed.

        Returns:
            The transformed input.
        """
        outputs = self.norm(self.conv(inputs))
        outputs = self.act(outputs)
        return self.upfn(outputs)


class ConvBNMultiBlock(nnx.Module):
    """Block constructed from sucessive applications of :class:`ConvBNBlock`."""

    def __init__(
        self,
        channels_in: int,
        num_blocks: int,
        num_filters: int,
        act: Callable[..., ArrayLike],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize convolution, batch normalization and activation multi-block.

        Args:
            channels_in: Number of channels in input signal.
            num_blocks: Number of convolutional batch normalization blocks to
                apply. Each block has its own parameters for convolution
                and batch normalization.
            num_filters: Number of filters in the convolutional layer of the
                block. Corresponds to the number of channels in the output
                tensor.
            act: Flax function defining the activation operation to apply.
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in
                convolution.
            kernel_init: Flax function for initializing the convolution kernels.
            dtype: Output dtype. Default: :attr:`~numpy.float32`.
            rngs: Random generation key.
        """
        super().__init__()

        self.blocks = nnx.Sequential(
            ConvBNBlock(
                channels_in, num_filters, act, kernel_size, strides, kernel_init, dtype, rngs=rngs
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
                for _ in range(num_blocks - 1)
            ],
        )

    def __call__(
        self,
        x: ArrayLike,
    ) -> ArrayLike:
        """Apply sucessive convolution, normalization and activation blocks.

        Apply sucessive blocks, each one composed of convolution
        normalization and activation.

        Args:
            x: The array to be transformed.

        Returns:
            The transformed input.
        """
        return self.blocks(x)


def upscale_nn(x: ArrayLike, scale: int = 2) -> ArrayLike:
    """Nearest neighbor upscale for image batches of shape (N, H, W, C).

    Args:
        x: Input tensor of shape (N, H, W, C).
        scale: Integer scaling factor.

    Returns:
        Output tensor of shape (N, H * scale, W * scale, C).
    """
    s = x.shape
    x = x.reshape((s[0],) + (s[1], 1, s[2], 1) + (s[3],))
    x = jnp.tile(x, (1, 1, scale, 1, scale, 1))
    return x.reshape((s[0],) + (scale * s[1], scale * s[2]) + (s[3],))
