# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementation of different neural network blocks for
autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Callable, Sequence, Tuple

from jax.typing import ArrayLike

import flax.linen as nn
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from scico.flax.blocks import upscale_nn

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


class MLP(nn.Module):
    """Basic definition of a multi layer perceptron (MLP) as a Flax
    block.

    Args:
        layer_widths: Sequential list with number of neurons per layer
            in the MLP.
        activation_fn: Flax function defining the activation operation
            to apply after each layer.
        activate_final: Flag to indicate if the activation function is
            to be applied after the final layer or not.
    """

    layer_widths: Sequence[int]
    activation_fn: Callable = nn.relu
    activate_final: bool = False
    flatten_first: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply fully connected (i.e. dense) layer(s) and activation(s).

        Args:
            x: The array to be transformed.

        Returns:
            The input after being transformed by the multiple layers
            of the MLP.
        """
        if self.flatten_first:
            # Flatten input (e.g. for latent representation).
            x = x.reshape((x.shape[0], -1))
        for layer_width in self.layer_widths[:-1]:
            x = self.activation_fn(nn.Dense(layer_width)(x))
        x = nn.Dense(self.layer_widths[-1])(x)
        if self.activate_final:
            x = self.activation_fn(x)

        return x


class CNN(nn.Module):
    """Basic definition of a network with multiple convolutional layers
       as a Flax block.

    The output can be returned as a flatten array preserving only the
    batch component (i.e. the first component). All the layers use the
    same specified kernel size and stride, use a circular padding, and
    do not use bias.

    Args:
        num_filters: Sequential list with number of filters in each
            convolutional layer of the block.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in
            convolution.
        activation_fn: Flax function defining the activation operation
            to apply after each layer.
        flatten_final: Flag to indicate if the output should be returned
            as a flattened array (preserving batch dimension). If not,
            the output is mapped back to the number of channels of the
            input signal.
    """

    num_filters: Sequence[int]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.relu
    flatten_final: bool = True

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply convolutional layer(s) and activation(s).

        Args:
            x: The array to be transformed.

        Returns:
            The input after being transformed by multiple convolutional
            layers. It has been flatten or has the same number of
            channels as the given input.
        """

        # CNN layers.
        for nfilters in self.num_filters:
            x = nn.Conv(
                nfilters, self.kernel_size, strides=self.strides, use_bias=False, padding="CIRCULAR"
            )(x)
            x = self.activation_fn(x)

        if self.flatten_final:
            # Flatten output (e.g. for latent representation).
            x = x.reshape((x.shape[0], -1))

        return x


class ConvPoolBlock(nn.Module):
    """Define convolution and pooling Flax block.

    Args:
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in convolution.
        activation_fn: Flax function defining the activation operation to apply.
        pooling_fn: Flax function defining the pooling operation to apply.
        window_shape: A shape tuple defining the window to reduce over in
            the pooling operation.
    """

    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu
    pooling_fn: Callable = nn.max_pool
    window_shape: Tuple[int, int] = (2, 2)

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply convolution followed by activation and pooling.

        Args:
            inputs: The array to be transformed.

        Returns:
            The transformed input.
        """
        x = nn.Conv(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
            use_bias=False,
            padding="CIRCULAR",
        )(x)
        x = self.activation_fn(x)
        x = self.pooling_fn(x, self.window_shape, strides=self.window_shape, padding="SAME")

        return x


class ConvUpsampleBlock(nn.Module):
    """Define convolution and upsample Flax block.

    Args:
        num_filters: Number of filters in the convolutional layer of the
            block. Corresponds to the number of channels in the output
            tensor.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in convolution.
        activation_fn: Flax function defining the activation operation to apply.
        upsampling_scale: Integer scaling factor.
    """

    num_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu
    upsampling_scale: int = 2

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = nn.ConvTranspose(
            self.num_filters,
            self.kernel_size,
            strides=self.strides,
            use_bias=False,
            padding="CIRCULAR",
        )(x)
        x = self.activation_fn(x)
        x = upscale_nn(x, self.upsampling_scale)

        return x
