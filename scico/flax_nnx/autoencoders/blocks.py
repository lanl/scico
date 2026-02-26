# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax nnx implementation of different neural network blocks for autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Callable, Sequence, Tuple

from jax.typing import ArrayLike

from flax import nnx
from flax.core import Scope  # noqa

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


class MLP(nnx.Module):
    """Basic definition of a multi layer perceptron (MLP) as a Flax
    nnx block."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        layer_widths: Sequence[int],
        activation_fn: Callable = nnx.relu,
        activate_final: bool = False,
        flatten_first: bool = False,
        batch_norm: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialization of MLP.

        Args:
            dim_in: Dimension of input.
            dim_out: Dimension of output.
            layer_widths: Sequential list with number of neurons per layer
                in the MLP.
            activation_fn: Flax function defining the activation operation
                to apply after each layer.
            activate_final: Flag to indicate if the activation function is
                to be applied after the final layer or not.
            flatten_first: Flag to indicate if the input signal should be flatten
                before passing through the MLP.
            batch_norm: Flag to indicate if batch norm is to be applied or not.
            rngs: Random generation key.
        """
        super().__init__()
        # Store model parameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation_fn = activation_fn
        self.activate_final = activate_final
        self.flatten_first = flatten_first

        # Declare layers
        if batch_norm:
            self.layers = nnx.Sequential(
                nnx.Linear(in_features=dim_in, out_features=layer_widths[0], rngs=rngs),
                *[
                    nnx.Sequential(
                        nnx.BatchNorm(layer_widths[i], rngs=rngs),
                        activation_fn,
                        nnx.Linear(in_features=layer_widths[i], out_features=lyw, rngs=rngs),
                    )
                    for i, lyw in enumerate(layer_widths[1:])
                ],
                nnx.BatchNorm(layer_widths[-1], rngs=rngs),
                activation_fn,
                nnx.Linear(
                    in_features=layer_widths[-1],
                    out_features=dim_out,
                    kernel_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                ),
            )
        else:
            self.layers = nnx.Sequential(
                nnx.Linear(in_features=dim_in, out_features=layer_widths[0], rngs=rngs),
                *[
                    nnx.Sequential(
                        activation_fn,
                        nnx.Linear(in_features=layer_widths[i], out_features=lyw, rngs=rngs),
                    )
                    for i, lyw in enumerate(layer_widths[1:])
                ],
                activation_fn,
                nnx.Linear(in_features=layer_widths[-1], out_features=dim_out, rngs=rngs),
            )

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
        x = self.layers(x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class CNN(nnx.Module):
    """Basic definition of a network with multiple convolutional layers as a Flax
    nnx block."""

    def __init__(
        self,
        channels: int,
        num_filters: Sequence[int],
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        activation_fn: Callable = nnx.relu,
        flatten_final: bool = True,
        kernel_init: Callable = nnx.initializers.kaiming_normal,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialization of CNN.

        Args:
            channels: Number of channels of signal to process.
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
            rngs: Random generation key.
        """
        super().__init__()
        self.flatten_final = flatten_final

        # Declare layers
        self.layers = nnx.Sequential(
            nnx.Conv(
                channels,
                num_filters[0],
                kernel_size=kernel_size,
                strides=strides,
                padding="CIRCULAR",
                use_bias=False,
                kernel_init=kernel_init(),
                rngs=rngs,
            ),
            *[
                nnx.Sequential(
                    activation_fn,
                    nnx.Conv(
                        num_filters[i - i],
                        num_filters[i],
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="CIRCULAR",
                        use_bias=False,
                        kernel_init=kernel_init(),
                        rngs=rngs,
                    ),
                )
                for i, lyf in enumerate(num_filters[1:])
            ],
            activation_fn,
        )
        self.final_conv = nnx.Conv(
            num_filters[-1], channels, kernel_size=(1, 1), use_bias=False, rngs=rngs
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply convolutional layer(s) and activation(s).

        Args:
            x: The array to be transformed.

        Returns:
            The input after being transformed by multiple convolutional
            layers. It has been flatten or has the same number of
            channels as the given input.
        """
        x = self.layers(x)
        x = self.final_conv(x)
        if self.flatten_final:
            # Flatten output (e.g. for latent representation).
            x = x.reshape((x.shape[0], -1))
        return x
