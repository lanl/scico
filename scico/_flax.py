# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Convolutional neural network models implemented in Flax."""

from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp

from flax import linen as nn
from flax import serialization
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa

from scico.blockarray import BlockArray
from scico.typing import JaxArray

# The imports of Scope and _Sentinel (above) and the definition of Module
# (below) are required to silence "cannot resolve forward reference"
# warnings when building sphinx api docs.


Module = nn.module.Module
ModuleDef = Any
Array = Any
PyTree = Any


class ConvBNBlock(nn.Module):
    r"""Define convolution and batch normalization Flax block.

    Attributes:
        num_filters: Number of filters in the convolutional layer
          of the block.
        conv: Class of convolution to apply.
        norm: Class of batch normalization to apply.
        act: Class of activation function to apply.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        stride: Convolution strides. Default: (1, 1).
    """

    num_filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
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


class DnCNNNet(nn.Module):
    r"""Flax implementation of DnCNN :cite:`zhang-2017-dncnn`.

    Flax implementation of the convolutional neural network (CNN)
    architecture for denoising described in :cite:`zhang-2017-dncnn`.

    Attributes:
        depth: Number of layers in the neural network.
        channels: Number of channels of input tensor.
        num_filters: Number of filters in the convolutional layers.
        kernel_size: Size of the convolution filters. Default: (3, 3).
        strides: Convolution strides. Default: (1, 1).
        dtype: Output dtype. Default: `jnp.float32`.
        act: Class of activation function to apply. Default: `nn.relu`.
    """

    depth: int
    channels: int
    num_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        train: bool = True,
    ) -> Array:
        """Apply DnCNN denoiser.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The denoised input.
        """
        # Definition using arguments common to all convolutions.
        conv = partial(
            nn.Conv,
            use_bias=False,
            dtype=self.dtype,
        )
        # Definition using arguments common to all batch normalizations.
        norm = partial(
            nn.BatchNorm,
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
            name="conv_start",
        )(inputs)
        y = self.act(y)
        for _ in range(self.depth - 2):
            y = ConvBNBlock(
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
            name="conv_end",
        )(y)
        return base - y  # residual-like network


def load_weights(filename: str):
    """Load trained model weights.

    Args:
        filename: Name of file containing parameters for trained model.
    """
    with open(filename, "rb") as data_file:
        bytes_input = data_file.read()

    variables = serialization.msgpack_restore(bytes_input)

    return variables


class FlaxMap:
    r"""A trained flax model."""

    def __init__(self, model: Callable[..., nn.Module], variables: PyTree):
        r"""Initialize a :class:`FlaxMap` object.

        Args:
            model: Flax model to apply.
            variables: Parameters and batch stats of trained model.
        """
        self.model = model
        self.variables = variables
        super().__init__()

    def __call__(self, x: JaxArray) -> JaxArray:
        r"""Apply trained flax model.

        Args:
            x: Input array.

        Returns:
            Output of flax model.
        """
        if isinstance(x, BlockArray):
            raise NotImplementedError

        # Add singleton to input as necessary:
        #   scico typically works with (H x W) or (H x Wx C) arrays
        #   flax expects (K x H x W x C) arrays
        #   H: spatial height  W: spatial width
        #   K: batch size  C: channel size
        x_shape = x.shape
        if x.ndim == 2:
            x = x.reshape((1,) + x.shape + (1,))
        elif x.ndim == 3:
            x = x.reshape((1,) + x.shape)
        y = self.model.apply(self.variables, x, train=False, mutable=False)
        return y.reshape(x_shape)
