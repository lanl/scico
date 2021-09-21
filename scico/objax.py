# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Convolutional neural network models implemented in ObJax."""

from typing import Callable, Optional, Sequence, Union

import objax
from objax.constants import ConvPadding
from objax.typing import ConvPaddingInt, JaxArray

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""


def conv_args(
    w_init: Callable = objax.nn.init.kaiming_normal,
    padding: Optional[Union[ConvPadding, str, ConvPaddingInt]] = ConvPadding.SAME,
):
    """Return list of arguments which are common to all convolutions.

    Args:
        w_init: function to initialize weights for convolution layers. Default: Kaiming He
           normal initializer.
        padding: type of padding to apply. Default: ConvPadding.SAME: input tensor is
           zero-padded by ⌊(k−1)/2⌋ for left and up sides and ⌊k/2⌋ for right and down sides.

    Returns:
        Dictionary with common convolution arguments.
    """
    return dict(w_init=w_init, use_bias=False, padding=padding)


class ConvBN_Block(objax.Module):
    r"""Convolution and batch normalization Objax block."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        stride: Union[int, Sequence[int]],
        normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
        activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu,
    ):

        r"""Initialize a :class:`ConvBN_Block` object.

        Args:
            in_channels : number of channels of input tensor.
            num_filters : number of filters in the convolutional layer of the block.
               Corresponds to the number of channels in the output tensor.
            kernel_size : size of the convolution filters.
            stride : convolution strides.
            normalization_fn : class of batch normalization to apply. Default:
               :class:`objax.nn.BatchNorm2D`.
            activation_fn : class of activation function to apply. Default:
               :func:`objax.functional.relu`.
        """
        # 2D convolution on a 4D-input batch of shape (N,C,H,W).
        self.conv = objax.nn.Conv2D(
            in_channels, num_filters, k=kernel_size, strides=stride, **conv_args()
        )
        self.norm = normalization_fn(num_filters)
        self.activation_fn = activation_fn

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        r"""Forward pass (evaluation) of module.

        Args:
            x : input to module. The expected shape of x is: (batch, channels, height, width).
            training : flag to specify if the forward pass is executed on training stage
               (i.e. when updates to the parameters of batch normalization layers are performed).
        """
        x = self.conv(x)
        x = self.norm(x, training)
        x = self.activation_fn(x)

        return x


class DnCNN_Net(objax.Module):
    r"""Objax implementation of DnCNN :cite:`zhang-2017-dncnn`.

    Objax implementation of the convolutional neural network architecture for denoising
    described in :cite:`zhang-2017-dncnn`."""

    def __init__(
        self,
        depth: int,
        in_channels: int,
        num_filters: int,
        kernel_size: int = 3,
        stride: Union[int, Sequence[int]] = 1,
        normalization_fn: Callable[..., objax.Module] = objax.nn.BatchNorm2D,
        activation_fn: Callable[[JaxArray], JaxArray] = objax.functional.relu,
    ):
        r"""Initialize a :class:`DnCNN_Net` object.

        Args:
            depth : number of layers in the neural network.
            in_channels : number of channels of input tensor.
            num_filters : number of filters in the convolutional layer of the block.
               Corresponds to the number of channels in the output tensor.
            kernel_size : size of the convolution filters. Default: 3.
            stride : convolution strides. Default: 1.
            normalization_fn : class of batch normalization to apply. Default:
               :class:`objax.nn.BatchNorm2D`.
            activation_fn : class of activation function to apply. Default:
               :func:`objax.functional.relu`.
        """
        self.pre_conv = objax.nn.Sequential(
            [
                objax.nn.Conv2D(
                    in_channels, num_filters, k=kernel_size, strides=stride, **conv_args()
                ),
                activation_fn,
            ]
        )

        self.layers = objax.nn.Sequential([])
        for i in range(depth - 2):
            self.layers.append(
                ConvBN_Block(
                    num_filters, num_filters, kernel_size, stride, normalization_fn, activation_fn
                )
            )

        self.post_conv = objax.nn.Conv2D(
            num_filters, in_channels, k=kernel_size, strides=stride, **conv_args()
        )

    def load_weights(self, filename: str):
        """Load trained model weights.

        Args:
            filename : name of file where parameters for trained model have been stored
        """
        objax.io.load_var_collection(filename, self.vars())

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        r"""Forward pass (evaluation) of model.

        Note that this model is a residual learning type model.

        Args:
            x : input to neural network. The expected shape of x is: (batch, channels,
                height, width).
            training : flag to specify if the forward pass is executed on training stage
                (i.e. when updates to the parameters of batch normalization layers are performed).
        """
        base = x
        x = self.pre_conv(x)

        for ly in self.layers:
            x = ly(x, training)

        x = self.post_conv(x)
        # residual-like output
        return base - x
