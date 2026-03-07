# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax nnx implementations of autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Callable, Optional, Sequence, Tuple

from numpy import prod

from jax.typing import ArrayLike

from flax import nnx
from flax.core import Scope  # noqa
from scico.flax_nnx.autoencoders.blocks import CNN, MLP, CTpNN

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


class Encoder(nnx.Module):
    """Generic encoder definition in Flax NNX."""

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize Encoder.

        Args:
            kwargs: Keyword arguments.
        """
        super().__init__()

    @property
    def latent_dim(self):
        """Expose the latent dimension attribute as a property (getter)."""
        return self.dim_latent

    def __call__(self, x: ArrayLike, *args) -> ArrayLike:
        """Apply encoder.

        Args:
            x: The array to be encoded.
            args: Positional arguments.

        Returns:
            The encoded array.
        """
        raise NotImplementedError


# Create an alias
Decoder = Encoder


class AutoEncoder(nnx.Module):
    """Basic definition of an autoencoder network as a Flax nnx model."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
    ):
        """Initialization of autoencoder.
        Args:
            encoder: Encoder module in Flax nnx.
            decoder: Decoder module in Flax nnx.
        """
        super().__init__()
        # Store submodules
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: ArrayLike) -> ArrayLike:
        """Apply encoder module.

        Args:
            x: The array to be encoded.

        Returns:
            The encoded representation.
        """
        x = self.encoder(x)
        return x

    def decode(self, x: ArrayLike):
        """Apply decoder module.

        Args:
            x: The array to be decoded.

        Returns:
            The decoded representation.
        """
        x = self.decoder(x)
        return x

    def __call__(self, x: ArrayLike, latent_rep: bool = False) -> Tuple[ArrayLike, ArrayLike]:
        """Apply sequence of encoder and decoder modules.

        Args:
            x: The array to be autoencoded.
            latent_rep: Flag to indicate if the latent representations is to be
                returned.

        Returns:
            The output of the autoencoder module and the encoded
            representation (if requested).
        """
        y = self.encode(x)
        x = self.decode(y)
        if latent_rep:
            return x, y
        return x


class MLPEncoder(Encoder):
    """Encoder using multi-layer perceptron (MLP) submodule."""

    def __init__(
        self,
        dim_in: int,
        widths_encoder: Sequence[int],
        dim_latent: int,
        activation_fn: Callable = nnx.leaky_relu,
        batch_norm: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize MLPEncoder.

        Args:
            dim_in: Dimension of input signal.
            widths_encoder: Sequential list with number of neurons per layer
                in the MLP.
            dim_latent: Latent dimension of encoder.
            activation_fn: Flax function defining the activation operation
                to apply after each layer (except output layer).
            batch_norm: Flag to indicate if batch norm is to be applied or not.
            rngs: Random generation key.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.flat_latent = True  # For sample generation

        self.mlp = MLP(
            dim_in,
            dim_latent,
            widths_encoder,
            activation_fn,
            activate_final=False,
            flatten_first=True,
            batch_norm=batch_norm,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply dense encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The encoded array.
        """
        return self.mlp(x)


class MLPDecoder(Decoder):
    """Decoder using multi-layer perceptron (MLP) submodule."""

    def __init__(
        self,
        dim_latent: int,
        widths_decoder: Sequence[int],
        shape_out: Tuple[int],
        activation_fn: Callable = nnx.leaky_relu,
        batch_norm: bool = False,
        reshape_final: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize MLPDecoder.

        Args:
            dim_latent: Latent dimension to decode from.
            widths_decoder: Sequential list with number of neurons per layer
                in the decoder MLP. An additional properly sized layer is
                added if reshape final is set to ``True``.
            shape_out: Tuple (height, width, channel) of image to decode (if
                reshape requested).
            activation_fn: Flax function defining the activation operation
                to apply after each layer (except output layer).
            batch_norm: Flag to indicate if batch norm is to be applied or not.
            reshape_final: Flag to indicate if the output should be reshaped
                before returning.
            rngs: Random generation key.
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.reshape_final = reshape_final
        self.shape_out = shape_out
        dim_out = int(prod(shape_out))

        if self.reshape_final:  # Restore specific shape
            all_widths_decoder = list(widths_decoder)
            all_widths_decoder.append(dim_out)
        else:
            all_widths_decoder = widths_decoder

        self.mlp = MLP(
            dim_latent,
            dim_out,
            all_widths_decoder,
            activation_fn,
            activate_final=False,
            batch_norm=batch_norm,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply dense encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The encoded array.
        """
        x = self.mlp(x)
        if self.reshape_final:
            x = x.reshape((x.shape[0],) + self.shape_out)
        return x


def MLPAutoEncoder(
    dim_in: int,
    widths_encoder: Tuple[int],
    dim_latent: int,
    widths_decoder: Tuple[int],
    shape_out: Tuple[int],
    activation_fn: Callable = nnx.leaky_relu,
    batch_norm: bool = False,
    rngs: nnx.Rngs = nnx.Rngs(0),
):
    """Function to construct autoencoder network using multi-layer
    perceptron (MLP), i.e. dense layers.

    Output is reshaped to given output shape via a properly sized layer
    added automatically to the tuple of the decoder widths.

    Args:
        dim_in: Dimension of input signal.
        widths_encoder: List with number of neurons per layer in the
            MLP encoder.
        dim_latent: Latent dimension of encoder.
        widths_decoder: List with number of neurons per layer in the
            MLP decoder.
        shape_out: Tuple (height, width, channels) of signal to decode
            (if reshape requested).
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
        batch_norm: Flag to indicate if batch norm is to be applied or not.
        rngs: Random generation key.

    Returns:
        Autoencoder model with the specified architecture.
    """
    encoder = MLPEncoder(
        dim_in,
        widths_encoder,
        dim_latent,
        activation_fn,
        batch_norm,
        rngs=rngs,
    )

    decoder = MLPDecoder(
        dim_latent,
        widths_decoder,
        shape_out,
        activation_fn,
        batch_norm,
        reshape_final=True,
        rngs=rngs,
    )

    return AutoEncoder(encoder, decoder)


class ConvEncoder(Encoder):
    """Encoder using convolutional neural network (CNN) submodule."""

    def __init__(
        self,
        shape_in: Tuple[int],
        channels: int,
        filters_encoder: Sequence[int],
        flat_latent: bool = False,
        dim_latent: Optional[int] = None,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (2, 2),
        activation_fn: Callable = nnx.leaky_relu,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize ConvEncoder.

        Args:
            shape_in: Tuple (height, width) of signal to encode.
            channels: Number of channels of signal to encode.
            filters_encoder: List with number of filters per layer in the
                CNN encoder.
            flat_latent: Flag to specify if the latent representation should be flatten.
            dim_latent: Latent dimension of encoder (only used if flag for flattening is
                true and a meanigful value is provided).
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in
                convolution.
            activation_fn: Flax function defining the activation operation
                to apply after each layer (except output layer).
            rngs: Random generation key.
        """
        super().__init__()
        self.cnn = CNN(
            channels,
            filters_encoder,
            kernel_size,
            strides,
            activation_fn,
            flatten_final=flat_latent,
            conv_final=False,
            rngs=rngs,
        )
        if strides[0] == 2:
            divisor = len(filters_encoder) ** 2
        elif strides[0] == 1:
            divisor = 1

        d0 = shape_in[0] // divisor
        d1 = shape_in[1] // divisor
        self.shape_latent = (d0, d1, filters_encoder[-1])
        self.flat_latent = flat_latent
        if flat_latent and dim_latent is not None:
            size_pre_latent = int(prod(self.shape_latent))
            self.linear_latent = nnx.Linear(size_pre_latent, dim_latent, rngs=rngs)
            self.dim_latent = dim_latent
        else:
            self.dim_latent = int(prod(self.shape_latent))
            self.linear_latent = nnx.nn.activations.identity

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.linear_latent(self.cnn(x))


class ConvDecoder(Decoder):
    """Decoder using convolutional layers.

    All the layers use the same specified kernel size and stride, use
    circular padding, and do not use bias.
    """

    def __init__(
        self,
        filters_decoder: Sequence[int],
        shape_latent: Tuple[int],
        channels: int,
        flat_latent: bool = False,
        dim_latent: Optional[int] = None,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (2, 2),
        activation_fn: Callable = nnx.leaky_relu,
        batch_norm=False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialize ConvDecoder model.

        Args:
            filters_decoder: List with number of filters per layer in the
                convolutional decoder.
            shape_latent: Tuple (height, width, channels) of signal before flattening
                for obtaining latent representation.
            channels: Number of channels of signal to decode.
            flat_latent: Flag to specify if the latent representation is flat.
            dim_latent: Latent dimension (only used if flag for flattening is
                true and a meanigful value is provided).
            kernel_size: A shape tuple defining the size of the convolution
                filters.
            strides: A shape tuple defining the size of strides in
                convolution.
            activation_fn: Flax function defining the activation operation
                to apply after each layer (except output layer).
            batch_norm: Flag to indicate if batch norm is to be applied or not.
            rngs: Random generation key.
        """
        super().__init__()
        self.shape_latent = shape_latent
        self.flat_latent = flat_latent

        if flat_latent and dim_latent is not None:
            len_latent = int(prod(shape_latent))
            self.initial_layer = nnx.Linear(dim_latent, len_latent, rngs=rngs)
            self.dim_latent = dim_latent
        else:
            self.initial_layer = nnx.nn.activations.identity
            self.dim_latent = int(prod(shape_latent))

        self.ctpnn = CTpNN(
            shape_latent[-1],
            channels,
            filters_decoder,
            kernel_size,
            strides,
            activation_fn,
            batch_norm=batch_norm,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = self.initial_layer(x)
        if self.flat_latent:
            x = x.reshape((x.shape[0],) + self.shape_latent)
        return self.ctpnn(x)


def ConvAutoEncoder(
    shape_in: Tuple[int],
    channels: int,
    filters_encoder: Sequence[int],
    filters_decoder: Sequence[int],
    flat_latent: bool = False,
    dim_latent: Optional[int] = None,
    kernel_size_encoder: Tuple[int, int] = (3, 3),
    strides_encoder: Tuple[int, int] = (2, 2),
    activation_fn_encoder: Callable = nnx.leaky_relu,
    kernel_size_decoder: Tuple[int, int] = (3, 3),
    strides_decoder: Tuple[int, int] = (2, 2),
    activation_fn_decoder: Callable = nnx.leaky_relu,
    batch_norm: bool = False,
    rngs: nnx.Rngs = nnx.Rngs(0),
):
    """Function to construct autoencoder network using convolutional layers.

    Args:
        shape_in: Tuple (height, width) of signal to encode.
        channels: Number of channels of signal to encode.
        filters_encoder: List with number of filters per layer in the
                CNN encoder.
        filters_decoder: List with number of filters per layer in the
                convolutional decoder.
        flat_latent: Flag to specify if the latent representation should be flatten.
        dim_latent: Latent dimension of encoder (only used if flag for flattening is
            true and a meanigful value is provided).
        kernel_size_encoder: A shape tuple defining the size of the convolution
                filters for encoder.
        strides_encoder: A shape tuple defining the size of strides in
                convolution for encoder.
        activation_fn_encoder: Flax function defining the activation operation
            to apply after each layer (except output layer) in encoder.
        kernel_size_decoder: A shape tuple defining the size of the convolution
                filters for encoder.
        strides_decoder: A shape tuple defining the size of strides in
                convolution for encoder.
        activation_fn_decoder: Flax function defining the activation operation
            to apply after each layer (except output layer) in encoder.
        batch_norm: Flag to indicate if batch norm is to be applied or not.
        rngs: Random generation key.

    Returns:
        Autoencoder model with the specified architecture.
    """
    encoder = ConvEncoder(
        shape_in,
        channels,
        filters_encoder,
        flat_latent,
        dim_latent,
        kernel_size_encoder,
        strides_encoder,
        activation_fn=activation_fn_encoder,
        rngs=rngs,
    )

    decoder = ConvDecoder(
        filters_decoder,
        encoder.shape_latent,
        channels,
        flat_latent,
        dim_latent,
        kernel_size_decoder,
        strides_decoder,
        activation_fn=activation_fn_decoder,
        batch_norm=batch_norm,
        rngs=rngs,
    )

    return AutoEncoder(encoder, decoder)
