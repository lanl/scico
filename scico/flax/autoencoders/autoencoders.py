# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementations of autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Any, Callable, Sequence, Tuple

from numpy import prod

import jax.numpy as jnp
from jax.typing import ArrayLike

import flax.linen as nn
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from scico.flax.autoencoders.blocks import CNN, MLP

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


class AE(nn.Module):
    """Basic definition of an autoencoder network as a Flax model.

    Args:
        encoder: Encoder module in Flax.
        decoder: Decoder module in Flax.
    """

    encoder: Callable
    decoder: Callable

    def setup(self):
        """Setup of encoder and decoder modules for autoencoder (AE)."""
        nn.share_scope(self, self.encoder)
        nn.share_scope(self, self.decoder)

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

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Apply sequence of encoder and decoder modules.

        Args:
            x: The array to be autoencoded.

        Returns:
            The output of the autoencoder module and the encoded
            representation.
        """
        y = self.encode(x)
        x = self.decode(y)
        return x, y


class DenseEncoder(nn.Module):
    """Encoder using densely connected layers (i.e multi layer
    perceptron, MLP).

    Args:
        encoder_widths: Sequential list with number of neurons per layer
            in the MLP.
        latent_dim: Latent dimension of encoder.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
    """

    encoder_widths: Sequence[int]
    latent_dim: int
    activation_fn: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply dense encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The encoded array.
        """
        x = x.reshape((x.shape[0], -1))
        x = MLP(self.encoder_widths, self.activation_fn, activate_final=True)(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class DenseDecoder(nn.Module):
    """Decoder using densely connected layers (i.e multi layer
    perceptron, MLP).

    The output can be reshaped to a pre-defined shape.

    Args:
        out_shape: Tuple (height, width, channel) of image to decode (if
            reshape requested).
        decoder_widths: Sequential list with number of neurons per layer
            in the decoder MLP. An additional properly sized layer is
            added if reshape final is set to ``True``.
        activation_fn: Flax function defining the activation operation
            to apply after each layer. (Except output layer).
        reshape_final: Flag to indicate if the output should be reshaped
            before returning.

    """

    out_shape: Tuple[int]
    decoder_widths: Sequence[int]
    activation_fn: Callable = nn.leaky_relu
    reshape_final: bool = True

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply dense decoder.

        Args:
            x: The array to be decoded.

        Returns:
            The decoded array.
        """
        if self.reshape_final:  # Restore specific shape
            dim_out = prod(self.out_shape)
            x = MLP(self.decoder_widths + (dim_out,), self.activation_fn, activate_final=False)(x)
            x = x.reshape((x.shape[0],) + self.out_shape)
        else:
            x = MLP(self.decoder_widths, self.activation_fn, activate_final=False)(x)
        return x


class DenseAE(AE):
    """Definition of autoencoder network using multi layer perceptron
    (MLP), i.e. dense layers.

    Output is reshaped to given output shape via a properly sized layer
    added automatically to the tuple of the decoder widths.

    Args:
        out_shape: Tuple (height, width, channels) of signal to decode
            (if reshape requested).
        encoder_widths: List with number of neurons per layer in the
            MLP encoder.
        latent_dim: Latent dimension of encoder.
        decoder_widths: List with number of neurons per layer in the
            MLP decoder.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    out_shape: Tuple[int]
    encoder_widths: Tuple[int]
    latent_dim: int
    decoder_widths: Tuple[int]
    activation_fn: Callable = nn.leaky_relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Apply sequence of encoder and decoder modules.

        Args:
            x: The array to be autoencoded.

        Returns:
            The output of the autoencoder module and the encoded
            representation.
        """
        encoder = DenseEncoder(
            self.encoder_widths,
            self.latent_dim,
            self.activation_fn,
        )

        decoder = DenseDecoder(
            self.out_shape,
            self.decoder_widths,
            self.activation_fn,
            reshape_final=True,
        )
        return AE(encoder, decoder)(x)


class ConvEncoder(nn.Module):
    """Encoder using convolutional layers.

    Args:
        encoder_filters: List with number of filters per layer in the
            convolutional encoder.
        latent_dim: Latent dimension of encoder.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
    """

    encoder_filters: Sequence[int]
    latent_dim: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply convolutional encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The encoded array.
        """
        x = CNN(
            self.encoder_filters,
            self.kernel_size,
            self.strides,
            activation_fn=self.activation_fn,
            flatten_final=True,
        )(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class ConvDecoder(nn.Module):
    """Decoder using convolutional layers.

    All the layers use the same specified kernel size and stride, use a
    circular padding, and do not use bias.

    Args:
        out_shape: Tuple (height, width) of signal to decode.
        channels: Number of channels of signal to encode.
        decoder_filters: List with number of filters per layer in the
            convolutional encoder.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in
            convolution.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
    """

    out_shape: Tuple[int]
    channels: int
    decoder_filters: Sequence[int]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply convolutional decoder.

        Args:
            x: The array to be decoded.

        Returns:
            The decoded array.
        """
        x = nn.Dense(prod(self.out_shape) * self.channels)(x)
        x = x.reshape((x.shape[0],) + self.out_shape + (self.channels,))

        # CNN transpose layers.
        for nfilters in self.decoder_filters:
            x = nn.ConvTranspose(
                nfilters, self.kernel_size, strides=self.strides, use_bias=False, padding="CIRCULAR"
            )(x)
            x = self.activation_fn(x)

        # Restore given channels.
        x = nn.ConvTranspose(
            self.channels,
            self.kernel_size,
            strides=self.strides,
            use_bias=False,
            padding="CIRCULAR",
        )(x)

        return x


class ConvAE(AE):
    """Definition of autoencoder network using convolutional layers.

    Args:
        out_shape: Tuple (height, width) of signal to decode.
        channels: Number of channels of signal to decode.
        encoder_filters: Sequential list with number of filters per
            layer in the convolutional encoder.
        latent_dim: Latent dimension of encoder.
        decoder_filters: Sequential list with number of filters per
            layer in the convolutional decoder.
        encoder_kernel_size: A shape tuple defining the size of the
            convolution filters in encoder.
        encoder_strides: A shape tuple defining the size of strides in
            convolutions in encoder.
        encoder_activation_fn: Flax function defining the activation
            operation to apply after each layer in encoder (except
            output layer).
        decoder_kernel_size: A shape tuple defining the size of the
            convolution filters in decoder.
        decoder_strides: A shape tuple defining the size of strides in
            convolutions in decoder.
        decoder_activation_fn: Flax function defining the activation
            operation to apply after each layer in decoder (except
            output layer).
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    out_shape: Tuple[int]
    channels: int
    encoder_filters: Sequence[int]
    latent_dim: int
    decoder_filters: Sequence[int]
    encoder_kernel_size: Tuple[int, int] = (3, 3)
    encoder_strides: Tuple[int, int] = (1, 1)
    encoder_activation_fn: Callable = nn.leaky_relu
    decoder_kernel_size: Tuple[int, int] = (3, 3)
    decoder_strides: Tuple[int, int] = (1, 1)
    decoder_activation_fn: Callable = nn.leaky_relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Apply sequence of encoder and decoder modules.

        Args:
            x: The array to be autoencoded.

        Returns:
            The output of the autoencoder module and the encoded
            representation.
        """
        encoder = ConvEncoder(
            self.encoder_filters,
            self.latent_dim,
            self.encoder_kernel_size,
            self.encoder_strides,
            activation_fn=self.encoder_activation_fn,
        )

        decoder = ConvDecoder(
            self.out_shape,
            self.channels,
            self.decoder_filters,
            self.decoder_kernel_size,
            self.decoder_strides,
            activation_fn=self.decoder_activation_fn,
        )
        return AE(encoder, decoder)(x)
