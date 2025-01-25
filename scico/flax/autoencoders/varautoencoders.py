# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementations of variational autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax.random import PRNGKey, normal
from jax.typing import ArrayLike

import flax.linen as nn
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from scico.flax.autoencoders.autoencoders import ConvDecoder, DenseDecoder
from scico.flax.autoencoders.blocks import CNN, MLP

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


class VarEncoder(nn.Module):
    """Generic variational encoder.

    Args:
        mean_block: Flax module for representing means of encoded latent
            space.
        logvar_block: Flax module for representing log variances of
            encoded latent space.
        latent_dim: Latent dimension of variational encoder.
    """

    mean_block: Callable
    logvar_block: Callable
    latent_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike]:
        """Apply variational encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The mean and logvar for generation in encoded latent space.
        """
        mean_ = self.mean_block(x)
        logvar_ = self.logvar_block(x)
        mean = nn.Dense(self.latent_dim)(mean_)
        logvar = nn.Dense(self.latent_dim)(logvar_)

        return mean, logvar


class VAE(nn.Module):
    """Basic definition of a variational autoencoder (VAE) network as a
    Flax model.

    Args:
        encoder: Variational encoder module in Flax.
        decoder: Decoder module in Flax.
        cond_width: Widht of layer for class conditional decoding. If
        zero, no class-conditional decoding is learned.
    """

    encoder: Callable
    decoder: Callable
    cond_width: int = 0

    def setup(self):
        """Setup of encoder and decoder modules for variational
        autoencoder (VAE)."""
        nn.share_scope(self, self.encoder)
        nn.share_scope(self, self.decoder)

        if self.cond_width > 0:
            # For Conditional decoding.
            self.post_latent_proj = nn.Dense(self.cond_width)
            self.class_proj = nn.Dense(self.cond_width)

    def encode(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Variational encoding."""
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode_cond(self, x: ArrayLike, c: ArrayLike):
        """Class-conditional decoding."""
        assert self.cond_width > 0
        x = self.post_latent_proj(x)
        x = x + self.class_proj(c)
        x = self.decoder(x)
        return x

    def decode(self, x: ArrayLike):
        """Class-independent decoding."""
        x = self.decoder(x)
        return x

    def reparameterize(self, mean: ArrayLike, logvar: ArrayLike, key: PRNGKey) -> ArrayLike:
        """Reparametrization trick for sample generation.

        Args:
            mean: Array with means for generation of normally
                distributed random samples.
            logvar: Array with log variances for generation of normally
                distributed random samples.
            key: The key for the random generation.

        Returns:
            The generated normally distributed random samples.
        """
        std = jnp.exp(0.5 * logvar)
        epsilon = normal(key, std.shape)
        return mean + epsilon * std

    def __call__(
        self, x: ArrayLike, key: PRNGKey, c: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply sequence of variational encoder and decoder modules.

        Args:
            x: The array to be processed via the variational autoencoder.
            key: The key for the random generation of sample.
            c: The array with the class for conditional generation.

        Returns:
            The generated sample, the mean and log variance used in the
            generation.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, key)
        if c is None:
            y = self.decoder(z)
        else:
            y = self.decode_cond(z, c)
        return y, mean, logvar


class DenseVAE(nn.Module):
    """Definition of variational autoencoder network using multi layer
    perceptron (MLP), i.e. dense layers.

    Output is reshaped to given shape via a properly sized layer added
    automatically to the tuple of the decoder widths.

    Args:
        out_shape: Tuple (height, width) of signal to decode.
        channels: Number of channels of signal to decode.
        encoder_widths: Sequential list with number of neurons per layer
            in the MLP encoder.
        latent_dim: Latent dimension of encoder.
        decoder_widths: Sequential list with number of neurons per layer
            in the MLP decoder.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
        class_conditional: Flag to specify if decoding will be
            conditioned on a sample class.
        dtype: Output dtype. Default: :attr:`~numpy.float32`.
    """

    out_shape: Tuple[int]
    channels: int
    encoder_widths: Sequence[int]
    latent_dim: int
    decoder_widths: Sequence[int]
    activation_fn: Callable = nn.relu
    class_conditional: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: ArrayLike, key: PRNGKey, c: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply sequence of variational encoder and decoder modules.

        Args:
            x: The array to be processed via the variational autoencoder.
            key: The key for the random generation of sample.
            c: The array with the class for conditional generation.

        Returns:
            The generated sample, the mean and log variance used in the
            generation.
        """
        mean_block = MLP(self.encoder_widths, self.activation_fn, flatten_first=True)
        logvar_block = MLP(self.encoder_widths, self.activation_fn, flatten_first=True)

        encoder = VarEncoder(
            mean_block,
            logvar_block,
            self.latent_dim,
        )

        decoder = DenseDecoder(
            self.out_shape + (self.channels,),
            self.decoder_widths,
            self.activation_fn,
            reshape_final=True,
        )

        if self.class_conditional:
            cond_width = encoder_widths[-1]
        else:
            cond_width = 0

        return VAE(encoder, decoder, cond_width)(x, key, c)


class ConvVAE(nn.Module):
    """Definition of variational autoencoder network using convolutional
    layers.

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
        class_conditional: Flag to specify if decoding will be
            conditioned on a sample class.
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
    class_conditional: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: ArrayLike, key: PRNGKey, c: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply sequence of variational encoder and decoder modules.

        Args:
            x: The array to be processed via the variational autoencoder.
            key: The key for the random generation of sample.
            c: The array with the class for conditional generation.

        Returns:
            The generated sample, the mean and log variance used in the
            generation.
        """
        mean_block = CNN(
            self.encoder_filters,
            self.encoder_kernel_size,
            self.encoder_strides,
            activation_fn=self.encoder_activation_fn,
            flatten_final=True,
        )
        logvar_block = CNN(
            self.encoder_filters,
            self.encoder_kernel_size,
            self.encoder_strides,
            activation_fn=self.encoder_activation_fn,
            flatten_final=True,
        )
        encoder = VarEncoder(
            mean_block,
            logvar_block,
            self.latent_dim,
        )
        decoder = ConvDecoder(
            self.out_shape,
            self.channels,
            self.decoder_filters,
            self.decoder_kernel_size,
            self.decoder_strides,
            activation_fn=self.decoder_activation_fn,
        )

        if self.class_conditional:
            cond_width = encoder_filters[-1]
        else:
            cond_width = 0

        return VAE(encoder, decoder, cond_width)(x, key, c)
