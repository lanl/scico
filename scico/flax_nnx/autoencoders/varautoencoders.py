# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax NNX implementations of variational autoencoders."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Callable, Optional, Sequence, Tuple

from numpy import prod

import jax.numpy as jnp
from jax.random import PRNGKey, normal
from jax.typing import ArrayLike

from flax import nnx
from flax.core import Scope  # noqa

from .autoencoders import (
    ConvDecoder,
    ConvEncoder,
    Decoder,
    Encoder,
    MLPDecoder,
    MLPEncoder,
)

# The import of Scope above is required to silence "cannot resolve
# forward reference" warnings when building sphinx api docs.


def reparameterize(mean: ArrayLike, logvar: ArrayLike, key: PRNGKey) -> ArrayLike:
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
    epsilon = normal(key, shape=std.shape)
    return mean + epsilon * std


class VarEncoder(Encoder):
    """Generic NNX-based variational encoder."""

    def __init__(
        self,
        mean_block: Callable,
        logvar_block: Callable,
    ):
        """Initialize VarEncoder.

        Args:
            mean_block: Flax NNX module for representing means of encoded latent
                space.
            logvar_block: Flax NNX module for representing log variances of
                encoded latent space.
        """
        super().__init__()
        # Input validation
        assert isinstance(mean_block, Encoder)
        assert isinstance(logvar_block, Encoder)
        assert mean_block.latent_dim == logvar_block.latent_dim
        # Store modules
        self.mean_block = mean_block
        self.logvar_block = logvar_block
        # Store properties
        self.dim_latent = mean_block.latent_dim
        self.flat_latent = mean_block.flat_latent
        if self.flat_latent:
            self.shape_latent = mean_block.latent_dim
        else:
            self.shape_latent = mean_block.shape_latent

    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Apply variational encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The mean and logvar for generation in encoded latent space.
        """
        mean = self.mean_block(x)
        logvar = self.logvar_block(x)

        return mean, logvar


class Conditioner(Encoder):
    """Generic NNX-based module for conditioning outputs."""

    def __init__(
        self,
        processing_block: Callable,
        conditioning_block: Callable,
    ):
        """Initialize VarEncoder.

        Args:
            processing_block: Flax NNX module representing the processing on
                the latent signal for conditioning.
            conditioning_block: Flax NNX module representing the processing on
                the conditioning signal (e.g., classes).
        """
        super().__init__()
        # Input validation
        assert isinstance(processing_block, Encoder)
        assert isinstance(conditioning_block, Encoder)
        assert processing_block.latent_dim == conditioning_block.latent_dim
        # Store modules
        self.processing_block = processing_block
        self.conditioning_block = conditioning_block
        self.dim_latent = processing_block.latent_dim

    def __call__(self, x: ArrayLike, c: ArrayLike) -> ArrayLike:
        """Apply conditioner module.

        Args:
            x: Signal dependent part of the conditioning.
            c: Conditioning property to generate samples from. For example, classes.

        Returns:
            The output of the conditioning block.
        """
        return self.processing_block(x) + self.conditioning_block(c)


class VAE(nnx.Module):
    """Basic definition of a variational autoencoder (VAE) network as a
    Flax NNX model."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        conditioner: Optional[Encoder] = None,
    ):
        """Initialize VAE.
        Args:
            encoder: Variational encoder module in Flax NNX.
            decoder: Decoder module in Flax NNX.
            conditioner: Flax NNX module for conditional decoding. If
                ``None``, no conditional decoding is learned.
        """
        super().__init__()
        assert encoder.dim_latent == decoder.dim_latent
        if conditioner is not None:
            assert encoder.dim_latent == conditioner.dim_latent
        self.encoder = encoder
        self.decoder = decoder
        self.conditioner = conditioner

    def encode(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Variational encoding.

        Args:
            x: Signals to encode.

        Returns:
            The mean and log variances of the encoded signals.
        """
        return self.encoder(x)  # yields: mean, logvar

    def decode_cond(self, x: ArrayLike, c: ArrayLike):
        """Conditional decoding.

        Args:
            x: Random generation in latent space to decode.
            c: Conditioning property to generate samples from. For example, classes.

        Returns:
            The generated samples.
        """
        assert self.conditioner is not None
        xshp = x.shape
        x = self.conditioner(x, c)
        if not self.decoder.flat_latent and x.shape != xshp:
            x = self.decoder(x.reshape(xshp))
        else:
            x = self.decoder(x)
        return x

    def decode(self, x: ArrayLike):
        """Class-independent decoding.

        Args:
            x: Random generation in latent space to decode.

        Returns:
            The generated samples.
        """
        return self.decoder(x)

    def __call__(
        self, x: ArrayLike, key: PRNGKey, c: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply sequence of variational encoder and decoder modules.

        Args:
            x: The array to be processed via the variational autoencoder.
            key: The jax key for the random generation of sample.
            c: The array with the property for conditional generation (e.g. class
                membership).

        Returns:
            The generated (conditional) sample, the mean and log variance used in the
            generation.
        """
        mean, logvar = self.encode(x)
        z = reparameterize(mean, logvar, key)
        if c is None:
            y = self.decode(z)
        else:
            y = self.decode_cond(z, c)
        return y, mean, logvar


def MLPVarAutoEncoder(
    dim_in: int,
    widths_mean_encoder: Tuple[int],
    widths_logvar_encoder: Tuple[int],
    dim_latent: int,
    widths_decoder: Tuple[int],
    shape_out: Tuple[int],
    activation_fn: Callable = nnx.leaky_relu,
    batch_norm: bool = False,
    conditional: bool = False,
    dim_cond_in: int = None,
    widths_condproc_encoder: Optional[Tuple[int]] = None,
    widths_cond_encoder: Optional[Tuple[int]] = None,
    rngs: nnx.Rngs = nnx.Rngs(0),
):
    """Function to construct variational autoencoder network using multi
    layer perceptron (MLP), i.e. dense layers.

    Args:
        dim_in: Dimension of input signal.
        widths_mean_encoder: List with number of neurons per layer in the
            MLP mean encoder.
        widths_logvar_encoder: List with number of neurons per layer in the
            MLP logvar encoder.
        dim_latent: Latent dimension of encoder.
        widths_decoder: List with number of neurons per layer in the
            MLP decoder.
        shape_out: Tuple (height, width, channels) of signal to decode
            (if reshape requested).
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
        batch_norm: Flag to indicate if batch norm is to be applied or not.
        conditional: Flag to specify if decoding will be
            conditioned on a given property (e.g. class membership).
        dim_cond_in: Dimension of conditioning signal.
        widths_condproc_encoder: List with number of neurons per layer in the
            MLP conditional processing encoder.
        widths_cond_encoder: List with number of neurons per layer in the
            MLP conditional property encoder.
        rngs: Random generation key.

    Returns:
        Variational autoencoder model with architecture based on MLP, i.e. dense layers.
    """
    # Build components for encoder
    mean_block = MLPEncoder(
        dim_in,
        widths_mean_encoder,
        dim_latent,
        activation_fn,
        batch_norm,
        rngs=rngs,
    )
    logvar_block = MLPEncoder(
        dim_in,
        widths_logvar_encoder,
        dim_latent,
        activation_fn,
        batch_norm,
        rngs=rngs,
    )
    # Build variational encoder
    encoder = VarEncoder(
        mean_block=mean_block,
        logvar_block=logvar_block,
    )

    # Build decoder
    decoder = MLPDecoder(
        dim_latent,
        widths_decoder,
        shape_out,
        activation_fn,
        batch_norm,
        reshape_final=True,
        rngs=rngs,
    )

    conditioner = None
    if conditional:
        assert widths_condproc_encoder is not None
        assert widths_cond_encoder is not None
        # Build components for conditioning
        proc_block = MLPEncoder(
            dim_latent,
            widths_condproc_encoder,
            dim_latent,
            activation_fn,
            batch_norm,
            rngs=rngs,
        )
        cond_block = MLPEncoder(
            dim_cond_in,
            widths_cond_encoder,
            dim_latent,
            activation_fn,
            batch_norm,
            rngs=rngs,
        )
        # Build conditioner
        conditioner = Conditioner(processing_block=proc_block, conditioning_block=cond_block)

    return VAE(encoder, decoder, conditioner)


def ConvVarAutoEncoder(
    shape_in: Tuple[int],
    channels: int,
    filters_mean_encoder: Sequence[int],
    filters_logvar_encoder: Sequence[int],
    filters_decoder: Sequence[int],
    kernel_size_mean_encoder: Tuple[int, int] = (3, 3),
    kernel_size_logvar_encoder: Tuple[int, int] = (3, 3),
    kernel_size_decoder: Tuple[int, int] = (3, 3),
    activation_fn: Callable = nnx.leaky_relu,
    batch_norm: bool = False,
    flat_latent: bool = False,
    dim_latent: Optional[int] = None,
    conditional: bool = False,
    mode_cond: str = "conv",
    filters_condproc_decoder: Optional[Tuple[int]] = None,
    kernel_size_condproc_decoder: Optional[Tuple[int, int]] = (3, 3),
    strides_condproc_decoder: Optional[Tuple[int, int]] = (1, 1),
    shape_cond_in: Optional[Tuple[int]] = None,
    channels_cond: Optional[int] = None,
    filters_cond_encoder: Optional[Tuple[int]] = None,
    kernel_size_cond_encoder: Optional[Tuple[int, int]] = (3, 3),
    widths_condproc_encoder: Optional[Tuple[int]] = None,
    dim_cond_in: Optional[int] = None,
    widths_cond_encoder: Optional[Tuple[int]] = None,
    rngs: nnx.Rngs = nnx.Rngs(0),
):
    """Function to construct variational autoencoder network using convolutional layers.

    Args:
        shape_in: Tuple (height, width) of signal to encode.
        channels: Number of channels of signal to encode.
        filters_mean_encoder: List with number of filters per layer in the
                CNN mean encoder.
        filters_logvar_encoder: List with number of filters per layer in the
                CNN logvar encoder.
        filters_decoder: List with number of filters per layer in the
                convolutional decoder.
        kernel_size_mean_encoder: A shape tuple defining the size of the convolution
                filters for the mean encoder.
        kernel_size_logvar_encoder: A shape tuple defining the size of the convolution
                filters for the logvar encoder.
        kernel_size_decoder: A shape tuple defining the size of the convolution
                filters for encoder.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer) in encoder.
        batch_norm: Flag to indicate if batch norm is to be applied or not.
        flat_latent: Flag to specify if the latent representation should be flatten.
        dim_latent: Latent dimension of encoder (only used if flag for flattening is
            true and a meanigful value is provided).
        conditional: Flag to specify if decoding will be
            conditioned on a given property (e.g. class membership).
        mode_cond: Mode of model for conditional signal. It can be conv, requiring
            shape, channels, filters and kernel size. If MLP, it requires dim in and
            widths for layers.
        filters_condproc_decoder: List with number of filters per layer in the
            CNN conditional processing decoder (if using conv).
        kernel_size_condproc_decoder: A shape tuple defining the size of the convolution
            filters for the conditional processing decoder (if using conv).
        strides_condproc_decoder: A shape tuple defining the size of strides in
            convolution for the conditional processing decoder (if using conv).
        shape_cond_in: Tuple (height, width) of conditioning signal (if using conv).
        channels_cond: Number of channels of conditioning signal (if using conv).
        filters_cond_encoder: List with number of filters per layer in the
            CNN conditional property encoder (if using conv).
        kernel_size_cond_encoder: A shape tuple defining the size of the convolution
            filters for the conditional property encoder (if using conv).
        widths_condproc_encoder: List with number of neurons per layer in the
            MLP conditional processing encoder (if using MLP).
        dim_cond_in: Dimension of conditioning signal (if using MLP).
        widths_cond_encoder: List with number of neurons per layer in the
            MLP conditional property encoder (if using MLP).
        rngs: Random generation key.

    Returns:
        Variational autoencoder model with architecture based on convolutional layers.
    """
    strides: Tuple[int, int] = (2, 2)

    # Build components for encoder
    mean_block = ConvEncoder(
        shape_in,
        channels,
        filters_mean_encoder,
        flat_latent,
        dim_latent,
        kernel_size_mean_encoder,
        strides,
        activation_fn=activation_fn,
        rngs=rngs,
    )
    logvar_block = ConvEncoder(
        shape_in,
        channels,
        filters_logvar_encoder,
        flat_latent,
        dim_latent,
        kernel_size_logvar_encoder,
        strides,
        activation_fn=activation_fn,
        rngs=rngs,
    )
    # Build variational encoder
    encoder = VarEncoder(
        mean_block=mean_block,
        logvar_block=logvar_block,
    )

    # Build decoder
    decoder = ConvDecoder(
        filters_decoder,
        mean_block.shape_latent,
        channels,
        flat_latent,
        dim_latent,
        kernel_size_decoder,
        strides,
        activation_fn=activation_fn,
        batch_norm=batch_norm,
        rngs=rngs,
    )

    conditioner = None
    if conditional:

        if mode_cond == "conv":
            assert shape_cond_in is not None
            assert channels_cond is not None
            assert filters_condproc_decoder is not None
            assert filters_cond_encoder is not None
        else:
            assert dim_cond_in is not None
            assert widths_condproc_encoder is not None
            assert widths_cond_encoder is not None

        if dim_latent is None:
            dim_latent = int(prod(mean_block.shape_latent))

        # Build components for conditioning
        if mode_cond == "conv":
            proc_block = ConvDecoder(
                filters_condproc_decoder,
                mean_block.shape_latent,
                mean_block.shape_latent[-1],
                flat_latent,
                dim_latent,
                kernel_size_condproc_decoder,
                strides_condproc_decoder,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                rngs=rngs,
            )
            cond_block = ConvEncoder(
                shape_cond_in,
                channels_cond,
                filters_cond_encoder,
                flat_latent,
                dim_latent,
                kernel_size_cond_encoder,
                strides,
                activation_fn=activation_fn,
                rngs=rngs,
            )
        else:
            proc_block = MLPEncoder(
                dim_latent,
                widths_condproc_encoder,
                dim_latent,
                activation_fn,
                batch_norm,
                rngs=rngs,
            )
            cond_block = MLPEncoder(
                dim_cond_in,
                widths_cond_encoder,
                dim_latent,
                activation_fn,
                batch_norm,
                rngs=rngs,
            )
        # Build conditioner
        conditioner = Conditioner(processing_block=proc_block, conditioning_block=cond_block)

    return VAE(encoder, decoder, conditioner)
