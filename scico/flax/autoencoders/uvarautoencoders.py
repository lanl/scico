# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Flax implementations of variational autoencoders with multi-level
structure like UNet networks. This allows for reconstructions composed
of encodings from different latent spaces."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax.random import PRNGKey, split
from jax.typing import ArrayLike

import flax.linen as nn
from flax.core import Scope  # noqa
from flax.linen.module import _Sentinel  # noqa
from scico.flax.autoencoders.blocks import ConvPoolBlock, ConvUpsampleBlock
from scico.flax.autoencoders.varautoencoders import reparameterize

# The imports of Scope and _Sentinel (above) are required to silence
# "cannot resolve forward reference" warnings when building sphinx api
# docs.


class MultiLevelVarEncoder(nn.Module):
    """Variational encoder with multiple latent levels and convolutional
    layers.

    The model will learn a collection of means and log variances where
    each mean and log variance pair corresponds to a particular latent
    level.

    Args:
        num_filters: Sequential list with number of filters per
            level in the convolutional encoder.
        latent_dims: Sequential list of latent dimensions of the multiple
            levels of the variational encoder.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in
            convolution.
        activation_fn: Flax function defining the activation
            operation to apply after each layer in encoder (except
            output layer).
        window_shape: A shape tuple defining the window to reduce over in
            the pooling operation.
    """

    num_filters: Sequence[int]
    latent_dims: Sequence[int]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu
    window_shape: Tuple[int, int] = (2, 2)

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Apply multi-level variational encoder.

        Args:
            x: The array to be encoded.

        Returns:
            The lists of mean and log variances for generation in each
            of the encoded latent spaces of the multi-level encoder.
        """
        mean = []
        logvar = []
        mean_ = x
        logvar_ = x
        for j, nfilters in enumerate(self.num_filters):
            mean_ = ConvPoolBlock(
                nfilters,
                self.kernel_size,
                self.strides,
                activation_fn=self.activation_fn,
                window_shape=self.window_shape,
            )(mean_)
            logvar_ = ConvPoolBlock(
                nfilters,
                self.kernel_size,
                self.strides,
                activation_fn=self.activation_fn,
                window_shape=self.window_shape,
            )(logvar_)
            # Flatten and combine
            mean.append(nn.Dense(self.latent_dims[j])(mean_.reshape((x.shape[0], -1))))
            logvar.append(nn.Dense(self.latent_dims[j])(logvar_.reshape((x.shape[0], -1))))
        return mean, logvar


class MultiLevelDecoder(nn.Module):
    """Decoder using convolutional layers and decoding multiple latent
    levels.

    All the layers use the same specified kernel size and stride, use a
    circular padding, and do not use bias.

    Args:
        platent_shape: Shape (height, width) of signal previous to
            final latent.
        channels: Number of channels of signal to encode.
        num_filters: List with number of filters per layer in the
            convolutional decoder.
        kernel_size: A shape tuple defining the size of the convolution
            filters.
        strides: A shape tuple defining the size of strides in
            convolution.
        activation_fn: Flax function defining the activation operation
            to apply after each layer (except output layer).
        upsampling_scale: Integer scaling factor.
    """

    platent_shape: Tuple[int]
    channels: int
    num_filters: Sequence[int]
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation_fn: Callable = nn.leaky_relu
    upsampling_scale: int = 2

    @nn.compact
    def __call__(self, xlist, lat=False) -> ArrayLike:
        """Apply multi-level decoder.

        Args:
            xlist: The list with multi-level latent representations to be
                decoded.
            lat: Flag to indicate if the latent representations are to be
                returned.

        Returns:
            The reconstructed signal and, if requested, the intermediate
            representations.
        """
        xlup = []
        h, w = self.platent_shape
        x = None
        for j, nfilters in enumerate(self.num_filters):
            ns = h * w
            xl = nn.Dense(ns)(xlist[-j - 1])
            xl = xl.reshape((xlist[0].shape[0], h, w, self.channels))
            if x is None:
                x = xl
            else:
                x = x + xl
            xlup.append(x)
            x = ConvUpsampleBlock(
                nfilters, self.kernel_size, self.strides, self.activation_fn, self.upsampling_scale
            )(x)
            xlup.append(x)
            h = h * self.upsampling_scale
            w = w * self.upsampling_scale

        # Restore given channels
        x = nn.ConvTranspose(
            self.channels,
            self.kernel_size,
            strides=self.strides,
            use_bias=False,
            padding="CIRCULAR",
        )(x)

        if lat:
            return x, xlup

        return x


class UNetVAE(nn.Module):
    """Definition of a UNet-like variational autoencoder network
    using convolutional layers and multiple levels of downsampling
    and upsampling.

    Args:
        out_shape: Tuple (height, width) of signal to decode.
        channels: Number of channels of signal to decode.
        encoder_filters: Sequential list with number of filters per
            layer in the convolutional encoder.
        latent_dims: Sequential list of latent dimensions of the multiple
            levels of the variational encoder.
        decoder_filters: Sequential list with number of filters per
            layer in the convolutional decoder.
        scale: Factor of reduction in each level.
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
    latent_dims: Sequence[int]
    decoder_filters: Sequence[int]
    scale: int = 2
    encoder_kernel_size: Tuple[int, int] = (3, 3)
    encoder_strides: Tuple[int, int] = (1, 1)
    encoder_activation_fn: Callable = nn.leaky_relu
    decoder_kernel_size: Tuple[int, int] = (3, 3)
    decoder_strides: Tuple[int, int] = (1, 1)
    decoder_activation_fn: Callable = nn.leaky_relu
    class_conditional: bool = False
    dtype: Any = jnp.float32

    def __post_init__(self):
        if self.class_conditional:
            self.cond_width = self.encoder_filters[-1]
        else:
            self.cond_width = 0

        super().__post_init__()

    def setup(self):
        self.encoder = MultiLevelVarEncoder(
            self.encoder_filters,
            self.latent_dims,
            self.encoder_kernel_size,
            self.encoder_strides,
            self.encoder_activation_fn,
            window_shape=(self.scale, self.scale),
        )

        hR = self.out_shape[0] // self.scale ** (len(self.encoder_filters))
        wR = self.out_shape[1] // self.scale ** (len(self.encoder_filters))

        self.decoder = MultiLevelDecoder(
            (hR, wR),
            self.channels,
            self.decoder_filters,
            self.decoder_kernel_size,
            self.decoder_strides,
            self.decoder_activation_fn,
            upsampling_scale=self.scale,
        )

        # Conditional decoding
        if self.class_conditional:
            self.post_latent_proj = [
                nn.Dense(self.cond_width) for _ in range(len(self.encoder_filters))
            ]
            self.class_proj = [nn.Dense(self.cond_width) for _ in range(len(self.encoder_filters))]

    def encode(self, x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Encode using multiple latent representations.

        Args:
            x: Signals to encode.

        Returns:
            The mean and log variances of the encoded signals.
        """
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode_cond(self, xlist: ArrayLike, c: ArrayLike):
        """Class-conditional decoding using multiple latent representations.

        Args:
            xlist: List of random generations in latent spaces.
            c: Classes to generate samples from.

        Returns:
            The generated samples.
        """
        xl = []
        for j, z_ in enumerate(xlist):
            x = self.post_latent_proj[j](z_)
            x = x + self.class_proj[j](c)
            xl.append(x)
        x = self.decoder(xl)
        return x

    def decode_cond_return_latent(self, xlist: ArrayLike, c: ArrayLike):
        """Class-conditional decoding using multiple latent representations.
        The different latent representations are returned also.

        Args:
            xlist: List of random generations in latent spaces.
            c: Classes to generate samples from.

        Returns:
            The generated samples as well as the intermediate representations.
        """
        xl = []
        for j, z_ in enumerate(xlist):
            x = self.post_latent_proj[j](z_)
            x = x + self.class_proj[j](c)
            xl.append(x)
        x, xlup = self.decoder(xl, lat=True)
        return x, xlup

    def decode(self, xlist: ArrayLike):
        """Class-independent decoding using multiple latent representations.

        Args:
            xlist: List of random generations in latent spaces.

        Returns:
            The generated samples.
        """
        x = self.decoder(xlist)
        return x

    @nn.compact
    def __call__(
        self, x: ArrayLike, key: PRNGKey, c: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply sequence of multi-level variational encoder and
            multi-level decoder modules.

        Args:
            x: The array to be processed via the variational autoencoder.
            key: The key for the random generation of sample.
            c: The array with the class for conditional generation.

        Returns:
            The generated sample, the mean and log variances of the
            multiple levels used in the generation.
        """
        mean, logvar = self.encode(x)
        z = []
        for j, m in enumerate(mean):
            key, subkey = split(key)
            z.append(reparameterize(m, logvar[j], subkey))
        if c is None:
            y = self.decoder(z)
        else:
            y = self.decode_cond(z, c)
        return y, mean, logvar
