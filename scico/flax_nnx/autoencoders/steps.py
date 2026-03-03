# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
autoencoder models."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx
from scico.metric import snr as snr_fn


def _kl_loss_fn(mean: ArrayLike, logvar: ArrayLike) -> ArrayLike:
    """Compute KL divergence loss from given mean and log variance.

    This KL loss is computed with respect to a standard normal distribution.

    Args:
        mean: Mean in latent space. For multi-level VAE this is a list
            of means for each latent level.
        logvar: Log variances in latent space. For multi-level VAE
            this is a list of log variances for each latent level.
    """
    if isinstance(mean, list):  # For multi-level VAE
        kl_loss = 0.0
        for j, m in enumerate(mean):
            reduce_dims = list(range(1, len(m.shape)))
            kl_loss = kl_loss + jnp.mean(
                -0.5 * jnp.sum(1 + logvar[j] - m**2 - jnp.exp(logvar[j]), axis=reduce_dims)
            )
    else:  # For regular VAE
        reduce_dims = list(range(1, len(mean.shape)))
        kl_loss = jnp.mean(-0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=reduce_dims))

    return kl_loss


def loss_fn(
    model: Callable,
    criterion: Callable,
    kl_loss_fn: Callable,
    kl_weight: float,
    x: ArrayLike,
    key: ArrayLike,
    y: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Loss function definition for diffusion step..

    Args:
        model: Model to train.
        criterion: Criterion to evaluate autoencoder reconstruction.
        kl_loss_fn: Function to evaluate KL divergence in latent space.
        kl_weight: Weight for KL term in loss function which, is the weighted sum of
            reconstruction and KL losses.
        x: Input data array.
        key: Key for random generation in VAE forward pass.
        y: Conditioning data array.

    Returns:
        Evaluated loss, KL term and reconstruction output.
    """
    output, mean, logvar = model(x, key, y)
    reduce_dims = list(range(1, len(x.shape)))
    reconstruction_loss = (1.0 - kl_weight) * criterion(output, x).sum(axis=reduce_dims).mean()
    kl_loss = kl_weight * kl_loss_fn(mean, logvar)
    loss = reconstruction_loss + kl_loss
    return loss, (kl_loss, output)


@partial(jax.jit, static_argnums=(2, 3))
def jax_train_step_vae(
    graphdef,
    state,
    criterion: Callable,
    kl_loss_fn: Callable,
    kl_weight: float,
    x: ArrayLike,
    key: ArrayLike,
    y: ArrayLike,
) -> Tuple[ArrayLike, Any]:
    """Train VAE for a single step.

    This function uses data and a criterion to optimize model parameters. It returns
    the current loss in the training batch.

    Args:
        graphdef: Graph representation of model.
        state: NNX state object including pytrees for all the
            model and optimizer graph nodes.
        criterion: Criterion to evaluate autoencoder reconstruction.
        kl_loss_fn: Function to evaluate KL divergence in latent space.
        kl_weight: Weight for KL term in loss function, which is the weighted sum of
            reconstruction and KL losses.
        x: Input data array.
        key: Key for random generation in VAE forward pass.
        y: Conditioning data array.

    Returns:
        Loss evaluated.
    """
    # Merge at the beginning of the function
    model, optimizer, metrics = nnx.merge(graphdef, state)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, criterion, kl_loss_fn, kl_weight, x, key, y)
    snr = snr_fn(x, aux[1])
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss, kl=aux[0], snr=snr)  # In-place updates.

    state = nnx.state((model, optimizer, metrics))
    return loss, state


@nnx.jit(static_argnums=(1, 2))
def eval_step_vae(
    model: Callable,
    criterion: Callable,
    kl_loss_fn: Callable,
    kl_weight: float,
    metrics: nnx.MultiMetric,
    x: ArrayLike,
    key: ArrayLike,
    y: ArrayLike,
) -> ArrayLike:
    """Evaluate VAE for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing batch.

    Args:
        model: Model to train.
        criterion: Criterion to evaluate autoencoder reconstruction.
        kl_loss_fn: Function to evaluate KL divergence in latent space.
        kl_weight: Weight for KL term in loss function which is the weighted combination of
            reconstruction and KL losses.
        metrics: Dictionary of metrics to evaluate.
        x: Input data array.
        key: Key for random generation in VAE forward pass.
        y: Conditioning data array.

    Returns:
        Loss evaluated.
    """
    # with model.eval():
    #    loss, aux = loss_fn(model, criterion, kl_loss_fn, kl_weight, x, key, y)
    loss, aux = loss_fn(model, criterion, kl_loss_fn, kl_weight, x, key, y)
    snr = snr_fn(x, aux[1])
    metrics.update(loss=loss, kl=aux[0], snr=snr)  # In-place updates.

    return loss


@jax.jit(static_argnums=0)
def generate_sample(
    model, key: ArrayLike, num_samples: Optional[int] = None, c: Optional[ArrayLike] = None
) -> ArrayLike:
    """Generate samples from latent representation.

    Args:
        key: The jax key for the random generation of sample.
        num_samples: Number of samples to generate. Applies if no conditioning is provided.
        c: Conditioning signal.

    Returns:
        Generated samples.
    """
    if c is None:
        assert num_samples is not None
        z = jax.random.normal(key, (num_samples, model.encoder.latent_dim))
        y = jax.jit(model.decode)(z)
    else:
        z = jax.random.normal(key, (c.shape[0], model.encoder.latent_dim))
        y = jax.jit(model.decode_cond)(z, c)
    return y
