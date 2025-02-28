# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
variational autoencoders."""

import sys
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

import optax

PyTree = Any

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from flax.training.train_state import TrainState


class VAEMetricsDict(TypedDict, total=False):
    """Dictionary structure for training metrics for variational
    autoencoder models.

    Definition of the dictionary structure for metrics computed or
    updates made during training.
    """

    loss: float
    mse: float
    kl: float
    learning_rate: float


def kl_loss_fn(mean, logvar):
    """Compute KL divergence loss from given mean and log variance.

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
        kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=reduce_dims)
        )

    return kl_loss


def train_step_vae(
    state: TrainState,
    batch: ArrayLike,
    key: ArrayLike,
    kl_weight: float,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    **kwargs,
) -> Tuple[TrainState, VAEMetricsDict]:
    """Perform a single data parallel training step.

    Assumes sharded batched data. This function is intended to be used via
    :class:`~scico.flax.BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the model apply function,
            the model parameters and an Optax optimizer.
        batch: Sharded and batched training data. Only input data is
            passed (i.e. no label is passed since the prediction must
            correspond to the input).
        key: Key for random generation.
        kl_weight: Weight of the KL divergence term in the total training loss.
        learning_rate_fn: A function to map step
            counts to values. This is only used for display purposes
            (optax optimizers are stateless, so the current learning rate
            is not stored). The real learning rate schedule applied is the
            one defined when creating the Flax state. If a different
            object is passed here, then the displayed value will be
            inaccurate.
        criterion: A function that specifies the loss being minimized in
            training.

    Returns:
        Updated parameters and diagnostic statistics.
    """

    key, step_key = jax.random.split(key)

    def loss_fn(params: PyTree, x: ArrayLike, key: ArrayLike):
        """Loss function used for training."""
        reduce_dims = list(range(1, len(x.shape)))
        output, mean, logvar = state.apply_fn(
            {
                "params": params,
            },
            x,
            key,
        )

        mse_loss = criterion(output, x).sum(axis=reduce_dims).mean()
        # KL loss term to keep encoder output close to standard
        # normal distribution.
        kl_loss = kl_loss_fn(mean, logvar)
        loss = mse_loss + kl_weight * kl_loss
        losses = (loss, mse_loss, kl_loss)
        return loss, losses

    step = state.step
    # Only to figure out current learning rate, which cannot be stored in stateless optax.
    # Requires agreement between the function passed here and the one used to create the
    # train state.
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params, batch["image"], step_key)
    losses = aux[1]
    # Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name="batch")
    metrics: VAEMetricsDict = {"loss": losses[0], "mse": losses[1], "kl": losses[2]}
    metrics = lax.pmean(metrics, axis_name="batch")
    metrics["learning_rate"] = lr

    # Update params
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, metrics


def train_step_vae_class_conditional(
    state: TrainState,
    batch: ArrayLike,
    num_classes: int,
    key: ArrayLike,
    kl_weight: float,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    **kwargs,
) -> Tuple[TrainState, VAEMetricsDict]:
    """Perform a single data parallel training step using class
    conditional information.

    Assumes sharded batched data. This function is intended to be used via
    :class:`~scico.flax.BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the model apply function,
            the model parameters and an Optax optimizer.
        batch: Sharded and batched training data. Data as well as class
            labels are passed.
        num_classes: Number of classes in dataset.
        key: Key for random generation.
        kl_weight: Weight of the KL divergence term in the total training loss.
        learning_rate_fn: A function to map step
            counts to values. This is only used for display purposes
            (optax optimizers are stateless, so the current learning rate
            is not stored). The real learning rate schedule applied is the
            one defined when creating the Flax state. If a different
            object is passed here, then the displayed value will be
            inaccurate.
        criterion: A function that specifies the loss being minimized in
            training.

    Returns:
        Updated parameters and diagnostic statistics.
    """

    key, step_key = jax.random.split(key)

    def loss_fn(params: PyTree, x: ArrayLike, c: ArrayLike, key: ArrayLike):
        """Loss function used for training."""
        c = jax.nn.one_hot(c, num_classes).squeeze()  # one hot encode the class index
        output, mean, logvar = state.apply_fn(
            {
                "params": params,
            },
            x,
            key,
            c,
        )
        reduce_dims = list(range(1, len(x.shape)))
        mse_loss = criterion(output, x).sum(axis=reduce_dims).mean()
        # KL loss term to keep encoder output close to standard
        # normal distribution.
        kl_loss = kl_loss_fn(mean, logvar)
        loss = mse_loss + kl_weight * kl_loss
        losses = (loss, mse_loss, kl_loss)
        return loss, losses

    step = state.step
    # Only to figure out current learning rate, which cannot be stored in stateless optax.
    # Requires agreement between the function passed here and the one used to create the
    # train state.
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params, batch["image"], batch["label"], step_key)
    losses = aux[1]
    # Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name="batch")
    metrics: VAEMetricsDict = {"loss": losses[0], "mse": losses[1], "kl": losses[2]}
    metrics = lax.pmean(metrics, axis_name="batch")
    metrics["learning_rate"] = lr

    # Update params
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, metrics


def eval_step_vae(
    state: TrainState,
    batch: ArrayLike,
    criterion: Callable,
    key: ArrayLike,
    kl_weight: float,
    **kwargs,
) -> VAEMetricsDict:
    """Evaluate current model state.

    Assumes sharded batched data. This function is intended to be used
    via :class:`~scico.flax.BasicFlaxTrainer` or
    :meth:`~scico.flax.only_evaluate`, not directly.

    Args:
        state: Flax train state which includes the model apply function
            and the model parameters.
        batch: Sharded and batched training data.
        criterion: Loss function.
        key: Key for random generation.
        kl_weight: Weight of the KL divergence term in the total training loss.

    Returns:
        Current diagnostic statistics.
    """
    variables = {
        "params": state.params,
    }
    key, step_key = jax.random.split(key)
    output, mean, logvar = state.apply_fn(variables, batch["image"], step_key)
    reduce_dims = list(range(1, len(batch["image"].shape)))
    mse_loss = criterion(output, batch["image"]).sum(axis=reduce_dims).mean()
    # KL loss term to keep encoder output close to standard
    # normal distribution.
    kl_loss = kl_loss_fn(mean, logvar)
    loss = mse_loss + kl_weight * kl_loss
    metrics: VAEMetricsDict = {"loss": loss, "mse": mse_loss, "kl": kl_loss}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def eval_step_vae_class_conditional(
    state: TrainState,
    batch: ArrayLike,
    num_classes: int,
    criterion: Callable,
    key: ArrayLike,
    kl_weight: float,
    **kwargs,
) -> VAEMetricsDict:
    """Evaluate current model state using class
    conditional information.

    Assumes sharded batched data. This function is intended to be used
    via :class:`~scico.flax.BasicFlaxTrainer` or
    :meth:`~scico.flax.only_evaluate`, not directly.

    Args:
        state: Flax train state which includes the model apply function
            and the model parameters.
        batch: Sharded and batched training data.
        num_classes: Number of classes in dataset.
        criterion: Loss function.
        key: Key for random generation.
        kl_weight: Weight of the KL divergence term in the total training loss.

    Returns:
        Current diagnostic statistics.
    """
    variables = {
        "params": state.params,
    }
    key, step_key = jax.random.split(key)
    c = jax.nn.one_hot(batch["label"], num_classes).squeeze()  # one hot encode the class index
    output, mean, logvar = state.apply_fn(variables, batch["image"], step_key, c)
    reduce_dims = list(range(1, len(batch["image"].shape)))
    mse_loss = criterion(output, batch["image"]).sum(axis=reduce_dims).mean()
    # KL loss term to keep encoder output close to standard
    # normal distribution.
    kl_loss = kl_loss_fn(mean, logvar)
    loss = mse_loss + kl_weight * kl_loss
    metrics: VAEMetricsDict = {"loss": loss, "mse": mse_loss, "kl": kl_loss}
    return metrics


def build_sample_fn(model: Callable, params: PyTree):
    """Function to generate samples from model.

    Args:
        model: Variational autoencoder model to generate samples from.
        params: Parameters of trained model.
    """

    @jax.jit
    def sample_fn(z: ArrayLike) -> ArrayLike:
        """Generate samples from latent representation.

        Args:
            z: Representation in latent space.
        """
        return model.apply(params, z, method=model.decode)

    return sample_fn


def build_sample_conditional_fn(model: Callable, params: PyTree):
    """Function to generate samples from model.

    Args:
        model: Variational autoencoder model to generate samples from.
        params: Parameters of trained model.
    """

    @jax.jit
    def sample_fn(z: ArrayLike, c: ArrayLike) -> ArrayLike:
        """Generate samples from latent representation conditioned
        on sample class.

        Args:
            z: Representation in latent space.
            c: Class to generate samples from.
        """
        return model.apply(params, z, c, method=model.decode_cond)

    return sample_fn


def build_sample_conditional_return_latent_fn(model, params):
    """Function to generate samples from model and return intermediate
    decodings too.

    Args:
        model: Variational autoencoder model to generate samples from.
        params: Parameters of trained model.
    """

    @jax.jit
    def sample_fn(z, c):
        """Generate samples from latent representation conditioned
        on sample class and return intermediate representations too.

        Args:
            z: Representation in latent space.
            c: Class to generate samples from.
        """
        return model.apply(params, z, c, method=model.decode_cond_return_latent)

    return sample_fn
