# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
diffusion models."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

eps = 1e-5  # Minimum t to sample


def _step_t(batch_x: ArrayLike, key: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    """Default t computation for diffusion step."""
    tshp = (batch_x.shape[0],) + (1,) * len(batch_x.shape[1:])
    batch_t = jax.random.uniform(key, (batch_x.shape[0], 1), minval=eps, maxval=1.0)
    t = batch_t.reshape(tshp)
    return t, batch_t


def _step_x(
    batch: ArrayLike, key: ArrayLike, t: ArrayLike, **kwargs
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Default x computation for diffusion step."""
    stddev_prior = kwargs.pop("stddev_prior")
    z = jax.random.normal(key, batch.shape)
    batch_std = jnp.sqrt((stddev_prior ** (2 * t) - 1.0) / 2.0 / jnp.log(stddev_prior))
    batch_x = batch + z * batch_std
    return z, batch_std, batch_x


def _step_loss(
    criterion: Callable, z: ArrayLike, std: ArrayLike, output: ArrayLike, **kwargs
) -> ArrayLike:
    """Default loss computation for diffusion step."""
    return criterion(output * std, -z)


def loss_fn(
    model: Callable,
    criterion: Callable,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_t: ArrayLike,
    batch_z: ArrayLike,
    batch_std: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Loss function definition for diffusion step..

    Args:
        model: Model to train.
        criterion: Criterion to evaluate model.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        batch_x: Noisy image data array.
        batch_t: Time embedding array.
        batch_z: Standard Gaussian random variable array.
        batch_std: Corresponding noise level array.

    Returns:
        Evaluated loss.
    """
    output = model(batch_x, batch_t)
    loss = step_loss(criterion, batch_z, batch_std, output).mean()
    return loss


@nnx.jit(static_argnums=(1, 3))
def eval_step_diffusion(
    model: Callable,
    criterion: Callable,
    metrics: nnx.MultiMetric,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_t: ArrayLike,
    batch_z: ArrayLike,
    batch_std: ArrayLike,
) -> ArrayLike:
    """Evaluate for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.

    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        metrics: Dictionary of metrics to evaluate.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        batch_x: Noisy image data array.
        batch_t: Time embedding array.
        batch_z: Standard Gaussian random variable array.
        batch_std: Corresponding noise level array.

    Returns:
        Loss evaluated.
    """

    # with model.eval():
    #    loss = loss_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    loss = loss_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    metrics.update(loss=loss)  # In-place updates.

    return loss


# @jax.jit(static_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def jax_train_step_diffusion(
    graphdef,
    state,
    criterion: Callable,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_t: ArrayLike,
    batch_z: ArrayLike,
    batch_std: ArrayLike,
) -> ArrayLike:
    """Train for a single step.

    This function uses data and a criterion to optimize model parameters. It returns
    the current loss in the training set.

    Args:
        graphdef: Graph representation of model.
        state: NNX state object including pytrees for all the
            model and optimizer graph nodes.
        criterion: Criterion to use for training.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        batch_x: Noisy image data array.
        batch_t: Time embedding array.
        batch_z: Standard Gaussian random variable array.
        batch_std: Corresponding noise level array.

    Returns:
        Loss evaluated.
    """
    # merge at the beginning of the function
    model, optimizer, metrics = nnx.merge(graphdef, state)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss)  # In-place updates.

    state = nnx.state((model, optimizer, metrics))
    return loss, state
