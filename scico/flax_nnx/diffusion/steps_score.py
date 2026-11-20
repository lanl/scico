# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Steps to iterate during training or evaluation of score models.

Applies to diffusion models under score, i.e. the gradient of the log of
the data distribution. This interface uses time in (0, 1]."""

from __future__ import annotations  # for type support python 3.9 or earlier

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from flax import nnx
from scico.numpy import Array

#: Smallest diffusion time sampled by :func:`_step_t`, keeping `t` away
#: from zero where the score is not well behaved.
eps = 1e-5


def _step_t(batch_x: Array, key: jax.Array, **kwargs: Any) -> tuple[Array, Array]:
    """Default t computation for diffusion step.

    Args:
        batch_x: Array of a batch of original data.
        key: Key for random generation.
        kwargs: Unused; accepted so that all `step_t` implementations
            share a signature.

    Returns:
        Time steps shaped for broadcasting against `batch_x`, and the
        same values shaped `(batch, 1)`, both drawn uniformly from
        `[eps, 1]`.
    """
    tshp = (batch_x.shape[0],) + (1,) * len(batch_x.shape[1:])
    batch_t = jax.random.uniform(key, (batch_x.shape[0], 1), minval=eps, maxval=1.0)
    t = batch_t.reshape(tshp)
    return t, batch_t


def _step_x(batch: Array, key: jax.Array, t: Array, **kwargs: Any) -> tuple[Array, Array, Array]:
    """Default x computation for diffusion step.

    Args:
        batch: Array of a batch of original data.
        key: Key for random generation.
        t: Array of a batch of time steps in the diffusion process.
        kwargs: Keyword arguments. Must include `stddev_prior`, the
            standard deviation of the prior noise; the trainer binds it
            via :func:`functools.partial`.

    Returns:
        Noise generated, equivalent standard deviation and noisy data.

    Raises:
        KeyError: If `stddev_prior` is not in `kwargs`.
    """
    stddev_prior = kwargs.pop("stddev_prior")
    z = jax.random.normal(key, batch.shape)
    batch_std = jnp.sqrt((stddev_prior ** (2 * t) - 1.0) / 2.0 / jnp.log(stddev_prior))
    batch_x = batch + z * batch_std
    return z, batch_std, batch_x


def _step_loss(criterion: Callable, z: Array, std: Array, output: Array, **kwargs: Any) -> Array:
    """Default loss computation for diffusion step.

    Args:
        criterion: Criterion to use for training.
        z: Noise added to the batch.
        std: Standard deviation corresponding to the sampled time steps.
        output: Model output.
        kwargs: Unused; accepted so that all `step_loss` implementations
            share a signature.

    Returns:
        Loss between the scaled model output and the negated noise.
    """
    return criterion(output * std, -z)


def loss_fn(
    model: nnx.Module,
    criterion: Callable,
    step_loss: Callable,
    batch_x: Array,
    batch_t: Array,
    batch_z: Array,
    batch_std: Array,
) -> Array:
    """Loss function definition for diffusion step.

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


# @jax.jit(static_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def jax_train_step_diffusion(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    criterion: Callable,
    step_loss: Callable,
    batch_x: Array,
    batch_t: Array,
    batch_z: Array,
    batch_std: Array,
) -> tuple[Array, nnx.State]:
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
        Evaluated loss and the updated NNX state.
    """
    # merge at the beginning of the function
    model, optimizer, metrics = nnx.merge(graphdef, state)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss)  # In-place updates.

    state = nnx.state((model, optimizer, metrics))
    return loss, state


@nnx.jit(static_argnums=(1, 2))
def eval_step_diffusion(
    model: nnx.Module,
    criterion: Callable,
    step_loss: Callable,
    metrics: nnx.MultiMetric,
    batch_x: Array,
    batch_t: Array,
    batch_z: Array,
    batch_std: Array,
) -> Array:
    """Evaluate for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.

    Args:
        model: Model to evaluate.
        criterion: Criterion to use for evaluation.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        metrics: Multi-metric object in which to accumulate metrics.
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
