# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
diffusion models under score, i.e. gradient of the log of the data
distribution, formulation :cite:`song-2019-score`."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx


def set_sigma_schedule(sigma_min: float, sigma_max: float, max_sigma_steps: int) -> ArrayLike:
    """Computation of noise schedule for score matching formulation.

    Args:
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        max_sigma_steps: Maximum number of noise levels.

    Returns:
        Array of noise levels.
    """
    return jnp.geomspace(sigma_max, sigma_min, max_sigma_steps)


def _step_x(batch: ArrayLike, key: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """x computation for diffusion step in score matching formulation.

    Args:
        batch: Array of a batch of original data.
        key: Key for random generation.
        kwargs: Keyword arguments. Relevant here: the processed :math:`\sigma_i`.

    Returns:
        Noise generated, noise std and noisy data.
    """
    sigma_schedule = kwargs.pop("sigma_schedule")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    z = jax.random.normal(subkey1, batch.shape)
    shp = (batch.shape[0],) + (1,) * len(batch.shape[1:])
    batch_sigma = jax.random.choice(
        subkey2, sigma_schedule, shape=(batch.shape[0], 1), replace=True
    )
    noise = batch_sigma.reshape(shp) ** 2 * z
    batch_x = batch + noise

    return noise, batch_sigma, batch_x


def _step_loss(
    criterion: Callable, noise: ArrayLike, sigma: ArrayLike, output: ArrayLike, **kwargs
) -> ArrayLike:
    """Default loss computation for diffusion step."""
    return criterion(output * sigma**2, -noise)


def loss_fn(
    model: Callable,
    criterion: Callable,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_sigma: ArrayLike,
    noise: ArrayLike,
) -> ArrayLike:
    """Loss function definition for diffusion step..

    Args:
        model: Model to train.
        criterion: Criterion to evaluate model.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        batch_x: Noisy image data array.
        batch_sigma: Standard deviation of noise applied.
        noise: Noise array.

    Returns:
        Evaluated loss.
    """
    output = model(batch_x, batch_sigma)
    shp = (batch_x.shape[0],) + (1,) * len(batch_x.shape[1:])
    loss = step_loss(criterion, noise, batch_sigma.reshape(shp), output).mean()
    return loss


# @jax.jit(static_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def jax_train_step_diffusion(
    graphdef,
    state,
    criterion: Callable,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_sigma: ArrayLike,
    noise: ArrayLike,
) -> Tuple[ArrayLike, nnx.State]:
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
        batch_sigma: Standard deviation of noise applied.
        noise: Noise array.

    Returns:
        Loss evaluated.
    """
    # merge at the beginning of the function
    model, optimizer, metrics = nnx.merge(graphdef, state)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, criterion, step_loss, batch_x, batch_sigma, noise)
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss)  # In-place updates.

    state = nnx.state((model, optimizer, metrics))
    return loss, state


@nnx.jit(static_argnums=(1, 2))
def eval_step_diffusion(
    model: Callable,
    criterion: Callable,
    step_loss: Callable,
    metrics: nnx.MultiMetric,
    batch_x: ArrayLike,
    batch_sigma: ArrayLike,
    noise: ArrayLike,
) -> ArrayLike:
    """Evaluate for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.

    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        metrics: Dictionary of metrics to evaluate.
        batch_x: Noisy image data array.
        batch_sigma: Standard deviation of noise applied.
        noise: Noise array.

    Returns:
        Loss evaluated.
    """

    # with model.eval():
    #    loss = loss_fn(model, criterion, step_loss, batch_x, batch_sigma, noise)
    loss = loss_fn(model, criterion, step_loss, batch_x, batch_sigma, noise)
    metrics.update(loss=loss)  # In-place updates.

    return loss
