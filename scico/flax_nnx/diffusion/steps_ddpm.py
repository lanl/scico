# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
diffusion models under Denoising Diffusion Probabilistic Models (DDPM)
formulation."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx


def alpha_fn(alpha_bar_schedule: ArrayLike, batch_t: ArrayLike) -> ArrayLike:
    """Computation of alpha for the given time steps in DDPM formulation.

    Alpha is related to the forward process variances beta by:
    :math:`\alpha_t = 1 - \beta_t`.

    Args:
        alpha_var_schedule: Array of alpha products computed for the whole t range.
        batch_t: Array of time steps to evaluate alpha.

    Returns:
        Array of alpha values evaluated for the given time steps.
    """
    alpha_eval = jnp.zeros(batch_t.shape[0])
    for i, tt in enumerate(batch_t):
        alpha_eval = alpha_eval.at[i].set(alpha_bar_schedule[tt - 1])
    return alpha_eval


def _step_t(batch_x: ArrayLike, key: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    """t computation for diffusion step in DDPM formulation.

    Args:
        batch_x: Array of a batch of original data.
        key: Key for random generation.
        kwargs: Keyword arguments. Relevant here: maximum number of steps in t range.

    Returns:
        Random generation of a batch of time steps for training.
    """
    maxsteps = kwargs.pop("maxsteps")
    batch_t = jax.random.randint(key, (batch_x.shape[0], 1), minval=1, maxval=maxsteps)
    return batch_t


def _step_x(
    batch: ArrayLike, key: ArrayLike, t: ArrayLike, **kwargs
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """x computation for diffusion step in DDPM formulation.

    Args:
        batch: Array of a batch of original data.
        key: Key for random generation.
        t: Array of a batch of time steps in the diffusion process.
        kwargs: Keyword arguments. Relevant here: the processed :math:`\bar{\alpha}_t`.

    Returns:
        Noise generated, equivalent standard deviation and noisy data.
    """
    alpha_bar_schedule = kwargs.pop("alpha_bar_schedule")
    z = jax.random.normal(key, batch.shape)
    shp = (batch.shape[0],) + (1,) * len(batch.shape[1:])
    batch_alpha_bar = alpha_fn(alpha_bar_schedule, t).reshape(shp)
    batch_std = jnp.sqrt((1.0 - batch_alpha_bar) / batch_alpha_bar)
    batch_x = jnp.sqrt(batch_alpha_bar) * batch + jnp.sqrt(1.0 - batch_alpha_bar) * z
    return z, batch_std, batch_x


def _step_loss(criterion: Callable, z: ArrayLike, output: ArrayLike, **kwargs) -> ArrayLike:
    """Loss computation for diffusion step in DDPM formulation."""
    return criterion(output, z)


def loss_fn(
    model: Callable,
    criterion: Callable,
    step_loss: Callable,
    batch_x: ArrayLike,
    batch_t: ArrayLike,
    batch_z: ArrayLike,
    batch_std: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Loss function definition for diffusion step.

    Args:
        model: Model to train.
        criterion: Criterion to evaluate model.
        step_loss: Function to evaluate difference between expected
                    and produced outputs.
        batch_x: Noisy image data array.
        batch_t: Time array.
        batch_z: Standard Gaussian random variable array.
        batch_std: Corresponding noise level array.

    Returns:
        Evaluated loss.
    """
    output = model(batch_x, batch_t)
    loss = step_loss(criterion, batch_z, output).mean()
    return loss


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
        batch_t: Time array.
        batch_z: Standard Gaussian random noise array.
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


@nnx.jit(static_argnums=(1, 2))
def eval_step_diffusion(
    model: Callable,
    criterion: Callable,
    step_loss: Callable,
    metrics: nnx.MultiMetric,
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
