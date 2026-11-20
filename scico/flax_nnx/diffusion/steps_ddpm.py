# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Steps to iterate during training or evaluation of DDPM models.

Applies to diffusion models under the Denoising Diffusion Probabilistic
Models (DDPM) formulation :cite:`ho-2020-ddpm`."""

from __future__ import annotations  # for type support python 3.9 or earlier

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from flax import nnx
from scico.numpy import Array


def alpha_fn(alpha_bar_schedule: Array, batch_t: Array) -> Array:
    r"""Computation of alpha bar for the given time steps in DDPM formulation.

    Alpha bar is the cumulative product
    :math:`\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s` of the quantities
    :math:`\alpha_t = 1 - \beta_t` determined by the forward process
    variances :math:`\beta_t`.

    Args:
        alpha_bar_schedule: Array of alpha products computed for the whole t range.
        batch_t: Array of time steps at which to evaluate alpha bar.

    Returns:
        Array of alpha bar values evaluated for the given time steps.
    """
    alpha_eval = jnp.zeros(batch_t.shape[0])
    for i, tt in enumerate(batch_t):
        alpha_eval = alpha_eval.at[i].set(alpha_bar_schedule[tt - 1])
    return alpha_eval


def _step_t(batch_x: Array, key: jax.Array, **kwargs: Any) -> Array:
    """t computation for diffusion step in DDPM formulation.

    Args:
        batch_x: Array of a batch of original data.
        key: Key for random generation.
        kwargs: Keyword arguments. Must include `maxsteps`, the maximum
            number of steps in the t range; the trainer binds it via
            :func:`functools.partial`.

    Returns:
        Random generation of a batch of time steps for training.

    Raises:
        KeyError: If `maxsteps` is not in `kwargs`.
    """
    maxsteps = kwargs.pop("maxsteps")
    batch_t = jax.random.randint(key, (batch_x.shape[0], 1), minval=1, maxval=maxsteps)
    return batch_t


def _step_x(batch: Array, key: jax.Array, t: Array, **kwargs: Any) -> tuple[Array, Array, Array]:
    r"""x computation for diffusion step in DDPM formulation.

    Args:
        batch: Array of a batch of original data.
        key: Key for random generation.
        t: Array of a batch of time steps in the diffusion process.
        kwargs: Keyword arguments. Must include `alpha_bar_schedule`,
            the processed :math:`\bar{\alpha}_t`; the trainer binds it
            via :func:`functools.partial`.

    Returns:
        Noise generated, equivalent standard deviation and noisy data.

    Raises:
        KeyError: If `alpha_bar_schedule` is not in `kwargs`.
    """
    alpha_bar_schedule = kwargs.pop("alpha_bar_schedule")
    z = jax.random.normal(key, batch.shape)
    shp = (batch.shape[0],) + (1,) * len(batch.shape[1:])
    batch_alpha_bar = alpha_fn(alpha_bar_schedule, t).reshape(shp)
    batch_std = jnp.sqrt((1.0 - batch_alpha_bar) / batch_alpha_bar)
    batch_x = jnp.sqrt(batch_alpha_bar) * batch + jnp.sqrt(1.0 - batch_alpha_bar) * z
    return z, batch_std, batch_x


def _step_loss(criterion: Callable, z: Array, output: Array, **kwargs: Any) -> Array:
    """Loss computation for diffusion step in DDPM formulation.

    Args:
        criterion: Criterion to use for training.
        z: Noise added to the batch.
        output: Model output.
        kwargs: Unused; accepted so that all `step_loss` implementations
            share a signature. Note that, unlike the score variant, this
            one takes no standard deviation.

    Returns:
        Loss between the model output and the noise.
    """
    return criterion(output, z)


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
        batch_t: Time array.
        batch_z: Standard Gaussian random variable array.
        batch_std: Corresponding noise level array. Not used by the DDPM
            step loss; accepted so that all the diffusion step variants
            share a single signature.

    Returns:
        Evaluated loss.
    """
    output = model(batch_x, batch_t)
    loss = step_loss(criterion, batch_z, output).mean()
    return loss


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
        batch_t: Time array.
        batch_z: Standard Gaussian random noise array.
        batch_std: Corresponding noise level array. Not used by the DDPM
            step loss; accepted so that all the diffusion step variants
            share a single signature.

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
        batch_std: Corresponding noise level array. Not used by the DDPM
            step loss; accepted so that all the diffusion step variants
            share a single signature.

    Returns:
        Loss evaluated.
    """

    # with model.eval():
    #    loss = loss_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    loss = loss_fn(model, criterion, step_loss, batch_x, batch_t, batch_z, batch_std)
    metrics.update(loss=loss)  # In-place updates.

    return loss
