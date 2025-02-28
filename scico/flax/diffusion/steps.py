# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation of
diffusion models."""

import sys
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

import optax

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from flax.training.train_state import TrainState

eps = 1e-5  # Minimum t to sample


class DiffusionMetricsDict(TypedDict, total=False):
    """Dictionary structure for training metrics for diffusion
    generative models.

    Definition of the dictionary structure for metrics computed or
    updates made during training.
    """

    loss: float
    std: float
    learning_rate: float


def train_step_diffusion(
    state: TrainState,
    batch: ArrayLike,
    key: ArrayLike,
    stddev_prior: float,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    **kwargs,
) -> Tuple[TrainState, DiffusionMetricsDict]:
    """Perform a single data parallel training step.

    Assumes sharded batched data. This function is intended to be used via
    :class:`~scico.flax.BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the model apply function,
            the model parameters and an Optax optimizer.
        batch: Sharded and batched training data. Only input data is
            passed (i.e. no label is passed since the prediction must
            correspond to data similar to the input).
        key: Key for random generation.
        stddev_prior: Standard deviation of prior noise.
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
    tshp = (batch["image"].shape[0],) + (1,) * len(batch["image"].shape[1:])
    batch_t = jax.random.uniform(step_key, (batch["image"].shape[0], 1), minval=eps, maxval=1.0)
    t = batch_t.reshape(tshp)
    key, step_key = jax.random.split(key)
    z = jax.random.normal(step_key, batch["image"].shape)
    std = jnp.sqrt((stddev_prior ** (2 * t) - 1.0) / 2.0 / jnp.log(stddev_prior))
    batch_x = batch["image"] + z * std

    def loss_fn(params):
        output = state.apply_fn({"params": params}, batch_x, batch_t)
        loss = criterion(output * std, -z)
        return loss

    step = state.step
    # Only to figure out current learning rate, which cannot be stored in stateless optax.
    # Requires agreement between the function passed here and the one used to create the
    # train state.
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name="batch")
    loss = lax.pmean(loss, axis_name="batch")
    metrics: DiffusionMetricsDict = {"loss": loss}
    metrics = lax.pmean(metrics, axis_name="batch")
    metrics["std"] = std[0]
    metrics["learning_rate"] = lr

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def eval_step_diffusion(
    state: TrainState,
    batch: ArrayLike,
    criterion: Callable,
    key: ArrayLike,
    stddev_prior: float,
    **kwargs,
) -> Tuple[TrainState, DiffusionMetricsDict]:
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
        stddev_prior: Standard deviation of prior noise.

    Returns:
        Current diagnostic statistics.
    """

    key, step_key = jax.random.split(key)
    tshp = (batch["image"].shape[0],) + (1,) * len(batch["image"].shape[1:])
    batch_t = jax.random.uniform(step_key, (batch["image"].shape[0], 1), minval=eps, maxval=1.0)
    t = batch_t.reshape(tshp)
    key, step_key = jax.random.split(key)
    z = jax.random.normal(step_key, batch["image"].shape)
    std = jnp.sqrt((stddev_prior ** (2 * t) - 1.0) / 2.0 / jnp.log(stddev_prior))
    batch_x = batch["image"] + z * std
    output = state.apply_fn({"params": state.params}, batch_x, batch_t)
    loss = criterion(output * std, -z)
    metrics: DiffusionMetricsDict = {"loss": loss}
    metrics = lax.pmean(metrics, axis_name="batch")
    metrics["std"] = std[0]
    return metrics
