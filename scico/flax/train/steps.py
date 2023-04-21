# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation."""

from typing import Any, Callable, List, Tuple, Union

import jax
from jax import lax

import optax

from scico.numpy import Array

from .state import TrainState
from .typed_dict import DataSetDict, MetricsDict

KeyArray = Union[Array, jax.random.PRNGKeyArray]
PyTree = Any


def train_step(
    state: TrainState,
    batch: DataSetDict,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    metrics_fn: Callable,
) -> Tuple[TrainState, MetricsDict]:
    """Perform a single data parallel training step.

    Assumes sharded batched data. This function is intended to be used via
    :class:`~scico.flax.BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the model apply function,
            the model parameters and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function to map step
            counts to values. This is only used for display purposes
            (optax optimizers are stateless, so the current learning rate
            is not stored). The real learning rate schedule applied is the
            one defined when creating the Flax state. If a different
            object is passed here, then the displayed value will be
            inaccurate.
        criterion: A function that specifies the loss being minimized in
            training.
        metrics_fn: A function to evaluate quality of current model.

    Returns:
        Updated parameters and diagnostic statistics.
    """

    def loss_fn(params: PyTree):
        """Loss function used for training."""
        output, new_model_state = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats,
            },
            batch["image"],
            mutable=["batch_stats"],
        )
        loss = criterion(output, batch["label"])
        return loss, (new_model_state, output)

    step = state.step
    # Only to figure out current learning rate, which cannot be stored in stateless optax.
    # Requires agreement between the function passed here and the one used to create the
    # train state.
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in call to pmap
    grads = lax.pmean(grads, axis_name="batch")
    new_model_state, output = aux[1]
    metrics = metrics_fn(output, batch["label"], criterion)
    metrics["learning_rate"] = lr

    # Update params and stats
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state["batch_stats"],
    )

    return new_state, metrics


def train_step_post(
    state: TrainState,
    batch: DataSetDict,
    learning_rate_fn: optax._src.base.Schedule,
    criterion: Callable,
    train_step_fn: Callable,
    metrics_fn: Callable,
    post_lst: List[Callable],
) -> Tuple[TrainState, MetricsDict]:
    """Perform a single data parallel training step with postprocessing.

    A list of postprocessing functions (i.e. for spectral normalization
    or positivity condition, etc.) is applied after the gradient update.
    Assumes sharded batched data.

    This function is intended to be used via
    :class:`~scico.flax.BasicFlaxTrainer`, not directly.

    Args:
        state: Flax train state which includes the model apply function,
            the model parameters and an Optax optimizer.
        batch: Sharded and batched training data.
        learning_rate_fn: A function to map step counts to values.
        criterion: A function that specifies the loss being minimized in
            training.
        train_step_fn: A function that executes a training step.
        metrics_fn: A function to evaluate quality of current model.
        post_lst: List of postprocessing functions to apply to parameter
            set after optimizer step (e.g. clip to a specified range,
            normalize, etc.).

    Returns:
        Updated parameters, fulfilling additional constraints, and
        diagnostic statistics.
    """

    new_state, metrics = train_step_fn(state, batch, learning_rate_fn, criterion, metrics_fn)

    # Post-process parameters
    for post_fn in post_lst:
        new_params = post_fn(new_state.params)
        new_state = new_state.replace(params=new_params)

    return new_state, metrics


def eval_step(
    state: TrainState, batch: DataSetDict, criterion: Callable, metrics_fn: Callable
) -> MetricsDict:
    """Evaluate current model state.

    Assumes sharded batched data. This function is intended to be used
    via :class:`~scico.flax.BasicFlaxTrainer` or
    :meth:`~scico.flax.only_evaluate`, not directly.

    Args:
        state: Flax train state which includes the model apply function
            and the model parameters.
        batch: Sharded and batched training data.
        criterion: Loss function.
        metrics_fn: A function to evaluate quality of current model.

    Returns:
        Current diagnostic statistics.
    """
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }
    output = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return metrics_fn(output, batch["label"], criterion)
