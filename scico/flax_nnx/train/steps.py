# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of steps to iterate during training or evaluation."""

from functools import partial
from typing import Callable, Tuple

import jax
from jax.typing import ArrayLike

from flax import nnx
from scico.metric import snr as snr_fn


def loss_fn(
    model: Callable, criterion: Callable, x: ArrayLike, y: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Loss function definition.

    Args:
        model: Model to train.
        criterion: Criterion to evaluate difference between expected
                    and produced outputs.
        x: Input (features) array.
        y: Output (labels) array.
    """
    output = model(x)
    loss = criterion(output, y).mean()
    return loss, output


@nnx.jit(static_argnums=1)
def train_step(
    model: Callable,
    criterion: Callable,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    x: ArrayLike,
    y: ArrayLike,
) -> ArrayLike:
    """Train for a single step.

    This function uses data and a criterion to optimize model parameters. It returns
    the current loss in the training set.

    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        optimizer: NNX optimizer object used to train model.
        metrics: Dictionary of metrics to evaluate.
        x: Input (features) array.
        y: Output (labels) array.

    Returns:
        Loss evaluated.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, output), grads = grad_fn(model, criterion, x, y)
    snr = snr_fn(y, output)
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss, snr=snr)  # In-place updates.
    return loss


@nnx.jit(static_argnums=1)
def eval_step(
    model: Callable, criterion: Callable, metrics: nnx.MultiMetric, x: ArrayLike, y: ArrayLike
) -> ArrayLike:
    """Evaluate for a single step.

    This function uses data and a criterion to evaluate performance of current model.
    It returns the current loss evaluated in the testing set.

    Args:
        model: Model to train.
        criterion: Criterion to use for training.
        metrics: Dictionary of metrics to evaluate.
        x: Input (features) array.
        y: Output (labels) array.

    Returns:
        Loss evaluated.
    """
    # with model.eval():
    #    loss, output = loss_fn(model, criterion, x, y)
    loss, output = loss_fn(model, criterion, x, y)
    snr = snr_fn(y, output)
    metrics.update(loss=loss, snr=snr)  # In-place updates.

    return loss


# @jax.jit(static_argnums=2)
@partial(jax.jit, static_argnums=2)
def jax_train_step(graphdef, state, criterion: Callable, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Train for a single step.

    This function uses data and a criterion to optimize model parameters. It returns
    the current loss in the training set.

    Args:
        graphdef: Graph representation of model.
        state: NNX state object including pytrees for all the
            model and optimizer graph nodes.
        criterion: Criterion to use for training.
        x: Input (features) array.
        y: Output (labels) array.

    Returns:
        Loss evaluated.
    """
    # merge at the beginning of the function
    model, optimizer, metrics = nnx.merge(graphdef, state)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, output), grads = grad_fn(model, criterion, x, y)
    snr = snr_fn(y, output)
    optimizer.update(model, grads, value=loss)  # In-place updates.
    metrics.update(loss=loss, snr=snr)  # In-place updates.

    state = nnx.state((model, optimizer, metrics))
    return loss, state
