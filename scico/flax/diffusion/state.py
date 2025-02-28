# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Configuration of diffusion model Flax Train State."""

from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import optax

from flax.training.train_state import TrainState
from scico.flax.train.typed_dict import ConfigDict, ModelVarDict
from scico.numpy import Array
from scico.typing import Shape

ModuleDef = Any
KeyArray = Union[Array, jax.Array]
PyTree = Any
ArrayTree = optax.Params


def initialize(key: KeyArray, model: ModuleDef, ishape: Shape) -> Tuple[PyTree, ...]:
    """Initialize Flax model.

    Args:
        key: A PRNGKey used as the random key.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Make sure
            that no batch dimension is included.

    Returns:
        Initial model parameters (including `batch_stats` if applicable).
    """
    if hasattr(model, "channels"):
        input_shape = (1, ishape[0], ishape[1], model.channels)
    else:
        input_shape = (1,) + ishape
    key, model_key = jax.random.split(key)
    key, call_key = jax.random.split(key)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({"params": model_key}, jnp.ones(input_shape, model.dtype), call_key)
    return variables["params"]


def initialize_with_time_embedding(
    key: KeyArray, model: ModuleDef, ishape: Shape
) -> Tuple[PyTree, ...]:
    """Initialize Flax model conditioned on time embeddings.

    Args:
        key: A PRNGKey used as the random key.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Make sure
            that no batch dimension is included.

    Returns:
        Initial model parameters.
    """
    if hasattr(model, "channels"):
        input_shape = (1, ishape[0], ishape[1], model.channels)
    else:
        input_shape = (1,) + ishape

    key, model_key = jax.random.split(key)

    @jax.jit
    def init(*args):
        return model.init(*args)

    fakex = jnp.ones(input_shape, model.dtype)  # Expected input shape
    faket = jnp.ones((1, 1))  # Expected time input
    variables = init({"params": model_key}, fakex, faket)
    return variables["params"]


def create_train_state(
    key: KeyArray,
    config: ConfigDict,
    model: ModuleDef,
    ishape: Shape,
    learning_rate_fn: optax._src.base.Schedule,
    variables0: Optional[ModelVarDict] = None,
) -> TrainState:
    """Create Flax basic train state and initialize.

    Args:
        key: A PRNGKey used as the random key.
        config: Dictionary of configuration. The values to use correspond
            to keywords: `opt_type` and `momentum`.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Ensure
            that no batch dimension is included.
        variables0: Optional initial state of model parameters. If not
            provided a random initialization is performed. Default:
            ``None``.
        learning_rate_fn: A function that maps step counts to values.

    Returns:
        state: Flax train state which includes the model apply function,
           the model parameters and an Optax optimizer.
    """
    batch_stats = None
    if variables0 is None:
        if model.time_embed:  # Model uses time embedding imput
            aux = initialize_with_time_embedding(key, model, ishape)
        else:
            aux = initialize(key, model, ishape)
        if isinstance(aux, tuple):
            params, batch_stats = aux
        else:
            params = aux
    else:
        params = variables0["params"]
        if "batch_stats" in variables0:
            batch_stats = variables0["batch_stats"]

    if config["opt_type"] == "SGD":
        # Stochastic Gradient Descent optimiser
        if "momentum" in config:
            tx = optax.sgd(
                learning_rate=learning_rate_fn, momentum=config["momentum"], nesterov=True
            )
        else:
            tx = optax.sgd(learning_rate=learning_rate_fn)
    elif config["opt_type"] == "ADAM":
        # Adam optimiser
        tx = optax.adam(
            learning_rate=learning_rate_fn,
        )
    elif config["opt_type"] == "ADAMW":
        # Adam with weight decay regularization
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
        )
    else:
        raise NotImplementedError(
            f"Optimizer specified {config['opt_type']} has not been included in SCICO."
        )

    if batch_stats is None:
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )
    else:
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
        )

    return state
