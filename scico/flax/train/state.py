# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Configuration of Flax Train State."""

from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import optax

from flax.training import train_state
from scico.numpy import Array
from scico.typing import Shape

from .typed_dict import ConfigDict, ModelVarDict

ModuleDef = Any
KeyArray = Union[Array, jax.random.PRNGKeyArray]
PyTree = Any
ArrayTree = optax.Params


class TrainState(train_state.TrainState):
    """Definition of Flax train state.

    Definition of Flax train state including `batch_stats` for batch
    normalization.
    """

    batch_stats: Any


def initialize(key: KeyArray, model: ModuleDef, ishape: Shape) -> Tuple[PyTree, ...]:
    """Initialize Flax model.

    Args:
        key: A PRNGKey used as the random key.
        model: Flax model to train.
        ishape: Shape of signal (image) to process by `model`. Make sure
            that no batch dimension is included.

    Returns:
        Initial model parameters (including `batch_stats`).
    """
    input_shape = (1, ishape[0], ishape[1], model.channels)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


def create_basic_train_state(
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
    if variables0 is None:
        params, batch_stats = initialize(key, model, ishape)
    else:
        params = variables0["params"]
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

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state
