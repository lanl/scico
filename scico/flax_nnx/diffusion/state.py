# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Configuration of diffusion model Flax Train State."""

from typing import Any, Optional, Union

import jax

import optax

from flax import nnx
from flax.training.train_state import TrainState
from scico.flax.train.typed_dict import ConfigDict, ModelVarDict
from scico.numpy import Array
from scico.typing import Shape

ModuleDef = Any
KeyArray = Union[Array, jax.Array]
PyTree = Any
ArrayTree = optax.Params


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
    model.train()
    graphdef, params = nnx.split(model, nnx.Param)
    # graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
    if variables0 is None:
        # if isinstance(aux, tuple):
        #    params, batch_stats = aux
        # else:
        #    params = aux
        if isinstance(params, tuple):
            params, batch_stats = params
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

    # Initialize training state
    if batch_stats is None:
        state = TrainState.create(
            apply_fn=graphdef.apply,
            params=params,
            tx=tx,
        )
    else:
        state = TrainState.create(
            apply_fn=graphdef.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
        )

    return state
