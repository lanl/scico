# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionality to evaluate Flax trained model.

Uses data parallel evaluation.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from flax import jax_utils
from scico.flax import create_input_iter
from scico.numpy import Array

from .checkpoints import checkpoint_restore
from .clu_utils import get_parameter_overview
from .learning_rate import create_cnst_lr_schedule
from .state import create_basic_train_state
from .typed_dict import ConfigDict, DataSetDict, ModelVarDict

ModuleDef = Any


def apply_fn(model: ModuleDef, variables: ModelVarDict, batch: DataSetDict) -> Array:
    """Apply current model.

    Assumes sharded batched data and replicated variables for
    distributed processing.

    This function is intended to be used via
    :meth:`~scico.flax.only_apply`, not directly.

    Args:
        model: Flax model to apply.
        variables: State of model parameters (replicated).
        batch: Sharded and batched training data.

    Returns:
        Output computed by given model.
    """
    output = model.apply(variables, batch["image"], train=False, mutable=False)
    return output


def only_apply(
    config: ConfigDict,
    model: ModuleDef,
    test_ds: DataSetDict,
    apply_fn: Callable = apply_fn,
    variables: Optional[ModelVarDict] = None,
) -> Tuple[Array, ModelVarDict]:
    """Execute model application loop.

    Args:
        config: Hyperparameter configuration.
        model: Flax model to apply.
        test_ds: Dictionary of testing data (includes images and
            labels).
        apply_fn: A hook for a function that applies current model.
            Default: :meth:`~scico.flax.train.apply.apply_fn`, i.e. use
            the standard apply function.
        variables: Model parameters to use for evaluation. Default:
            ``None`` (i.e. read from checkpoint).

    Returns:
        Output of model evaluated at the input provided in `test_ds`.

    Raises:
        RuntimeError: If no model variables and no checkpoint are
            specified.
    """
    if "workdir" in config:
        workdir: str = config["workdir"]
    else:
        workdir = "./"

    if "checkpointing" in config:
        checkpointing: bool = config["checkpointing"]
    else:
        checkpointing = False

    # Configure seed.
    key = jax.random.key(config["seed"])

    if variables is None:
        if checkpointing:  # pragma: no cover
            ishape = test_ds["image"].shape[1:3]
            lr_ = create_cnst_lr_schedule(config)
            empty_state = create_basic_train_state(key, config, model, ishape, lr_)
            state = checkpoint_restore(empty_state, workdir)
            if hasattr(state, "batch_stats"):
                variables = {
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                }  # type: ignore
                print(get_parameter_overview(variables["params"]))
                print(get_parameter_overview(variables["batch_stats"]))
            else:
                variables = {"params": state.params, "batch_stats": {}}
                print(get_parameter_overview(variables["params"]))
        else:
            raise RuntimeError("No variables or checkpoint provided.")

    # For distributed testing
    local_batch_size = config["batch_size"] // jax.process_count()
    size_device_prefetch = 2  # Set for GPU
    # Set data iterator
    eval_dt_iter = create_input_iter(
        key,  # eval: no permutation
        test_ds,
        local_batch_size,
        size_device_prefetch,
        model.dtype,
        train=False,
    )
    p_apply_step = jax.pmap(apply_fn, axis_name="batch", static_broadcasted_argnums=0)

    # Evaluate model with provided variables
    variables = jax_utils.replicate(variables)
    num_examples = test_ds["image"].shape[0]
    steps_ = num_examples // config["batch_size"]
    output_lst = []
    for _ in range(steps_):
        eval_batch = next(eval_dt_iter)
        output_batch = p_apply_step(model, variables, eval_batch)
        output_lst.append(output_batch.reshape((-1,) + output_batch.shape[-3:]))

    # Allow for completing the async run
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    # Extract one copy of variables
    variables = jax_utils.unreplicate(variables)
    # Convert to array
    output = jnp.array(output_lst)
    # Remove leading dimension
    output = output.reshape((-1,) + output.shape[-3:])

    return output, variables  # type: ignore
