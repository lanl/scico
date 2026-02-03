# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities to configure Optax optimizer."""

from functools import partial
from typing import Callable

import optax

from .typed_dict import ConfigDict


def build_optax_optimizer(config: ConfigDict):
    """Build optax optimizer to include in NNX optimizer.

    Args:
        opt_type: Type of optimizer to use. It can be: SGD, ADAM, ADAMW.

    Returns:
        Optax optimizer object.
    """
    # Build learning rate scheduler
    if "lr_schedule" in config:
        create_lr_schedule: Callable = config["lr_schedule"]
        learning_rate_fn = create_lr_schedule(config)
    else:
        learning_rate_fn = create_cnst_lr_schedule(config)

    # Build optimizer
    if config["opt_type"] == "SGD":
        # Stochastic Gradient Descent optimiser
        if "momentum" in config:
            opt_core = partial(optax.sgd, momentum=config["momentum"], nesterov=True)
        else:
            opt_core = optax.sgd
    elif config["opt_type"] == "ADAM":
        # Adam optimiser
        opt_core = optax.adam
    elif config["opt_type"] == "ADAMW":
        # Adam with weight decay regularization
        opt_core = optax.adamw
    else:
        raise NotImplementedError(
            f"Optimizer specified {config['opt_type']} has not been included."
        )

    # Build optax optimizer to be able to get lr later
    tx = optax.inject_hyperparams(opt_core)(learning_rate=learning_rate_fn)

    return tx


# Learning rate schedulers in optax


def create_cnst_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate to be a constant specified
    value.

    Args:
        config: Dictionary of configuration. The value to use corresponds
           to the `base_learning_rate` keyword.

    Returns:
        schedule: A function that maps step counts to values.
    """
    schedule = optax.constant_schedule(config["base_learning_rate"])
    return schedule


def create_exp_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate schedule to have an exponential decay.

    Args:
        config: Dictionary of configuration. The values to use correspond
            to `base_learning_rate`, `num_epochs`, `steps_per_epochs` and
            `lr_decay_rate`.

    Returns:
        schedule: A function that maps step counts to values.
    """
    decay_epochs = config["num_epochs"]
    schedule = optax.exponential_decay(
        config["base_learning_rate"], decay_epochs, config["lr_decay_rate"]
    )
    return schedule


def create_cosine_lr_schedule(config: ConfigDict) -> optax._src.base.Schedule:
    """Create learning rate to follow a pre-specified schedule.

    Create learning rate to follow a pre-specified schedule with warmup
    and cosine stages.

    Args:
        config: Dictionary of configuration. The parameters to use
            correspond to keywords: `base_learning_rate`, `num_epochs`,
            `warmup_epochs` and `steps_per_epoch`.

    Returns:
        schedule: A function that maps step counts to values.
    """
    # Warmup stage
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config["base_learning_rate"],
        transition_steps=config["warmup_epochs"],
    )
    # Cosine stage
    cosine_epochs = max(config["num_epochs"] - config["warmup_epochs"], 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config["base_learning_rate"],
        decay_steps=cosine_epochs,
    )

    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config["warmup_epochs"]],
    )

    return schedule


def create_reduce_on_plateau_schedule(
    config: ConfigDict,
    patience: int = 20,
    cooldown: int = 5,
    factor: float = 0.5,
    rtol: float = 1e-4,
    accumulation_size=5,
):
    """Create scheduler to reduce learning rate if loss does not improve.

    Args:
        config: Dictionary of configuration. The values to use correspond
            to `patience` and `lr_decay_rate`.
        patience: Number of epochs with no improvement after which learning rate will be reduced.
        cooldown: Number of epochs to wait before resuming normal operation after the learning rate reduction.
        factor: Factor by which to reduce the learning rate.
        rtol: Relative tolerance for measuring the new optimum.
        accumulation_size: Number of iterations to accumulate an average value

    Returns:
        schedule: A function that maps step counts to values.
    """
    if "patience" in config:
        patience = config["patience"]
    else:
        patience = patience

    if "lr_decay_rate" in config:
        factor = config["lr_decay_rate"]
    else:
        factor = factor

    schedule = optax.contrib.reduce_on_plateau(
        patience=patience,
        cooldown=cooldown,
        factor=factor,
        rtol=rtol,
        accumulation_size=accumulation_size,
    )

    return schedule
