# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Learning rate schedulers."""

import optax

from .typed_dict import ConfigDict


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
    decay_steps = config["num_epochs"] * config["steps_per_epoch"]
    schedule = optax.exponential_decay(
        config["base_learning_rate"], decay_steps, config["lr_decay_rate"]
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
        transition_steps=config["warmup_epochs"] * config["steps_per_epoch"],
    )
    # Cosine stage
    cosine_epochs = max(config["num_epochs"] - config["warmup_epochs"], 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config["base_learning_rate"],
        decay_steps=cosine_epochs * config["steps_per_epoch"],
    )

    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config["warmup_epochs"] * config["steps_per_epoch"]],
    )

    return schedule
