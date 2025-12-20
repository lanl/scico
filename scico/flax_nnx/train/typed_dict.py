# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of typed dictionaries for objects in training functionality."""

import sys
from typing import Any, Callable, List

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from jax.typing import ArrayLike



class DataSetDict(TypedDict):
    """Dictionary structure for training data sets.

    Definition of the dictionary structure
    expected for the training data sets.
    """

    #: Input (Num. samples x Height x Width x Channels).
    image: ArrayLike
    #: Output (Num. samples x Height x Width x Channels) or (Num. samples x Classes).
    label: ArrayLike


class ConfigDict(TypedDict):
    """Dictionary structure for training parameters.

    Definition of the dictionary structure expected for specifying
    training parameters.
    """

    #: Value to initialize seed for random generation.
    seed: float
    #: Type of optimizer. Options: SGD, ADAM, ADAMW.
    opt_type: str
    #: Momentum for SGD optimizer in case Nesterov is ``True``.
    momentum: float
    #: Size of batch for training.
    batch_size: int
    #: Number of epochs for training (an epoch is one whole pass through the training dataset).
    num_epochs: int
    #: Starting learning rate for scheduling.
    base_learning_rate: float
    #: Rate for decaying learning rate when scheduling is used.
    lr_decay_rate: float
    #: Number of epochs if warmup scheduling is used.
    warmup_epochs: int
    #: Period of training epochs to print current train and test metrics.
    log_every_epochs: int
    #: Period of training epochs to save model (if checkpointing is ``True``).
    checkpoint_every_epochs: int
    #: Flag to indicate if evolution metrics are to be printed.
    log: bool
    #: Path to directory for checkpointing model parameters.
    workdir: str
    #: Flag to indicate if model parameters and optimizer state are to
    #: be stored while training.
    checkpointing: bool
    #: Function to modify the learning rate while training (type optax schedule).
    lr_schedule: Callable
    #: Criterion to optimize during training.
    criterion: Callable
    #: Function to create and initialize trainig state. Should include initialization
    #: of optimizer and of batch_stats (if applicable).
    create_train_state: Callable
    #: Function to execute each training step.
    train_step_fn: Callable
    #: Function to execute each evaluation step.
    eval_step_fn: Callable
    #: Function to track metrics during training.
    metrics_fn: Callable
    #: List of post-processing functions to apply after a train step (if any).
    post_lst: List[Callable]

