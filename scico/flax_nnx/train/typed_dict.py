# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of typed dictionaries for objects in training functionality."""

from __future__ import annotations  # for typing support for python 3.9 or earlier

import sys
from collections.abc import Callable
from typing import Any

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import NotRequired, TypedDict

from scico.numpy import Array


class DataSetDict(TypedDict):
    """Dictionary structure for training data sets.

    Definition of the dictionary structure
    expected for the training data sets.
    """

    #: Input (Num. samples x Height x Width x Channels).
    image: Array
    #: Output (Num. samples x Height x Width x Channels) or (Num. samples x Classes).
    label: Array


class ConfigDict(TypedDict):
    """Dictionary structure for training parameters.

    Definition of the dictionary structure expected for specifying
    training parameters. Only `opt_type` is required: every other entry
    is optional, and a default is substituted when it is absent. Note,
    though, that `seed` is required if no key is passed to
    :meth:`~.trainer.BasicFlaxNNXTrainer.train`, and that
    `base_learning_rate` is required unless `lr_schedule` is set to a
    schedule that does not use it.
    """

    #: Value to initialize seed for random generation.
    seed: NotRequired[float]
    #: Type of optimizer. Options: SGD, ADAM, ADAMW.
    opt_type: str
    #: Default clipping value for gradient norm in optimizer.
    opt_grad_max_norm: NotRequired[float]
    #: Momentum for SGD optimizer in case Nesterov is ``True``.
    momentum: NotRequired[float]
    #: Size of batch for training.
    batch_size: NotRequired[int]
    #: Number of epochs for training (an epoch is one whole pass through the training dataset).
    num_epochs: NotRequired[int]
    #: Starting learning rate for scheduling.
    base_learning_rate: NotRequired[float]
    #: Rate for decaying learning rate when scheduling is used.
    lr_decay_rate: NotRequired[float]
    #: Number of epochs if warmup scheduling is used.
    warmup_epochs: NotRequired[int]
    #: Number of epochs with no improvement after which learning rate will be reduced (if enabled).
    patience: NotRequired[int]
    #: Period of training epochs to print current train and test metrics.
    log_every_epochs: NotRequired[int]
    #: Period of training epochs to save model (if checkpointing is ``True``).
    checkpoint_every_epochs: NotRequired[int]
    #: Flag to indicate if evolution metrics are to be printed.
    log: NotRequired[bool]
    #: Path to directory for checkpointing model parameters.
    workdir: NotRequired[str]
    #: Flag to indicate if model parameters and optimizer state are to
    #: be stored while training.
    checkpointing: NotRequired[bool]
    #: Function to modify the learning rate while training (type optax schedule).
    lr_schedule: NotRequired[Callable]
    #: Criterion to optimize during training.
    criterion: NotRequired[Callable]
    #: Function to execute each training step.
    train_step_fn: NotRequired[Callable]
    #: Function to execute each evaluation step.
    eval_step_fn: NotRequired[Callable]
    #: List of post-processing functions to apply after a train step (if any).
    post_lst: NotRequired[list[Callable]]
    #: Size of buffer for shuffling data for large trainig sets
    shuffle_buffer_size: NotRequired[int]
    #: Function to create objects for logging training statistics.
    stats_obj: NotRequired[Any]  # type: ignore
