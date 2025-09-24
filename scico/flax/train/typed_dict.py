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

from scico.numpy import Array

PyTree = Any


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
    #: Period of training steps to evaluate over test set.
    steps_per_eval: int
    #: Period of training steps to print current train and test metrics.
    log_every_steps: int
    #: Training steps to be executed per epoch (depends on batch size).
    steps_per_epoch: int
    #: Period of training steps to save model (if checkpointing is ``True``).
    steps_per_checkpoint: int
    #: Flag to indicate if evolution metrics are to be printed.
    log: bool
    #: Path to directory for checkpointing model parameters.
    workdir: str
    #: Flag to indicate if model parameters and optimizer state are to
    #: be stored while training.
    checkpointing: bool
    #: Flag to indicate if state (params and batch_stats) are to
    #: be returned at the end of training.
    return_state: bool
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


class ModelVarDict(TypedDict):
    """Dictionary structure for Flax variables.

    Definition of the dictionary structure grouping all Flax model
    variables.
    """

    #: Model weights and biases.
    params: PyTree
    #: Batch statistics (e.g. normalization parameters that depend on training data).
    batch_stats: PyTree


class MetricsDict(TypedDict, total=False):
    """Dictionary structure for training metrics.

    Definition of the dictionary structure for metrics computed or
    updates made during training.
    """

    loss: float  #: Evaluation of criterion being optimized.
    snr: float  #: Evaluation of signal to noise ratio.
    learning_rate: float  #: Current learning rate.
