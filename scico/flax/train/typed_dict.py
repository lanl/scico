# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
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
    expected for the training data sets."""

    image: Array  # input
    label: Array  # output


class ConfigDict(TypedDict):
    """Dictionary structure for training parmeters.

    Definition of the dictionary structure expected for specifying
    training parameters."""

    seed: float
    opt_type: str
    momentum: float
    batch_size: int
    num_epochs: int
    base_learning_rate: float
    lr_decay_rate: float
    warmup_epochs: int
    steps_per_eval: int
    log_every_steps: int
    steps_per_epoch: int
    steps_per_checkpoint: int
    log: bool
    workdir: str
    checkpointing: bool
    return_state: bool
    lr_schedule: Callable
    criterion: Callable
    create_train_state: Callable
    train_step_fn: Callable
    eval_step_fn: Callable
    metrics_fn: Callable
    post_lst: List[Callable]


class ModelVarDict(TypedDict):
    """Dictionary structure for Flax variables.

    Definition of the dictionary structure grouping all Flax model
    variables.
    """

    params: PyTree
    batch_stats: PyTree


class MetricsDict(TypedDict, total=False):
    """Dictionary structure for training metrics.

    Definition of the dictionary structure for metrics computed or
    updates made during training.
    """

    loss: float
    snr: float
    learning_rate: float
