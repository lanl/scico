# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for computing and displaying performance metrics during training.

Assumes sharded batched data.
"""

from typing import Callable, Tuple

from scico.diagnostics import IterationStats
from scico.flax_nnx.train.diagnostics import create_itstat


def stats_obj() -> Tuple[IterationStats, Callable]:
    """Functionality to log and store a specific subset of diffusion
    models training statistics.

    This function specifies some statistics to register while training
    together with their titles and printing format. The statistics
    collected are: epoch, time, learning rate and loss, both in training
    and in evaluation. These statistics are displayed if logging is enabled
    during training.

    Returns:
        An object :class:`~.diagnostics.IterationStats` to log and store
        iteration statistics and an object
        :class:`~.diagnostics.IterationStats` to printing stats to command
        line and storing them for further analysis.
    """
    # epoch time learning_rate noise_std loss (train and eval) fields
    itstat_fields = {
        "Epoch": "%d",
        "Time": "%8.2e",
        "Train_LR": "%.6f",
        "Train_Loss": "%.6f",
        "Eval_Loss": "%.6f",
    }
    itstat_attrib = [
        "epoch",
        "time",
        "train_learning_rate",
        "train_loss",
        "test_loss",
    ]

    return create_itstat(itstat_fields, itstat_attrib)
