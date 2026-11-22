# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for computing and displaying performance metrics during training.

Assumes sharded batched data.
"""

from __future__ import annotations  # for typing support for python 3.9 or earlier

from collections.abc import Callable

from scico.diagnostics import IterationStats
from scico.flax_nnx.train.diagnostics import create_itstat


def stats_obj() -> tuple[IterationStats, Callable]:
    """Log and store a specific subset of diffusion model training statistics.

    This function specifies some statistics to register while training
    together with their titles and printing format. The statistics
    collected are: epoch, time, learning rate and loss, both in training
    and in evaluation. These statistics are displayed if logging is enabled
    during training.

    Returns:
        An object :class:`~.diagnostics.IterationStats` that logs and stores
        the iteration statistics, and a function that extracts the
        registered statistics from an object, for insertion into it.
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
