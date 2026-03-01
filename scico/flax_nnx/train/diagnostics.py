# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for computing and displaying performance metrics during training.

Assumes sharded batched data.
"""

from typing import Callable, Dict, Tuple, Union

from scico.diagnostics import IterationStats


class ArgumentStruct:
    """Class that converts a dictionary into an object with named entries.

    Class that converts a python dictionary into an object with named
    entries given by the dictionary keys. After the object instantiation
    both modes of access (dictionary or object entries) can be used.
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)


def create_itstat(
    itstat_fields: IterationStats, itstat_attrib: Callable
) -> Tuple[IterationStats, Callable]:
    """Function to create objects for logging training statistics.

    These objects are created and used if logging is enabled during training.

    Args:
        itstat_fields: Dictionary with titles and printing format for selected statistics.
        itstat_attrib: List of names of registered statistics.

    Returns:
        An object :class:`~.diagnostics.IterationStats` to log and store iteration
        statistics and an object :class:`~.diagnostics.IterationStats` to printing stats
        to command line and storing them for further analysis.
    """
    # dynamically create itstat_func; see https://stackoverflow.com/questions/24733831
    itstat_return = "return(" + ", ".join(["obj." + attr for attr in itstat_attrib]) + ")"
    scope: Dict[str, Callable] = {}
    exec("def itstat_func(obj): " + itstat_return, scope)
    default_itstat_options: Dict[str, Union[dict, Callable, bool]] = {
        "fields": itstat_fields,
        "itstat_func": scope["itstat_func"],
        "display": True,
    }
    itstat_insert_func: Callable = default_itstat_options.pop("itstat_func")  # type: ignore
    itstat_object = IterationStats(**default_itstat_options)  # type: ignore

    return itstat_object, itstat_insert_func


def stats_obj() -> Tuple[IterationStats, Callable]:
    """Functionality to log and store a specific subset of iteration statistics.

    This function specifies some statistics to register while training
    together with their titles and printing format. The statistics
    collected are: epoch, time, learning rate, loss and snr in training
    and loss and snr in evaluation.

    Returns:
        An object :class:`~.diagnostics.IterationStats` to log and store iteration
        statistics if logging is enabled during training and an object
        :class:`~.diagnostics.IterationStats` to printing stats to command line and
        storing them for further analysis.
    """
    # epoch, time learning rate loss and snr (train and
    # eval) fields
    itstat_fields = {
        "Epoch": "%d",
        "Time": "%8.2e",
        "Train_LR": "%.6f",
        "Train_Loss": "%.6f",
        "Train_SNR": "%.2f",
        "Eval_Loss": "%.6f",
        "Eval_SNR": "%.2f",
    }
    itstat_attrib = [
        "epoch",
        "time",
        "train_learning_rate",
        "train_loss",
        "train_snr",
        "test_loss",
        "test_snr",
    ]

    return create_itstat(itstat_fields, itstat_attrib)
