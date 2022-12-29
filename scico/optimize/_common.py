# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions common to multiple optimizer modules."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from scico.diagnostics import IterationStats
from scico.util import Timer


def itstat_func_and_object(
    itstat_fields: dict, itstat_attrib: List, itstat_options: Optional[dict] = None
) -> Tuple[Callable, IterationStats]:
    """Iteration statistics initialization.

    Iteration statistics initialization steps common to all optimizer
    classes.

    Args:
        itstat_fields: A dictionary associating field names with format
              strings for displaying the corresponding values.
        itstat_attrib: A list of expressions corresponding of optimizer
              class attributes to be evaluated when computing iteration
              statistics.
        itstat_options: A dict of named parameters to be passed to
              the :class:`.diagnostics.IterationStats` initializer. The
              dict may also include an additional key "itstat_func"
              with the corresponding value being a function with two
              parameters, an integer and an optimizer object,
              responsible for constructing a tuple ready for insertion
              into the :class:`.diagnostics.IterationStats` object. If
              ``None``, default values are used for the dict entries,
              otherwise the default dict is updated with the dict
              specified by this parameter.

    Returns:
        A tuple consisting of the statistics insertion function and the
        :class:`.diagnostics.IterationStats` object.
    """
    # dynamically create itstat_func; see https://stackoverflow.com/questions/24733831
    itstat_return = "return(" + ", ".join(["obj." + attr for attr in itstat_attrib]) + ")"
    scope: Dict[str, Callable] = {}
    exec("def itstat_func(obj): " + itstat_return, scope)

    # determine itstat options and initialize IterationStats object
    default_itstat_options: Dict[str, Union[dict, Callable, bool]] = {
        "fields": itstat_fields,
        "itstat_func": scope["itstat_func"],
        "display": False,
    }
    if itstat_options:
        default_itstat_options.update(itstat_options)

    itstat_insert_func: Callable = default_itstat_options.pop("itstat_func", None)  # type: ignore
    itstat_object = IterationStats(**default_itstat_options)  # type: ignore

    return itstat_insert_func, itstat_object


class Optimizer:
    """Base class for optimizer classes."""

    def __init__(self, itstat_fields: dict, itstat_attrib: list, **kwargs):
        itstat_options = kwargs.pop("itstat_options", None)
        self.maxiter: int = kwargs.pop("maxiter", 100)
        self.itnum: int = 0
        self.timer: Timer = Timer()
        self._itstat_init(itstat_fields, itstat_attrib, itstat_options=itstat_options)

    def _itstat_init(
        self, itstat_fields: dict, itstat_attrib: list, itstat_options: Optional[dict] = None
    ):
        """Initialize iteration statistics mechanism.

        Args:
           itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and a :class:`PDHG` object,
                responsible for constructing a tuple ready for insertion
                into the :class:`.diagnostics.IterationStats` object. If
                ``None``, default values are used for the dict entries,
                otherwise the default dict is updated with the dict
                specified by this parameter.
        """
        # iteration number and time fields
        _itstat_fields = {
            "Iter": "%d",
            "Time": "%8.2e",
        }
        _itstat_attrib = ["itnum", "timer.elapsed()"]
        # objective function can be evaluated if 'g' function can be evaluated
        if self.g.has_eval:
            _itstat_fields.update({"Objective": "%9.3e"})
            _itstat_attrib.append("objective()")
        # primal and dual residual fields
        # itstat_fields.update({"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"})
        # itstat_attrib.extend(["norm_primal_residual()", "norm_dual_residual()"])
        _itstat_fields.update(itstat_fields)
        _itstat_attrib.extend(itstat_attrib)

        self.itstat_insert_func, self.itstat_object = itstat_func_and_object(
            _itstat_fields, _itstat_attrib, itstat_options
        )

    def solve(
        self,
        callback: Optional[Callable[[Optimizer], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        r"""Initialize and run the optimization algorithm.

        Initialize and run the opimization algorithm for a total of
        `self.maxiter` iterations.

        Args:
            callback: An optional callback function, taking an a single
              argument of type :class:`Optimizer`, that is called
              at the end of every iteration.

        Returns:
            Computed solution.
        """
        self.timer.start()
        for self.itnum in range(self.itnum, self.itnum + self.maxiter):
            self.step()
            self.itstat_object.insert(self.itstat_insert_func(self))
            if callback:
                self.timer.stop()
                callback(self)
                self.timer.start()
        self.timer.stop()
        self.itnum += 1
        self.itstat_object.end()
        return self.x
