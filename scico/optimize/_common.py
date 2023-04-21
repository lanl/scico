# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions common to multiple optimizer modules."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scico.diagnostics import IterationStats
from scico.numpy import Array, BlockArray
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
              parameters, an integer and a :class:`Optimizer` object,
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
    """Base class for optimizer classes.

    Attributes:
        itnum (int): Optimizer iteration counter.
        maxiter (int): Maximum number of optimizer outer-loop iterations.
        timer (:class:`.Timer`): Iteration timer.
    """

    def __init__(self, **kwargs: Any):
        """Initialize common attributes of :class:`Optimizer` objects.

        Args:
            **kwargs: Optional parameter dict. Valid keys are:

                maxiter:
                  Maximum iterations on call to :meth:`solve`.

                nanstop:
                  If ``True``, stop iterations if a ``NaN`` or ``Inf``
                  value is encountered in a solver working variable.

                itstat_options:
                  A dict of named parameters to be passed to
                  the :class:`.diagnostics.IterationStats` initializer.
                  The dict may also include an additional key
                  "itstat_func" with the corresponding value being a
                  function with two parameters, an integer and an
                  :class:`Optimizer` object, responsible for constructing
                  a tuple ready for insertion into the
                  :class:`.diagnostics.IterationStats` object. If
                  ``None``, default values are used for the dict entries,
                  otherwise the default dict is updated with the dict
                  specified by this parameter.
        """
        self.maxiter: int = kwargs.pop("maxiter", 100)
        self.nanstop: bool = kwargs.pop("nanstop", False)
        itstat_options = kwargs.pop("itstat_options", None)

        if kwargs:
            raise TypeError(f"Unrecognized keyword argument(s) {', '.join([k for k in kwargs])}")

        self.itnum: int = 0
        self.timer: Timer = Timer()

        itstat_fields, itstat_attrib = self._itstat_default_fields()
        itstat_extra_fields, itstat_extra_attrib = self._itstat_extra_fields()
        itstat_fields.update(itstat_extra_fields)
        itstat_attrib.extend(itstat_extra_attrib)

        self.itstat_insert_func, self.itstat_object = itstat_func_and_object(
            itstat_fields, itstat_attrib, itstat_options
        )

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        raise NotImplementedError(
            "NaN check requested but _working_vars_finite not implemented." ""
        )

    def _itstat_default_fields(self) -> Tuple[Dict[str, str], List[str]]:
        """Define iterations stats default fields.

        Return a dict mapping field names to format strings, and a list
        of strings containing the names of attributes or methods to call
        in order to determine the value for each field.
        """
        # iteration number and time fields
        itstat_fields = {
            "Iter": "%d",
            "Time": "%8.2e",
        }
        itstat_attrib = ["itnum", "timer.elapsed()"]
        # objective function can be evaluated if 'g' function can be evaluated
        if self._objective_evaluatable():
            itstat_fields.update({"Objective": "%9.3e"})
            itstat_attrib.append("objective()")

        return itstat_fields, itstat_attrib

    def _objective_evaluatable(self) -> bool:
        """Determine whether the objective function can be evaluated.

        Determine whether the objective function for a :class:`Optimizer`
        object can be evaluated.
        """
        return False

    def _itstat_extra_fields(self) -> Tuple[Dict[str, str], List[str]]:
        """Define additional iterations stats fields.

        Define iterations stats fields that are not common to all
        :class:`Optimizer` classes. Return a dict mapping field names to
        format strings, and a list of strings containing the names of
        attributes or methods to call in order to determine the value for
        each field.
        """
        return {}, []

    def minimizer(self) -> Union[Array, BlockArray]:
        """Return the current estimate of the functional mimimizer."""

    def step(self):
        """Perform a single optimizer step."""

    def solve(
        self,
        callback: Optional[Callable[[Optimizer], None]] = None,
    ) -> Union[Array, BlockArray]:
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
            if self.nanstop and not self._working_vars_finite():
                raise ValueError(
                    f"NaN or Inf value encountered in working variable in iteration {self.itnum}."
                    ""
                )
            self.itstat_object.insert(self.itstat_insert_func(self))
            if callback:
                self.timer.stop()
                callback(self)
                self.timer.start()
        self.timer.stop()
        self.itnum += 1
        self.itstat_object.end()
        return self.minimizer()
