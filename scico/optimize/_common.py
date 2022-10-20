# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions common to multiple optimizer modules."""


from typing import Callable, List, Optional, Tuple, Union

from scico.diagnostics import IterationStats


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
    scope: dict[str, Callable] = {}
    exec("def itstat_func(obj): " + itstat_return, scope)

    # determine itstat options and initialize IterationStats object
    default_itstat_options: dict[str, Union[dict, Callable, bool]] = {
        "fields": itstat_fields,
        "itstat_func": scope["itstat_func"],
        "display": False,
    }
    if itstat_options:
        default_itstat_options.update(itstat_options)

    itstat_insert_func: Callable = default_itstat_options.pop("itstat_func", None)  # type: ignore
    itstat_object = IterationStats(**default_itstat_options)  # type: ignore

    return itstat_insert_func, itstat_object
