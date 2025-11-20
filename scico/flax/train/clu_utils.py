# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for displaying Flax models."""

# These utilities have been copied from the Common Loop Utils (CLU)
#   https://github.com/google/CommonLoopUtils/tree/main/clu
# and have been modified to remove TensorFlow dependencies
# CLU is licensed under the Apache License, Version 2.0, which may
# be obtained from
#   http://www.apache.org/licenses/LICENSE-2.0


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import jax

import flax

PyTree = Any
ParamsContainer = Union[Dict[str, np.ndarray], Mapping[str, Mapping[str, Any]]]


@dataclasses.dataclass
class ParamRow:
    """Definition of the structure of a row for printing parameters without stats."""

    name: str
    shape: Tuple[int]
    size: int


@dataclasses.dataclass
class ParamRowWithStats(ParamRow):
    """Definition of the structure of a row for printing parameters with stats."""

    mean: float
    std: float


def flatten_dict(
    input_dict: Dict[str, Any], prefix: str = "", delimiter: str = "/"
) -> Dict[str, Any]:
    """Flatten keys of a nested dictionary.

    Args:
        input_dict: Nested dictionary.
        prefix: Prefix of already flatten. Default: empty string.
        delimiter: Delimiter for displaying. Default: ``/``.

    Returns:
        A dictionary with the keys flattened.
    """
    output_dict = {}
    for key, value in input_dict.items():
        nested_key = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, (dict, flax.core.FrozenDict)):
            output_dict.update(flatten_dict(value, prefix=nested_key, delimiter=delimiter))
        else:
            output_dict[nested_key] = value
    return output_dict


def count_parameters(params: PyTree) -> int:
    """Return count of variables for the parameter dictionary.

    Args:
        params: Flax model parameters.

    Returns:
        The number of parameters in the model.
    """
    flat_params = flatten_dict(params)
    return sum(np.prod(v.shape) for v in flat_params.values())  # type: ignore


def get_parameter_rows(
    params: ParamsContainer,
    *,
    include_stats: bool = False,
) -> List[Union[ParamRow, ParamRowWithStats]]:
    """Return information about parameters as a list of dictionaries.

    Args:
        params: Dictionary with parameters as NumPy arrays. The dictionary
            can be nested.
        include_stats: If ``True`` add columns with mean and std for each
            variable. Note that this can be considerably more compute
            intensive and cause a lot of memory to be transferred to the
            host.

    Returns:
        A list of `ParamRow`, or `ParamRowWithStats`, depending on the
        passed value of `include_stats`.
    """
    assert isinstance(params, (dict, flax.core.FrozenDict))
    if params:
        params = flatten_dict(params)
        names, values = map(list, tuple(zip(*sorted(params.items()))))
    else:
        names, values = [], []

    def make_row(name, value):
        if include_stats:
            return ParamRowWithStats(
                name=name,
                shape=value.shape,
                size=int(np.prod(value.shape)),
                mean=float(value.mean()),
                std=float(value.std()),
            )
        else:
            return ParamRow(name=name, shape=value.shape, size=int(np.prod(value.shape)))

    return [make_row(name, value) for name, value in zip(names, values)]


def _default_table_value_formatter(value):
    """Format ints with "," between thousands, and floats to 3 digits."""
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, int):
        return "{:,}".format(value)
    elif isinstance(value, float):
        return "{:.3}".format(value)
    else:
        return str(value)


def make_table(
    rows: List[Any],
    *,
    column_names: Optional[Sequence[str]] = None,
    value_formatter: Callable[[Any], str] = _default_table_value_formatter,
    max_lines: Optional[int] = None,
) -> str:
    """Render list of rows to a table.

    Args:
        rows: List of dataclass instances of a single type
            (e.g. `ParamRow`).
        column_names: List of columns that that should be included in the
            output. If not provided, then the columns are taken from keys
            of the first row.
        value_formatter: Callable used to format cell values.
        max_lines: Don't render a table longer than this.

    Returns:
        A string representation of a table as in the example below.

        ::

          +---------+---------+
          | Col1    | Col2    |
          +---------+---------+
          | value11 | value12 |
          | value21 | value22 |
          +---------+---------+
    """
    if any(not dataclasses.is_dataclass(row) for row in rows):
        raise ValueError("Expected argument 'rows' to be list of dataclasses")
    if len(set(map(type, rows))) > 1:
        raise ValueError("Expected elements of argument 'rows' be of same type.")

    class Column:
        """Definition of a column for printing parameters."""

        def __init__(self, name, values):
            self.name = name.capitalize()
            self.values = values
            self.width = max(len(v) for v in values + [name])

    if column_names is None:
        if not rows:
            return "(empty table)"
        column_names = [field.name for field in dataclasses.fields(rows[0])]

    columns = [
        Column(name, [value_formatter(getattr(row, name)) for row in rows]) for name in column_names
    ]

    var_line_format = "|" + "".join(f" {{: <{c.width}s}} |" for c in columns)
    sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")
    header = var_line_format.replace(">", "<").format(*[c.name for c in columns])
    separator = sep_line_format.format(*["" for c in columns])

    lines = [separator, header, separator]
    for i in range(len(rows)):
        if max_lines and len(lines) >= max_lines - 3:
            lines.append("[...]")
            break
        lines.append(var_line_format.format(*[c.values[i] for c in columns]))
    lines.append(separator)

    return "\n".join(lines)


def get_parameter_overview(
    params: ParamsContainer, *, include_stats: bool = True, max_lines: Optional[int] = None
) -> str:
    """Return string with variables names, their shapes, count.

    Args:
        params: Dictionary with parameters as NumPy arrays. The dictionary
            can be nested.
        include_stats: If ``True``, add columns with mean and std for each
            variable.
        max_lines: If not ``None``, the maximum number of variables to
            include.

    Returns:
        A string with a table as in the example below.

        ::

          +----------------+---------------+------------+
          | Name           | Shape         | Size       |
          +----------------+---------------+------------+
          | FC_1/weights:0 | (63612, 1024) | 65,138,688 |
          | FC_1/biases:0  |       (1024,) |      1,024 |
          | FC_2/weights:0 |    (1024, 32) |     32,768 |
          | FC_2/biases:0  |         (32,) |         32 |
          +----------------+---------------+------------+
          Total weights: 65,172,512
    """
    if isinstance(params, (dict, flax.core.FrozenDict)):
        params = jax.tree_util.tree_map(np.asarray, params)
    rows = get_parameter_rows(params, include_stats=include_stats)
    total_weights = count_parameters(params)
    RowType = ParamRowWithStats if include_stats else ParamRow
    # Pass in `column_names` to enable rendering empty tables.
    column_names = [field.name for field in dataclasses.fields(RowType)]
    table = make_table(rows, max_lines=max_lines, column_names=column_names)
    return table + f"\nTotal weights: {total_weights:,}"
