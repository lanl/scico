# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Type definitions."""

from typing import Any, List, Tuple, Union

try:
    # available in python 3.10
    from types import EllipsisType  # type: ignore
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias  # type: ignore

    EllipsisType: TypeAlias = Any  # type: ignore


from jax import Array
from jax.typing import DTypeLike

PRNGKey: TypeAlias = Array
"""A key for jax random number generators (see :mod:`jax.random`)."""

# The jnp scalar types (jnp.float32 and the like) are instances of
# jax._src.numpy.scalar_types._ScalarMeta rather than types, so a union of
# them is not a valid type annotation: mypy rejects each component with
# "Variable "jax.numpy.float32" is not valid as a type" whenever the jax
# types are visible to it. The jax alias covers the same scalar types, as
# well as the strings, numpy dtypes, and numpy scalar types that the
# annotated parameters equally accept.
DType: TypeAlias = DTypeLike
"""An entity suitable for specifying the dtype of a numpy or jax array."""

Shape: TypeAlias = Tuple[int, ...]
"""A shape of a numpy or jax array."""

BlockShape: TypeAlias = Tuple[Tuple[int, ...], ...]
"""A shape of a :class:`.BlockArray`."""

Axes: TypeAlias = Union[int, Tuple[int, ...], List[int]]
"""Specification of one or more array axes."""

AxisIndex: TypeAlias = Union[slice, EllipsisType, int, None]
"""An entity suitable for indexing/slicing of a single array axis; either
a slice object, Ellipsis, int, or ``None`` (:data:`numpy.newaxis`, which
inserts a new axis of length one)."""

ArrayIndex: TypeAlias = Union[AxisIndex, Tuple[AxisIndex, ...]]
"""An entity suitable for indexing/slicing of multi-dimentional arrays."""
