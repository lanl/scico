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


import jax.numpy as jnp
from jax import Array

PRNGKey: TypeAlias = Array
"""A key for jax random number generators (see :mod:`jax.random`)."""

DType: TypeAlias = Union[
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.int64,
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.uint64,
    jnp.float16,
    jnp.float32,
    jnp.float64,
    jnp.complex64,
    jnp.complex128,
    bool,
]
"""A jax dtype."""

Shape: TypeAlias = Tuple[int, ...]
"""A shape of a numpy or jax array."""

BlockShape: TypeAlias = Tuple[Tuple[int, ...], ...]
"""A shape of a :class:`.BlockArray`."""

Axes: TypeAlias = Union[int, Tuple[int, ...], List[int]]
"""Specification of one or more array axes."""

AxisIndex: TypeAlias = Union[slice, EllipsisType, int]
"""An entity suitable for indexing/slicing of a single array axis; either
a slice object, Ellipsis, or int."""

ArrayIndex: TypeAlias = Union[AxisIndex, Tuple[AxisIndex, ...]]
"""An entity suitable for indexing/slicing of multi-dimentional arrays."""
