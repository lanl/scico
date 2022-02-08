# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Type definitions."""

from typing import Any, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp

__author__ = """Luke Pfister <luke.pfister@gmail.com>"""


JaxArray = Union[jax.interpreters.xla.DeviceArray, jax.interpreters.pxla.ShardedDeviceArray]
"""A jax array."""

Array = Union[np.ndarray, JaxArray]
"""Either a numpy or jax array."""

PRNGKey = jnp.ndarray
"""A key for jax random number generators (see :mod:`jax.random`)."""

DType = Any
"""A numpy or jax dtype."""

Shape = Tuple[int, ...]
"""A shape of a numpy or jax array."""

BlockShape = Tuple[Tuple[int, ...], ...]
"""A shape of a :class:`.BlockArray`."""

Axes = Union[int, Tuple[int, ...]]
"""Specification of one or more array axes."""

AxisIndex = Union[slice, type(Ellipsis), int]
"""An entity suitable for indexing/slicing of a single array axis; either
a slice object, Ellipsis, or int."""

ArrayIndex = Union[AxisIndex, Tuple[AxisIndex]]
"""An entity suitable for indexing/slicing of multi-dimentional arrays."""
