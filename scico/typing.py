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

DType = Any  # TODO: can we do better than this? Maybe with the new numpy typing?
"""A numpy or jax dtype."""

Shape = Tuple[int, ...]  # shape of an array
"""A shape of a numpy or jax array."""

BlockShape = Tuple[Tuple[int, ...], ...]  # shape of a BlockArray
"""A shape of a :class:`.BlockArray`."""

Axes = Union[int, Tuple[int, ...]]  # one or more axes
"""Specification of one or more array axes."""

Slice = Union[slice, type(Ellipsis), int]
"""An entity suitable for slicing; either a slice object, Ellipsis, or int."""

MultiSlice = Union[Slice, Tuple[Slice]]
"""An entity suitable for slicing of multi-dimentional arrays."""
