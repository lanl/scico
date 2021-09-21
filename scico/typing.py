# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Type definitions."""

from typing import Any, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp

__author__ = """Luke Pfister <pfister@lanl.gov>"""


JaxArray = Union[jax.interpreters.xla.DeviceArray, jax.interpreters.pxla.ShardedDeviceArray]
"""A jax array"""

Array = Union[np.ndarray, JaxArray]
"""Either a numpy or jax array"""

PRNGKey = jnp.ndarray
"""A key for jax random number generators (see :mod:`jax.random`)"""

DType = Any  # TODO: can we do better than this?  Maybe with the new numpy typing?
"""A numpy or jax dtype"""

Shape = Tuple[int, ...]  # Shape of an array
"""A shape of a numpy or jax array"""

BlockShape = Tuple[Tuple[int, ...], ...]  # Shape of a BlockArray
"""A shape of a :class:`.BlockArray`"""

Axes = Union[int, Tuple[int, ...]]  # one or more axes
