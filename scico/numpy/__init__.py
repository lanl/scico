# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r""":class:`.BlockArray` and compatible functions.

This module consists of :class:`.BlockArray` and functions that support
both instances of this class and jax arrays. This includes all the
functions from :mod:`jax.numpy` and :mod:`numpy.testing`, where many have
been extended to automatically map over block array blocks as described
in :ref:`numpy_functions_blockarray`. Also included are additional
functions unique to SCICO in :mod:`.util`.
"""

import sys
from functools import partial, wraps
from types import ModuleType
from typing import Union

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from . import _wrappers, util
from ._blockarray import BlockArray, TransparentTuple
from ._wrapped_function_lists import (
    CREATION_ROUTINES,
    MATHEMATICAL_FUNCTIONS,
    REDUCTIONS,
    TESTING_FUNCTIONS,
)

__all__ = ["fft", "linalg", "testing", "util", "BlockArray", "blockarray"]


def blockarray(a):
    """Construct a :class:`.BlockArray` from a list or tuple of existing array-like."""
    return BlockArray(a)


# copy most of jnp without wrapping
_wrappers.add_attributes(
    to_dict=vars(), from_dict=jax.numpy.__dict__, modules_to_recurse=["linalg", "fft"]
)


def ravel(ba: Union[Array | TransparentTuple]) -> Array:
    """Completely flatten a :class:`BlockArray` into a single ``Array``.

    When called on an ``Array``, flattens the array.

    Args:
        ba: The :class:`BlockArray` to flatten.

    Returns:
        `ba` flattened into a single ``Array.``
    """
    if isinstance(ba, TransparentTuple):
        return jnp.concatenate(ba.ravel())

    return ba.ravel()


def stack(ba, *args, **kwargs):
    """Collapse a block array to :class:`jax.Array`.

    Collapse a block array to :class:`jax.Array` by stacking
    the blocks on axis `axis`.

    Args:
        axis: Index of new axis on which blocks are to be stacked.

    Returns:
        A :class:`jax.Array` obtained by stacking.

    Raises:
        ValueError: When called on a :class:`.BlockArray` that is not
            stackable.
    """
    if not isinstance(ba, TransparentTuple):
        return jnp.stack(ba, *args, **kwargs)

    if util.is_collapsible(ba.shape):
        return jnp.stack(ba, *args, **kwargs)
    else:
        raise ValueError(f"BlockArray of shape {ba.shape} cannot be collapsed to an Array.")


# wrap jnp funcs
_wrappers.wrap_recursively(
    vars(),
    CREATION_ROUTINES,
    partial(
        _wrappers.map_func_over_args,
        map_if_nested_args=["shape"],
        map_if_list_args=["device"],
    ),
)
_wrappers.wrap_recursively(vars(), MATHEMATICAL_FUNCTIONS, _wrappers.map_func_over_args)

_wrappers.wrap_recursively(vars(), REDUCTIONS, _wrappers.add_full_reduction)


@wraps(jnp.linalg.norm)
def norm(x, ord=None, axis=np._NoValue, *args, **kwargs):
    """Wrapper or jnp.linalg.norm"""
    if not isinstance(x, TransparentTuple):
        if axis is np._NoValue:
            # jnp funcs can't accept axis=np._NoValue
            return jnp.linalg.norm(x, ord, *args, **kwargs)
        else:
            return jnp.linalg.norm(x, ord, *args, **kwargs)

    if axis is np._NoValue:
        return jnp.linalg.norm(ravel(x), ord, *args, **kwargs)

    return TransparentTuple(norm(x_i, ord, axis, *args, **kwargs) for x_i in x)


linalg.norm = norm
