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
from functools import partial
from typing import Union

import numpy as np

import jax
import jax.numpy
from jax import Array

from . import _wrappers, fft, linalg, testing, util
from ._blockarray import BlockArray, TransparentTuple, blockarray
from ._wrapped_function_lists import (
    BINARY_OPS,
    REDUCTIONS,
    creation_routines,
    mathematical_functions,
    testing_functions,
)

__all__ = ["fft", "linalg", "testing", "util"]


# BlockArray appears to originate in this module
sys.modules[__name__].BlockArray.__module__ = __name__

# copy most of jnp without wrapping
_wrappers.add_attributes(to_dict=vars(), from_dict=jax.numpy.__dict__)

# wrap jnp funcs
_wrappers.wrap_recursively(
    vars(),
    creation_routines,
    partial(
        _wrappers.map_func_over_args,
        map_if_nested_args=["shape"],
        map_if_list_args=["device"],
    ),
)
_wrappers.wrap_recursively(vars(), mathematical_functions, _wrappers.map_func_over_args)

# _wrappers.wrap_recursively(vars(), reduction_functions, _wrappers.add_full_reduction)


def sum(a, axis=None, *args, **kwargs):
    if axis is not None:
        return jax.numpy.sum(a, axis, *args, **kwargs)

    return jax.numpy.sum(ravel(a), axis, *args, **kwargs)


def ravel(ba: Union[Array | TransparentTuple]) -> Array:
    """Completely flatten a :class:`BlockArray` into a single ``Array``.

    When called on an ``Array``, flattens the array.

    Args:
        ba: The :class:`BlockArray` to flatten.

    Returns:
        `ba` flattened into a single ``Array.``
    """
    if isinstance(ba, TransparentTuple):
        return jax.numpy.concatenate([arr.ravel() for arr in ba])

    return ba.ravel()


# wrap testing funcs
_wrappers.wrap_recursively(
    vars(), testing_functions, partial(_wrappers.map_func_over_args, is_void=True)
)

# clean up
del np, _wrappers
