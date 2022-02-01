# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

""":class:`scico.blockarray.BlockArray`-compatible
versions of :mod:`jax.numpy` functions.

This modules consists of functions from :mod:`jax.numpy` wrapped to
support compatibility with :class:`scico.blockarray.BlockArray`. This
module is a work in progress and therefore not all functions are
wrapped. Functions that have not been wrapped yet have WARNING text in
their documentation, below.
"""

import sys
from functools import wraps

import numpy as np

import jax
from jax import numpy as jnp

from scico.array import is_nested

# These functions rely on the definition of a BlockArray and must be in
# scico.blockarray to avoid a circular import
from scico.blockarray import (
    BlockArray,
    _block_array_matmul_wrapper,
    _block_array_reduction_wrapper,
    _block_array_ufunc_wrapper,
    _flatten_blockarrays,
    atleast_1d,
    reshape,
)
from scico.typing import BlockShape, JaxArray, Shape

from ._create import (
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from ._util import _attach_wrapped_func, _get_module_functions, _not_implemented

# Numpy constants
pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = np.inf
NINF = np.NINF
PZERO = np.PZERO
NZERO = np.NZERO
nan = np.nan

bool_ = jnp.bool_
uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
bfloat16 = jnp.bfloat16
float16 = jnp.float16
float32 = single = jnp.float32
float64 = double = jnp.float64
complex64 = csingle = jnp.complex64
complex128 = cdouble = jnp.complex128

dtype = jnp.dtype
newaxis = None

# Functions to which _block_array_ufunc_wrapper is to be applied
_ufunc_functions = [
    ("abs", jnp.abs),
    jnp.maximum,
    jnp.sign,
    jnp.where,
    jnp.true_divide,
    jnp.floor_divide,
    jnp.real,
    jnp.imag,
    jnp.conjugate,
    jnp.angle,
    jnp.exp,
    jnp.sqrt,
    jnp.log,
    jnp.log10,
]
# Functions to which _block_array_reduction_wrapper is to be applied
_reduction_functions = [
    jnp.count_nonzero,
    jnp.sum,
    jnp.mean,
    jnp.median,
    jnp.any,
    jnp.var,
    ("max", jnp.max),
    ("min", jnp.min),
    jnp.amin,
    jnp.amax,
    jnp.all,
    jnp.any,
]

dot = _block_array_matmul_wrapper(jnp.dot)
matmul = _block_array_matmul_wrapper(jnp.matmul)


@wraps(jnp.vdot)
def vdot(a, b):
    """Dot product of `a` and `b` (with first argument complex conjugated).
    Wrapped to work on `BlockArray`s."""
    if isinstance(a, BlockArray):
        a = a.ravel()
    if isinstance(b, BlockArray):
        b = b.ravel()
    return jnp.vdot(a, b)


vdot.__doc__ = ":func:`vdot` wrapped to operate on :class:`.BlockArray`" + "\n\n" + jnp.vdot.__doc__

# Attach wrapped functions to this module
_attach_wrapped_func(
    _ufunc_functions,
    _block_array_ufunc_wrapper,
    module_name=sys.modules[__name__],
    fix_mod_name=True,
)
_attach_wrapped_func(
    _reduction_functions,
    _block_array_reduction_wrapper,
    module_name=sys.modules[__name__],
    fix_mod_name=True,
)

# divide is just an alias to true_divide
divide = true_divide
conj = conjugate

# Find functions that exist in jax.numpy but not scico.numpy
# see jax.numpy.__init__.py
_not_implemented_functions = []
for name, func in _get_module_functions(jnp).items():
    if name not in globals():
        _not_implemented_functions.append((name, func))

_attach_wrapped_func(
    _not_implemented_functions, _not_implemented, module_name=sys.modules[__name__]
)


# these must be imported towards the end to avoid a circular import with
# linalg and _matrixop
from . import fft, linalg
