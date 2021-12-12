# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construct wrapped versions of :mod:`jax.numpy.linalg` functions.

This modules consists of functions from :mod:`jax.numpy.linalg`. Some of
these functions are wrapped to support compatibility with
:class:`scico.blockarray.BlockArray` and are documented here. The
remaining functions are imported directly from :mod:`jax.numpy.linalg`.
While they can be imported from the :mod:`scico.numpy.linalg` namespace,
they are not documented here; please consult the documentation for the
source module :mod:`jax.numpy.linalg`.
"""


import sys
from functools import wraps

import jax
import jax.numpy.linalg as jla

from scico.blockarray import _block_array_reduction_wrapper
from scico.linop._matrix import MatrixOperator

from ._util import _attach_wrapped_func, _not_implemented


def _extract_if_matrix(x):
    if isinstance(x, MatrixOperator):
        return x.A
    return x


def _matrixop_linalg_wrapper(func):
    """Wrap :mod:`jax.numpy.linalg` functions.

    Wrap :mod:`jax.numpy.linalg` functions for joint operation on
    `MatrixOperator` and `DeviceArray`."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        all_args = args + tuple(kwargs.items())
        if any([isinstance(_, MatrixOperator) for _ in all_args]):
            args = [_extract_if_matrix(_) for _ in args]
            kwargs = {key: _extract_if_matrix(val) for key, val in kwargs.items()}
        return func(*args, **kwargs)

    if hasattr(func, "__doc__"):
        wrapper.__doc__ = (
            f":func:`{func.__name__}` wrapped to operate on :class:`.MatrixOperator`"
            + "\n\n"
            + func.__doc__
        )
    return wrapper


# norm is a reduction and gets both block array and matrixop wrapping
norm = _block_array_reduction_wrapper(_matrixop_linalg_wrapper(jla.norm))

svd = _matrixop_linalg_wrapper(jla.svd)
cond = _matrixop_linalg_wrapper(jla.cond)
det = _matrixop_linalg_wrapper(jla.det)
eig = _matrixop_linalg_wrapper(jla.eig)
eigh = _matrixop_linalg_wrapper(jla.eigh)
eigvals = _matrixop_linalg_wrapper(jla.eigvals)
eigvalsh = _matrixop_linalg_wrapper(jla.eigvalsh)
inv = _matrixop_linalg_wrapper(jla.inv)
lstsq = _matrixop_linalg_wrapper(jla.lstsq)
matrix_power = _matrixop_linalg_wrapper(jla.matrix_power)
matrix_rank = _matrixop_linalg_wrapper(jla.matrix_rank)
pinv = _matrixop_linalg_wrapper(jla.pinv)
qr = _matrixop_linalg_wrapper(jla.qr)
slogdet = _matrixop_linalg_wrapper(jla.slogdet)
solve = _matrixop_linalg_wrapper(jla.solve)


# multidot is somewhat unique
def multi_dot(arrays, *, precision=None):
    """Compute the dot product of two or more arrays.

    Compute the dot product of two or more arrays.
    Wrapped to work with `MatrixOperator`s.
    """
    arrays_ = [_extract_if_matrix(_) for _ in arrays]
    return jla.multi_dot(arrays_, precision=precision)


multi_dot.__doc__ = (
    f":func:`multi_dot` wrapped to operate on :class:`.MatrixOperator`"
    + "\n\n"
    + jla.multi_dot.__doc__
)


# Attach unwrapped functions
# jla.tensorinv, jla.tensorsolve use n-dim arrays; not supported by MatrixOperator
_not_implemented_functions = []
for name, func in jax._src.util.get_module_functions(jla).items():
    if name not in globals():
        _not_implemented_functions.append((name, func))

_attach_wrapped_func(
    _not_implemented_functions, _not_implemented, module_name=sys.modules[__name__]
)
