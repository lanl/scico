# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions for creating new arrays."""

from typing import Union

import numpy as np

import jax
from jax import numpy as jnp

from scico.array import is_nested
from scico.blockarray import BlockArray
from scico.typing import BlockShape, DType, JaxArray, Shape


def zeros(
    shape: Union[Shape, BlockShape], dtype: DType = np.float32
) -> Union[JaxArray, BlockArray]:
    """Return a new array of given shape and type, filled with zeros.

    If `shape` is a list of tuples, returns a BlockArray of zeros.

    Args:
       shape: Shape of the new array.
       dtype: Desired data-type of the array. Default is `np.float32`.
    """
    if is_nested(shape):
        return BlockArray.zeros(shape, dtype=dtype)
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Union[Shape, BlockShape], dtype: DType = np.float32) -> Union[JaxArray, BlockArray]:
    """Return a new array of given shape and type, filled with ones.

    If `shape` is a list of tuples, returns a BlockArray of ones.

    Args:
       shape: Shape of the new array.
       dtype: Desired data-type of the array. Default is `np.float32`.
    """
    if is_nested(shape):
        return BlockArray.ones(shape, dtype=dtype)
    return jnp.ones(shape, dtype=dtype)


def empty(
    shape: Union[Shape, BlockShape], dtype: DType = np.float32
) -> Union[JaxArray, BlockArray]:
    """Return a new array of given shape and type, filled with zeros.

    If `shape` is a list of tuples, returns a BlockArray of zeros.

    Args:
       shape: Shape of the new array.
       dtype: Desired data-type of the array. Default is `np.float32`.
    """
    if is_nested(shape):
        return BlockArray.empty(shape, dtype=dtype)
    return jnp.empty(shape, dtype=dtype)


def full(
    shape: Union[Shape, BlockShape],
    fill_value: Union[float, complex],
    dtype: DType = None,
) -> Union[JaxArray, BlockArray]:
    """Return a new array of given shape and type, filled with `fill_value`.

    If `shape` is a list of tuples, returns a BlockArray filled with
    `fill_value`.

    Args:
       shape: Shape of the new array.
       fill_value : Fill value.
       dtype: Desired data-type of the array. The default, None,
           means `np.array(fill_value).dtype`.
    """
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(type(fill_value))
    if is_nested(shape):
        return BlockArray.full(shape, fill_value=fill_value, dtype=dtype)
    return jnp.full(shape, fill_value=fill_value, dtype=dtype)


def zeros_like(x: Union[JaxArray, BlockArray], dtype=None):
    """Return an array of zeros with same shape and type as a given array.

    If input is a BlockArray, returns a BlockArray of zeros with same
    shape and type as a given array.

    Args:
         x (array like): The shape and dtype of `x` define these
            attributes on the returned array.
         dtype (data-type, optional): Overrides the data type of the
            result.
    """
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(x.dtype)

    if isinstance(x, BlockArray):
        return BlockArray.zeros(x.shape, dtype=dtype)
    return jnp.zeros_like(x, dtype=dtype)


def empty_like(x: Union[JaxArray, BlockArray], dtype: DType = None):
    """Return an array of zeros with same shape and type as a given array.

    If input is a BlockArray, returns a BlockArray of zeros with same
    shape and type as a given array.

    Note: like :func:`jax.numpy.empty_like`, this does not return an
          uninitalized array.

    Args:
         x (array like): The shape and dtype of `x` define these
             attributes on the returned array.
         dtype (data-type, optional): Overrides the data type of the
             result.
    """
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(x.dtype)

    if isinstance(x, BlockArray):
        return BlockArray.zeros(x.shape, dtype=dtype)
    return jnp.zeros_like(x, dtype=dtype)


def ones_like(x: Union[JaxArray, BlockArray], dtype: DType = None):
    """Return an array of ones with same shape and type as a given array.

    If input is a BlockArray, returns a BlockArray of ones with same
    shape and type as a given array.

    Args:
         x (array like): The shape and dtype of `x` define these
             attributes on the returned array.
         dtype (data-type, optional):  Overrides the data type of the
             result.
    """
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(x.dtype)

    if isinstance(x, BlockArray):
        return BlockArray.ones(x.shape, dtype=dtype)
    return jnp.ones_like(x, dtype=dtype)


def full_like(
    x: Union[JaxArray, BlockArray], fill_value: Union[float, complex], dtype: DType = None
):
    """Return an array filled with `fill_value`.

    Return an array of with same shape and type as a given array, filled
    with `fill_value`. If input is a BlockArray, returns a BlockArray of
    `fill_value` with same shape and type as a given array.

    Args:
         x (array like): The shape and dtype of `x` define these
            attributes on the returned array.
         fill_value (scalar): Fill value.
         dtype (data-type, optional): Overrides the data type of the
            result.
    """
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(x.dtype)

    if isinstance(x, BlockArray):
        return BlockArray.full(x.shape, fill_value=fill_value, dtype=dtype)
    return jnp.full_like(x, fill_value=fill_value, dtype=dtype)
