# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions for arrays, array shapes, array indexing, etc."""


from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import jax
from jax.interpreters.pxla import ShardedDeviceArray
from jax.interpreters.xla import DeviceArray

from scico.numpy import BlockArray
from scico.typing import ArrayIndex, Axes, AxisIndex, JaxArray, Shape


def ensure_on_device(
    *arrays: Union[np.ndarray, JaxArray, BlockArray]
) -> Union[JaxArray, BlockArray]:
    """Cast ndarrays to DeviceArrays.

    Cast ndarrays to DeviceArrays and leaves DeviceArrays, BlockArrays,
    and ShardedDeviceArray as is. This is intended to be used when
    initializing optimizers and functionals so that all arrays are either
    DeviceArrays, BlockArrays, or ShardedDeviceArray.

    Args:
        *arrays: One or more input arrays (ndarray, DeviceArray,
           BlockArray, or ShardedDeviceArray).

    Returns:
        Modified array or arrays. Modified are only those that were
           necessary.

    Raises:
        TypeError: If the arrays contain something that is neither
           ndarray, DeviceArray, BlockArray, nor ShardedDeviceArray.
    """
    arrays = list(arrays)

    for i, array in enumerate(arrays):

        if isinstance(array, np.ndarray):
            warnings.warn(
                f"Argument {i+1} of {len(arrays)} is an np.ndarray. "
                f"Will cast it to DeviceArray. "
                f"To suppress this warning cast all np.ndarrays to DeviceArray first.",
                stacklevel=2,
            )

        elif not isinstance(
            array,
            (DeviceArray, BlockArray, ShardedDeviceArray),
        ):
            raise TypeError(
                "Each item of `arrays` must be ndarray, DeviceArray, BlockArray, or "
                f"ShardedDeviceArray; Argument {i+1} of {len(arrays)} is {type(arrays[i])}."
            )

        arrays[i] = jax.device_put(arrays[i])

    if len(arrays) == 1:
        return arrays[0]
    return arrays


def parse_axes(
    axes: Axes, shape: Optional[Shape] = None, default: Optional[List[int]] = None
) -> List[int]:
    """Normalize `axes` to a list and optionally ensure correctness.

    Normalize `axes` to a list and (optionally) ensure that entries refer
    to axes that exist in `shape`.

    Args:
        axes: User specification of one or more axes: int, list, tuple,
           or ``None``.
        shape: The shape of the array of which axes are being specified.
           If not ``None``, `axes` is checked to make sure its entries
           refer to axes that exist in `shape`.
        default: Default value to return if `axes` is ``None``. By
           default, `list(range(len(shape)))`.

    Returns:
        List of axes (never an int, never ``None``).
    """

    if axes is None:
        if default is None:
            if shape is None:
                raise ValueError("`axes` cannot be `None` without a default or shape specified.")
            axes = list(range(len(shape)))
        else:
            axes = default
    elif isinstance(axes, (list, tuple)):
        axes = axes
    elif isinstance(axes, int):
        axes = (axes,)
    else:
        raise ValueError(f"Could not understand axes {axes} as a list of axes")
    if shape is not None and max(axes) >= len(shape):
        raise ValueError(
            f"Invalid axes {axes} specified; each axis must be less than `len(shape)`={len(shape)}."
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate value in axes {axes}; each axis must be unique.")
    return axes


def slice_length(length: int, idx: AxisIndex) -> Optional[int]:
    """Determine the length of an array axis after indexing.

    Determine the length of an array axis after slicing. An exception is
    raised if the indexing expression is an integer that is out of bounds
    for the specified axis length. A value of ``None`` is returned for
    valid integer indexing expressions as an indication that the
    corresponding axis shape is an empty tuple; this value should be
    converted to a unit integer if the axis size is required.

    Args:
        length: Length of axis being sliced.
        idx: Indexing/slice to be applied to axis.

    Returns:
        Length of indexed/sliced axis.

    Raises:
        ValueError: If `idx` is an integer index that is out bounds for
            the axis length.
    """
    if idx is Ellipsis:
        return length
    if isinstance(idx, int):
        if idx < -length or idx > length - 1:
            raise ValueError(f"Index {idx} out of bounds for axis of length {length}.")
        return None
    start, stop, stride = idx.indices(length)
    if start > stop:
        start = stop
    return (stop - start + stride - 1) // stride


def indexed_shape(shape: Shape, idx: ArrayIndex) -> Tuple[int, ...]:
    """Determine the shape of an array after indexing/slicing.

    Args:
        shape: Shape of array.
        idx: Indexing expression.

    Returns:
        Shape of indexed/sliced array.

    Raises:
        ValueError: If `idx` is longer than `shape`.
    """
    if not isinstance(idx, tuple):
        idx = (idx,)
    if len(idx) > len(shape):
        raise ValueError(f"Slice {idx} has more dimensions than shape {shape}.")
    idx_shape = list(shape)
    offset = 0
    for axis, ax_idx in enumerate(idx):
        if ax_idx is Ellipsis:
            offset = len(shape) - len(idx)
            continue
        idx_shape[axis + offset] = slice_length(shape[axis + offset], ax_idx)
    return tuple(filter(lambda x: x is not None, idx_shape))
