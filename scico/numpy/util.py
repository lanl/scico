# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions for working with jax arrays and BlockArrays."""


from __future__ import annotations

import warnings
from math import prod
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import jax

import scico.numpy as snp
from scico.typing import ArrayIndex, Axes, AxisIndex, BlockShape, DType, Shape

from ._blockarray import BlockArray


def ensure_on_device(
    *arrays: Union[np.ndarray, snp.Array, BlockArray]
) -> Union[snp.Array, BlockArray]:
    """Cast numpy arrays to jax arrays.

    Cast numpy arrays to jax arrays and leave jax arrays and BlockArrays,
    as they are. This is intended to be used when initializing optimizers
    and functionals so that all arrays are either jax arrays or
    BlockArrays.

    Args:
        *arrays: One or more input arrays (numpy array, jax array, or
            BlockArray).

    Returns:
        Array or arrays, modified where appropriate.

    Raises:
        TypeError: If the arrays contain anything that is neither
           numpy array, jax array, nor BlockArray.
    """
    arrays = list(arrays)

    for i, array in enumerate(arrays):
        if isinstance(array, np.ndarray):
            warnings.warn(
                f"Argument {i+1} of {len(arrays)} is a numpy array. "
                "Will cast it to a jax array. "
                f"To suppress this warning cast all numpy arrays to jax arrays.",
                stacklevel=2,
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
                raise ValueError(
                    "Parameter axes cannot be None without a default or shape specified."
                )
            axes = list(range(len(shape)))
        else:
            axes = default
    elif isinstance(axes, (list, tuple)):
        axes = axes
    elif isinstance(axes, int):
        axes = (axes,)
    else:
        raise ValueError(f"Could not understand axes {axes} as a list of axes.")
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
    idx_shape: List[Optional[int]] = list(shape)
    offset = 0
    for axis, ax_idx in enumerate(idx):
        if ax_idx is Ellipsis:
            offset = len(shape) - len(idx)
            continue
        idx_shape[axis + offset] = slice_length(shape[axis + offset], ax_idx)
    return tuple(filter(lambda x: x is not None, idx_shape))  # type: ignore


def no_nan_divide(
    x: Union[BlockArray, snp.Array], y: Union[BlockArray, snp.Array]
) -> Union[BlockArray, snp.Array]:
    """Return `x/y`, with 0 instead of :data:`~numpy.NaN` where `y` is 0.

    Args:
        x: Numerator.
        y: Denominator.

    Returns:
        `x / y` with 0 wherever `y == 0`.
    """

    return snp.where(y != 0, snp.divide(x, snp.where(y != 0, y, 1)), 0)


def shape_to_size(shape: Union[Shape, BlockShape]) -> int:
    r"""Compute array size corresponding to a specified shape.

    Compute array size corresponding to a specified shape, which may be
    nested, i.e. corresponding to a :class:`.BlockArray`.

    Args:
        shape: A shape tuple.

    Returns:
        The number of elements in an array or :class:`.BlockArray` with
        shape `shape`.
    """

    if is_nested(shape):
        return sum(prod(s) for s in shape)

    return prod(shape)


def is_nested(x: Any) -> bool:
    """Check if input is a list/tuple containing at least one list/tuple.

    Args:
        x: Object to be tested.

    Returns:
        ``True`` if `x` is a list/tuple containing at least one
        list/tuple, ``False`` otherwise.

    Example:
        >>> is_nested([1, 2, 3])
        False
        >>> is_nested([(1,2), (3,)])
        True
        >>> is_nested([[1, 2], 3])
        True

    """
    return isinstance(x, (list, tuple)) and any([isinstance(_, (list, tuple)) for _ in x])


def broadcast_nested_shapes(
    shape_a: Union[Shape, BlockShape], shape_b: Union[Shape, BlockShape]
) -> Union[Shape, BlockShape]:
    r"""Compute the result of broadcasting on array shapes.

    Compute the result of applying a broadcasting binary operator to
    (block) arrays with (possibly nested) shapes `shape_a` and `shape_b`.
    Extends :func:`numpy.broadcast_shapes` to also support the nested
    tuple shapes of :class:`.BlockArray`\ s.

    Args:
        shape_a: First array shape.
        shape_b: Second array shape.

    Returns:
        A (possibly nested) shape tuple.

    Example:
        >>> broadcast_nested_shapes(((1, 1, 3), (2, 3, 1)), ((2, 3,), (2, 1, 4)))
        ((1, 2, 3), (2, 3, 4))
    """
    if not is_nested(shape_a) and not is_nested(shape_b):
        return snp.broadcast_shapes(shape_a, shape_b)

    if is_nested(shape_a) and not is_nested(shape_b):
        return tuple(snp.broadcast_shapes(s, shape_b) for s in shape_a)

    if not is_nested(shape_a) and is_nested(shape_b):
        return tuple(snp.broadcast_shapes(shape_a, s) for s in shape_b)

    if is_nested(shape_a) and is_nested(shape_b):
        return tuple(snp.broadcast_shapes(s_a, s_b) for s_a, s_b in zip(shape_a, shape_b))


def is_real_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is real.

    Args:
        dtype: A :mod:`numpy` or :mod:`scico.numpy` dtype (e.g.
               :attr:`~numpy.float32`, :attr:`~numpy.complex64`).

    Returns:
        ``False`` if the dtype is complex, otherwise ``True``.
    """
    return snp.dtype(dtype).kind != "c"


def is_complex_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is complex.

    Args:
        dtype: A :mod:`numpy` or :mod:`scico.numpy` dtype (e.g.
               :attr:`~numpy.float32`, :attr:`~numpy.complex64`).

    Returns:
        ``True`` if the dtype is complex, otherwise ``False``.
    """
    return snp.dtype(dtype).kind == "c"


def real_dtype(dtype: DType) -> DType:
    """Construct the corresponding real dtype for a given complex dtype.

    Construct the corresponding real dtype for a given complex dtype,
    e.g. the real dtype corresponding to :attr:`~numpy.complex64` is
    :attr:`~numpy.float32`.

    Args:
        dtype: A complex numpy or scico.numpy dtype (e.g.
               :attr:`~numpy.complex64`, :attr:`~numpy.complex128`).

    Returns:
        The real dtype corresponding to the input dtype
    """

    return snp.zeros(1, dtype).real.dtype


def complex_dtype(dtype: DType) -> DType:
    """Construct the corresponding complex dtype for a given real dtype.

    Construct the corresponding complex dtype for a given real dtype,
    e.g. the complex dtype corresponding to :attr:`~numpy.float32` is
    :attr:`~numpy.complex64`.

    Args:
        dtype: A real numpy or scico.numpy dtype (e.g. :attr:`~numpy.float32`,
               :attr:`~numpy.float64`).

    Returns:
        The complex dtype corresponding to the input dtype.
    """

    return (snp.zeros(1, dtype) + 1j).dtype
