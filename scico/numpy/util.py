# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions for working with jax arrays and BlockArrays."""

from __future__ import annotations

import collections
from math import prod
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

import jax

from typing_extensions import TypeGuard

import scico.numpy as snp
from scico.typing import ArrayIndex, Axes, AxisIndex, BlockShape, DType, Shape


def transpose_ntpl_of_list(ntpl: NamedTuple) -> List[NamedTuple]:
    """Convert a namedtuple of lists/arrays to a list of namedtuples.

    Args:
        ntpl: Named tuple object to be transposed.

    Returns:
        List of namedtuple objects.
    """
    cls = ntpl.__class__
    numentry = len(ntpl[0]) if isinstance(ntpl[0], list) else ntpl[0].shape[0]
    nfields = len(ntpl._fields)
    return [cls(*[ntpl[m][n] for m in range(nfields)]) for n in range(numentry)]


def transpose_list_of_ntpl(ntlist: List[NamedTuple]) -> NamedTuple:
    """Convert a list of namedtuples to namedtuple of lists.

    Args:
        ntpl: List of namedtuple objects to be transposed.

    Returns:
        Named tuple of lists.
    """
    cls = ntlist[0].__class__
    numentry = len(ntlist)
    nfields = len(ntlist[0])
    return cls(*[[ntlist[m][n] for m in range(numentry)] for n in range(nfields)])  # type: ignore


def namedtuple_to_array(ntpl: NamedTuple) -> snp.Array:
    """Convert a namedtuple to an array.

    Convert a :func:`collections.namedtuple` object to a
    :class:`numpy.ndarray` object that can be saved using
    :func:`numpy.savez`.

    Args:
        ntpl: Named tuple object to be converted to ndarray.

    Returns:
      Array representation of input named tuple.
    """
    return np.asarray(
        {
            "name": ntpl.__class__.__name__,
            "fields": ntpl._fields,
            "data": {fname: fval for fname, fval in zip(ntpl._fields, ntpl)},
        }
    )


def array_to_namedtuple(array: snp.Array) -> NamedTuple:
    """Convert an array representation of a namedtuple back to a namedtuple.

    Convert a :class:`numpy.ndarray` object constructed by
    :func:`namedtuple_to_array` back to the original
    :func:`collections.namedtuple` representation.

    Args:
      Array representation of named tuple constructed by
        :func:`namedtuple_to_array`.

    Returns:
      Named tuple object with the same name and fields as the original
      named tuple object provided to :func:`namedtuple_to_array`.
    """
    cls = collections.namedtuple(array.item()["name"], array.item()["fields"])  # type: ignore
    return cls(**array.item()["data"])


def normalize_axes(
    axes: Optional[Axes],
    shape: Optional[Shape] = None,
    default: Optional[List[int]] = None,
    sort: bool = False,
) -> Sequence[int]:
    """Normalize `axes` to a sequence and optionally ensure correctness.

    Normalize `axes` to a tuple or list and (optionally) ensure that
    entries refer to axes that exist in `shape`.

    Args:
        axes: User specification of one or more axes: int, list, tuple,
           or ``None``. Negative values count from the last to the first
           axis.
        shape: The shape of the array of which axes are being specified.
           If not ``None``, `axes` is checked to make sure its entries
           refer to axes that exist in `shape`.
        default: Default value to return if `axes` is ``None``. By
           default, `tuple(range(len(shape)))`.
        sort: If ``True``, sort the returned axis indices.

    Returns:
        Tuple or list of axes (never an int, never ``None``). The output
        will only be a list if the input is a list or if the input is
        ``None`` and `defaults` is a list.
    """

    if axes is None:
        if default is None:
            if shape is None:
                raise ValueError(
                    "Argument 'axes' cannot be None without a default or shape specified."
                )
            axes = tuple(range(len(shape)))
        else:
            axes = default
    elif isinstance(axes, (list, tuple)):
        axes = axes
    elif isinstance(axes, int):
        axes = (axes,)
    else:
        raise ValueError(f"Could not understand argument 'axes' {axes} as a list of axes.")
    if shape is not None:
        if min(axes) < 0:
            axes = tuple([len(shape) + a if a < 0 else a for a in axes])
        if max(axes) >= len(shape):
            raise ValueError(
                f"Invalid axes {axes} specified; each axis must be less than `len(shape)`={len(shape)}."
            )
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate value in axes {axes}; each axis must be unique.")
    if sort:
        axes = tuple(sorted(axes))
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
            the axis length or if the type of `idx` is not one of
            `Ellipsis`, `int`, or `slice`.
    """
    if idx is Ellipsis:
        return length
    if isinstance(idx, int):
        if idx < -length or idx > length - 1:
            raise ValueError(f"Index {idx} out of bounds for axis of length {length}.")
        return None
    if not isinstance(idx, slice):
        raise ValueError(f"Index expression {idx} is of an unrecognized type.")
    start, stop, stride = idx.indices(length)
    if start > stop:
        start = stop
    return (stop - start + stride - 1) // stride


def indexed_shape(shape: Shape, idx: ArrayIndex) -> Tuple[int, ...]:
    """Determine the shape of an array after indexing/slicing.

    The indexed shape is determined by replicating the observed effects
    of NumPy/JAX array indexing/slicing syntax. It is significantly
    faster than :func:`.jax_indexed_shape`, and has a minimal memory
    footprint in all circumstances.

    Args:
        shape: Shape of array.
        idx: Indexing expression (singleton or tuple of `Ellipsis`,
           `int`, `slice`, or ``None`` (`np.newaxis`)).

    Returns:
        Shape of indexed/sliced array.

    Raises:
        ValueError: If any element of `idx` is not one of `Ellipsis`,
        `int`, `slice`, or ``None`` (`np.newaxis`), or if an integer
        index is out bounds for the corresponding axis length.
    """
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx_shape: List[Optional[int]] = list(shape)
    offset = 0
    newaxis = 0
    for axis, ax_idx in enumerate(idx):
        if ax_idx is None:
            idx_shape.insert(axis, 1)
            newaxis += 1
            continue
        if ax_idx is Ellipsis:
            offset = len(shape) - len(idx)
            continue
        idx_shape[axis + offset + newaxis] = slice_length(shape[axis + offset], ax_idx)
    return tuple(filter(lambda x: x is not None, idx_shape))  # type: ignore


def jax_indexed_shape(shape: Shape, idx: ArrayIndex) -> Tuple[int, ...]:
    """Determine the shape of an array after indexing/slicing.

    The indexed shape is determined by constructing and indexing an array
    of the appropriate shape, relying on :func:`jax.jit` to avoid memory
    allocation. It is potentially more reliable than
    :func:`.indexed_shape` because the indexing/slicing calculations are
    referred to JAX, but is significantly slower, and will involved
    potentially significant memory allocations if JIT is disabled, e.g.
    for debugging purposes.

    Args:
        shape: Shape of array.
        idx: Indexing expression (singleton or tuple of `Ellipsis`,
           `int`, `slice`, or ``None`` (`np.newaxis`)).

    Returns:
        Shape of indexed/sliced array.
    """
    if not isinstance(idx, tuple):
        idx = (idx,)

    # Convert any slices to its representation (slice, (start, stop, step))
    # allowing hashing, needed for jax.jit
    idx = tuple(exp.__reduce__() if isinstance(exp, slice) else exp for exp in idx)  # type: ignore

    def get_shape(in_shape, ind_expr):
        # convert slices representations back to slices
        ind_expr = tuple(
            (slice(*exp[1]) if isinstance(exp, tuple) and len(exp) > 0 and exp[0] == slice else exp)
            for exp in ind_expr
        )
        return jax.numpy.empty(in_shape)[ind_expr].shape

    # This compiles each time it gets new arguments because all arguments are static.
    f = jax.jit(get_shape, static_argnums=(0, 1))

    return tuple(t.item() for t in f(shape, idx))  # type: ignore


def no_nan_divide(
    x: Union[snp.BlockArray, snp.Array], y: Union[snp.BlockArray, snp.Array]
) -> Union[snp.BlockArray, snp.Array]:
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
    nested, i.e. corresponding to a :class:`BlockArray`.

    Args:
        shape: A shape tuple.

    Returns:
        The number of elements in an array or :class:`BlockArray` with
        shape `shape`.
    """

    if is_nested(shape):
        return sum(prod(s) for s in shape)  # type: ignore

    return prod(shape)  # type: ignore


def is_arraylike(x: Any) -> bool:
    """Check if input is of type :class:`jax.ArrayLike`.

    `isinstance(x, jax.typing.ArrayLike)` does not work in Python < 3.10,
    see https://jax.readthedocs.io/en/latest/jax.typing.html#jax-typing-best-practices.

    Args:
        x: Object to be tested.

    Returns:
        ``True`` if `x` is an ArrayLike, ``False`` otherwise.
    """
    return isinstance(x, (np.ndarray, jax.Array)) or np.isscalar(x)


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


def is_collapsible(shapes: Sequence[Union[Shape, BlockShape]]) -> bool:
    """Determine whether a sequence of shapes can be collapsed.

    Return ``True`` if the a list of shapes represent arrays that can
    be stacked, i.e., they are all the same."""
    return all(s == shapes[0] for s in shapes)


def is_blockable(shapes: Sequence[Union[Shape, BlockShape]]) -> TypeGuard[Union[Shape, BlockShape]]:
    """Determine whether a sequence of shapes could be a :class:`BlockArray` shape.

    Return ``True`` if the sequence of shapes represent arrays that can
    be combined into a :class:`BlockArray`, i.e., none are nested."""
    return not any(is_nested(s) for s in shapes)


def broadcast_nested_shapes(
    shape_a: Union[Shape, BlockShape], shape_b: Union[Shape, BlockShape]
) -> Union[Shape, BlockShape]:
    r"""Compute the result of broadcasting on array shapes.

    Compute the result of applying a broadcasting binary operator to
    (block) arrays with (possibly nested) shapes `shape_a` and `shape_b`.
    Extends :func:`numpy.broadcast_shapes` to also support the nested
    tuple shapes of :class:`BlockArray`\ s.

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

    raise RuntimeError("Unexpected case encountered in broadcast_nested_shapes.")


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


def is_scalar_equiv(s: Any) -> bool:
    """Determine whether an object is a scalar or is scalar-equivalent.

    Determine whether an object is a scalar or a singleton array.

    Args:
        s: Object to be tested.

    Returns:
        ``True`` if the object is a scalar or a singleton array,
        otherwise ``False``.
    """
    return snp.isscalar(s) or (isinstance(s, jax.Array) and s.ndim == 0)
