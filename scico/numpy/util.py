""" Utility functions for working with BlockArrays and DeviceArrays. """

from math import prod
from typing import Any, Union

import scico.numpy as snp
from scico.typing import Axes, BlockShape, DType, JaxArray, Shape

from .blockarray import BlockArray


def no_nan_divide(
    x: Union[BlockArray, JaxArray], y: Union[BlockArray, JaxArray]
) -> Union[BlockArray, JaxArray]:
    """Return `x/y`, with 0 instead of NaN where `y` is 0.

    Args:
        x: Numerator.
        y: Denominator.

    Returns:
        `x / y` with 0 wherever `y == 0`.
    """

    return snp.where(y != 0, snp.divide(x, snp.where(y != 0, y, 1)), 0)


def shape_to_size(shape: Union[Shape, BlockShape]) -> Axes:
    r"""Compute the size corresponding to a (possibly nested) shape.

    Args:
       shape: A shape tuple; possibly tuples.
    """

    if is_nested(shape):
        return sum(prod(s) for s in shape)

    return prod(shape)


def is_nested(x: Any) -> bool:
    """Check if input is a list/tuple containing at least one list/tuple.

    Args:
        x: Object to be tested.

    Returns:
        ``True`` if `x` is a list/tuple of list/tuples, ``False`` otherwise.


    Example:
        >>> is_nested([1, 2, 3])
        False
        >>> is_nested([(1,2), (3,)])
        True
        >>> is_nested([[1, 2], 3])
        True

    """
    if isinstance(x, (list, tuple)):
        return any([isinstance(_, (list, tuple)) for _ in x])
    return False


def is_real_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is real.

    Args:
        dtype: A numpy or scico.numpy dtype (e.g. np.float32,
               snp.complex64).

    Returns:
        ``False`` if the dtype is complex, otherwise ``True``.
    """
    return snp.dtype(dtype).kind != "c"


def is_complex_dtype(dtype: DType) -> bool:
    """Determine whether a dtype is complex.

    Args:
        dtype: A numpy or scico.numpy dtype (e.g. ``np.float32``,
               ``np.complex64``).

    Returns:
        ``True`` if the dtype is complex, otherwise ``False``.
    """
    return snp.dtype(dtype).kind == "c"


def real_dtype(dtype: DType) -> DType:
    """Construct the corresponding real dtype for a given complex dtype.

    Construct the corresponding real dtype for a given complex dtype,
    e.g. the real dtype corresponding to `np.complex64` is
    `np.float32`.

    Args:
        dtype: A complex numpy or scico.numpy dtype (e.g. ``np.complex64``,
               ``np.complex128``).

    Returns:
        The real dtype corresponding to the input dtype
    """

    return snp.zeros(1, dtype).real.dtype


def complex_dtype(dtype: DType) -> DType:
    """Construct the corresponding complex dtype for a given real dtype.

    Construct the corresponding complex dtype for a given real dtype,
    e.g. the complex dtype corresponding to ``np.float32`` is
    ``np.complex64``.

    Args:
        dtype: A real numpy or scico.numpy dtype (e.g. ``np.float32``,
               ``np.float64``).

    Returns:
        The complex dtype corresponding to the input dtype.
    """

    return (snp.zeros(1, dtype) + 1j).dtype
