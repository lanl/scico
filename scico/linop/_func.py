# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operators constructed from functions."""

from typing import Any, Callable, Optional, Union

import scico.numpy as snp
from scico.typing import BlockShape, DType, Shape

from ._linop import LinearOperator

__all__ = [
    "operator_from_function",
    "Tranpose",
    "Sum",
    "Pad",
]


def linop_from_function(f: Callable, classname: str, f_name: Optional[str] = None):
    """Make a :class:`LinearOperator` from a function.

    Example
    -------
    >>> Sum = linop_from_function(snp.sum, 'Sum')
    >>> H = Sum((2, 10), axis=1)
    >>> H @ snp.ones((2, 10))
    DeviceArray([10., 10.], dtype=float32)

    Args:
        f: Function from which to create a :class:`LinearOperator`.
        classname: Name of the resulting class.
        f_name: Name of `f` for use in docstrings. Useful for getting
            the correct version of wrapped functions. Defaults to
            `f"{f.__module__}.{f.__name__}"`.
    """

    if f_name is None:
        f_name = f"{f.__module__}.{f.__name__}"

    f_doc = rf"""

        Args:
            input_shape: Shape of input array.
            args: Positional arguments passed to :func:`{f_name}`.
            input_dtype: `dtype` for input argument.
                Defaults to ``float32``. If :class:`LinearOperator`
                implements complex-valued operations, this must be
                ``complex64`` for proper adjoint and gradient calculation.
            jit: If ``True``, call :meth:`~.LinearOperator.jit` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling
                :meth:`~.LinearOperator.jit` after the
                :class:`LinearOperator` is created.
            kwargs: Keyword arguments passed to :func:`{f_name}`.
        """

    def __init__(
        self,
        input_shape: Union[Shape, BlockShape],
        *args: Any,
        input_dtype: DType = snp.float32,
        jit: bool = True,
        **kwargs: Any,
    ):
        self._eval = lambda x: f(x, *args, **kwargs)
        super().__init__(input_shape, input_dtype=input_dtype, jit=jit)  # type: ignore

    OpClass = type(classname, (LinearOperator,), {"__init__": __init__})
    __class__ = OpClass  # needed for super() to work

    OpClass.__doc__ = f"Linear operator version of :func:`{f_name}`."
    OpClass.__init__.__doc__ = f_doc  # type: ignore

    return OpClass


Transpose = linop_from_function(snp.transpose, "Transpose", "scico.numpy.transpose")
Sum = linop_from_function(snp.sum, "Sum")
Pad = linop_from_function(snp.pad, "Pad", "scico.numpy.pad")
