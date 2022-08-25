# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Operators constructed from functions."""

from typing import Any, Callable, Optional, Union

import scico.numpy as snp
from scico.typing import BlockShape, DType, Shape

from ._operator import Operator

__all__ = [
    "operator_from_function",
    "Abs",
    "Angle",
    "Exp",
]


def operator_from_function(f: Callable, classname: str, f_name: Optional[str] = None):
    """Make an :class:`.Operator` from a function.

    Example
    -------
    >>> AbsVal = operator_from_function(snp.abs, 'AbsVal')
    >>> H = AbsVal((2,))
    >>> H(snp.array([1.0, -1.0]))
    DeviceArray([1., 1.], dtype=float32)

    Args:
        f: Function from which to create an :class:`.Operator`.
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
                Defaults to ``float32``. If :class:`.Operator` implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            jit: If ``True``, call :meth:`.Operator.jit` on this
                `Operator` to jit the forward, adjoint, and gram
                functions. Same as calling :meth:`.Operator.jit` after
                the :class:`.Operator` is created.
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

    OpClass = type(classname, (Operator,), {"__init__": __init__})
    __class__ = OpClass  # needed for super() to work

    OpClass.__doc__ = f"Operator version of :func:`{f_name}`."
    OpClass.__init__.__doc__ = f_doc  # type: ignore

    return OpClass


Abs = operator_from_function(snp.abs, "Abs", "scico.numpy.abs")
Angle = operator_from_function(snp.angle, "Angle", "scico.numpy.angle")
Exp = operator_from_function(snp.exp, "Exp", "scico.numpy.exp")
