# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Miscellaneous linear operator definitions."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import operator
from functools import partial
from typing import Optional, Union

import scico.numpy as snp
from scico.numpy import BlockArray
from scico.numpy.util import ensure_on_device, is_nested
from scico.operator._operator import _wrap_mul_div_scalar
from scico.typing import BlockShape, DType, JaxArray, Shape

from ._linop import LinearOperator, _wrap_add_sub

__all__ = [
    "Diagonal",
    "Identity",
]


class Diagonal(LinearOperator):
    """Diagonal linear operator."""

    def __init__(
        self,
        diagonal: Union[JaxArray, BlockArray],
        input_shape: Optional[Shape] = None,
        input_dtype: Optional[DType] = None,
        **kwargs,
    ):
        r"""
        Args:
            diagonal: Diagonal elements of this :class:`LinearOperator`.
            input_shape:  Shape of input array. By default, equal to
               `diagonal.shape`, but may also be set to a shape that is
               broadcast-compatiable with `diagonal.shape`.
            input_dtype: `dtype` of input argument. The default,
               ``None``, means `diagonal.dtype`.
        """

        self.diagonal = ensure_on_device(diagonal)

        if input_shape is None:
            input_shape = self.diagonal.shape

        if input_dtype is None:
            input_dtype = self.diagonal.dtype

        if isinstance(diagonal, BlockArray) and is_nested(input_shape):
            output_shape = (snp.empty(input_shape) * diagonal).shape
        elif not isinstance(diagonal, BlockArray) and not is_nested(input_shape):
            output_shape = snp.broadcast_shapes(input_shape, self.diagonal.shape)
        elif isinstance(diagonal, BlockArray):
            raise ValueError("`diagonal` was a BlockArray but `input_shape` was not nested.")
        else:
            raise ValueError("`diagonal` was a not BlockArray but `input_shape` was nested.")

        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_shape=output_shape,
            output_dtype=input_dtype,
            **kwargs,
        )

    def _eval(self, x):
        return x * self.diagonal

    @partial(_wrap_add_sub, op=operator.add)
    def __add__(self, other):
        if self.diagonal.shape == other.diagonal.shape:
            return Diagonal(diagonal=self.diagonal + other.diagonal)
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}")

    @partial(_wrap_add_sub, op=operator.sub)
    def __sub__(self, other):
        if self.diagonal.shape == other.diagonal.shape:
            return Diagonal(diagonal=self.diagonal - other.diagonal)
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}")

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return Diagonal(diagonal=self.diagonal * scalar)

    @_wrap_mul_div_scalar
    def __rmul__(self, scalar):
        return Diagonal(diagonal=self.diagonal * scalar)

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return Diagonal(diagonal=self.diagonal / scalar)


class Identity(Diagonal):
    """Identity operator."""

    def __init__(
        self, input_shape: Union[Shape, BlockShape], input_dtype: DType = snp.float32, **kwargs
    ):
        """
        Args:
            input_shape: Shape of input array.
        """
        super().__init__(diagonal=snp.ones(input_shape, dtype=input_dtype), **kwargs)

    def _eval(self, x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        return x

    def __rmatmul__(self, x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        return x
