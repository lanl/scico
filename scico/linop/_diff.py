# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Finite difference linear operator class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional

import numpy as np

import scico.numpy as snp
from scico.array import parse_axes
from scico.typing import Axes, DType, JaxArray, Shape

from ._linop import LinearOperator
from ._stack import LinearOperatorStack


class FiniteDifference(LinearOperatorStack):
    """Finite Difference operator.

    Computes finite differences along the specified axes, returning the
    results in a `DeviceArray` (whenever possible) or `BlockArray`. See
    :class:`LinearOperatorStack` for details on how this choice is made.

    Example
    -------
    >>> A = FiniteDifference((2, 3))
    >>> x = snp.array([[1, 2, 4],
                       [0, 4, 1]])
    >>> (A @ x)[0]
    DeviceArray([[-1,  2, -3]], dtype=int32)
    >>> (A @ x)[1]
    DeviceArray([[ 1,  2],
                 [ 4, -3]], dtype=int32)
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        axes: Optional[Axes] = None,
        append: Optional[float] = None,
        circular: bool = False,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                `float32`. If `LinearOperator` implements complex-valued
                operations, this must be `complex64` for proper adjoint
                and gradient calculation.
            axes: Axis or axes over which to apply finite difference
                operator. If not specified, or ``None``, differences are
                evaluated along all axes.
            append: Value to append to the input along each axis before
                taking differences. Zero is a typical choice. If not
                `None`, `circular` must be ``False``.
            circular: If ``True``, perform circular differences, i.e.,
                include x[-1] - x[0]. If ``True``, `append` must be
                `None`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        self.axes = parse_axes(axes, input_shape)

        if axes is None:
            axes_list = range(len(input_shape))
        elif isinstance(axes, (list, tuple)):
            axes_list = axes
        else:
            axes_list = (axes,)
        single_kwargs = dict(append=append, circular=circular, jit=False, input_dtype=input_dtype)
        ops = [FiniteDifferenceSingleAxis(axis, input_shape, **single_kwargs) for axis in axes_list]

        super().__init__(
            ops,
            jit=jit,
            **kwargs,
        )


class FiniteDifferenceSingleAxis(LinearOperator):
    """Finite Difference operator acting along a single axis."""

    def __init__(
        self,
        axis: int,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        append: Optional[float] = None,
        circular: bool = False,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            axis: Axis over which to apply finite difference operator.
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                `float32`. If `LinearOperator` implements complex-valued
                operations, this must be `complex64` for proper adjoint
                and gradient calculation.
            append: Value to append to the input along `axis` before
                taking differences. Defaults to 0.
            circular: If ``True``, perform circular differences, i.e.,
                include x[-1] - x[0]. If ``True``, `append` must be
                ``None``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if not isinstance(axis, int):
            raise TypeError(f"Expected `axis` to be of type int, got {type(axis)} instead")

        if axis >= len(input_shape):
            raise ValueError(
                f"Invalid axis {axis} specified; `axis` must be less than "
                f"`len(input_shape)`={len(input_shape)}"
            )

        self.axis = axis

        if append is not None and circular:
            raise ValueError(
                "`append` and `circular` are mutually exclusive but both were specified"
            )

        self.circular = circular
        self.append = append

        if self.append is None and not circular:
            output_shape = tuple(x - (i == axis) for i, x in enumerate(input_shape))
        else:
            output_shape = input_shape

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        if self.circular:
            # set append to the first slice along the specified axis
            ind = tuple(
                slice(0, 1) if i == self.axis else slice(None) for i in range(len(self.input_shape))
            )
            append = x[ind]
        else:
            append = self.append

        return snp.diff(x, axis=self.axis, append=append)
