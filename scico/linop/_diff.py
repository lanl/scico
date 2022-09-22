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

from typing import Literal, Optional, Union

import numpy as np

import scico.numpy as snp
from scico.numpy.util import parse_axes
from scico.typing import Axes, DType, JaxArray, Shape

from ._linop import LinearOperator
from ._stack import VerticalStack


class FiniteDifference(VerticalStack):
    """Finite difference operator.

    Computes finite differences along the specified axes, returning the
    results in a `DeviceArray` (whenever possible) or :class:`BlockArray`.
    See :class:`VerticalStack` for details on how this choice is made.
    See :class:`SingleAxisFiniteDifference` for the mathematical
    implications of the different boundary handling options `prepend`,
    `append`, and `circular`.

    Example
    -------
    >>> A = FiniteDifference((2, 3))
    >>> x = snp.array([[1, 2, 4],
    ...                [0, 4, 1]])
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
        prepend: Optional[Union[Literal[0], Literal[1]]] = None,
        append: Optional[Union[Literal[0], Literal[1]]] = None,
        circular: bool = False,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                ``float32``. If :class:`LinearOperator` implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            axes: Axis or axes over which to apply finite difference
                operator. If not specified, or ``None``, differences are
                evaluated along all axes.
            prepend: Flag indicating handling of the left/top/etc.
                boundary. If ``None``, there is no boundary extension.
                Values of `0` or `1` indicate respectively that zeros or
                the initial value in the array are prepended to the
                difference array.
            append: Flag indicating handling of the right/bottom/etc.
                boundary. If ``None``, there is no boundary extension.
                Values of `0` or `1` indicate respectively that zeros or
                -1 times the final value in the array are appended to the
                difference array.
            circular: If ``True``, perform circular differences, i.e.,
                include x[-1] - x[0]. If ``True``, `prepend` and `append
                must both be ``None``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """

        if axes is None:
            axes_list = tuple(range(len(input_shape)))
        elif isinstance(axes, (list, tuple)):
            axes_list = axes
        else:
            axes_list = (axes,)
        self.axes = parse_axes(axes_list, input_shape)
        single_kwargs = dict(
            input_dtype=input_dtype, prepend=prepend, append=append, circular=circular, jit=False
        )
        ops = [
            SingleAxisFiniteDifference(input_shape, axis=axis, **single_kwargs)
            for axis in axes_list
        ]

        super().__init__(
            ops,  # type: ignore
            jit=jit,
            **kwargs,
        )


class SingleAxisFiniteDifference(LinearOperator):
    r"""Finite difference operator acting along a single axis.

    By default (i.e. `prepend` and `append` set to ``None`` and `circular`
    set to ``False``), the difference operator corresponds to the matrix

    .. math::

       \left(\begin{array}{rrrrr}
       -1 & 1 & 0 & \ldots & 0\\
       0 & -1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & -1 & 1
       \end{array}\right) \;,

    mapping :math:`\mbb{R}^N \rightarrow \mbb{R}^{N-1}`, while if `circular`
    is ``True``, it corresponds to the :math:`\mbb{R}^N \rightarrow \mbb{R}^N`
    mapping

    .. math::

       \left(\begin{array}{rrrrr}
       -1 & 1 & 0 & \ldots & 0\\
       0 & -1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & -1 & 1\\
       1 & 0 & \dots & 0 & -1
       \end{array}\right) \;.

    Other possible choices include `prepend` set to ``None`` and `append`
    set to `0`, giving the :math:`\mbb{R}^N \rightarrow \mbb{R}^N`
    mapping

    .. math::

       \left(\begin{array}{rrrrr}
       -1 & 1 & 0 & \ldots & 0\\
       0 & -1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & -1 & 1\\
       0 & 0 & \dots & 0 & 0
       \end{array}\right) \;,

    and both `prepend` and `append` set to `1`, giving the
    :math:`\mbb{R}^N \rightarrow \mbb{R}^{N+1}` mapping

    .. math::

       \left(\begin{array}{rrrrr}
        1 & 0 & 0 & \ldots & 0\\
       -1 & 1 & 0 & \ldots & 0\\
       0 & -1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & -1 & 1\\
       0 & 0 & \dots & 0 & -1
       \end{array}\right) \;.
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        axis: int = -1,
        prepend: Optional[Union[Literal[0], Literal[1]]] = None,
        append: Optional[Union[Literal[0], Literal[1]]] = None,
        circular: bool = False,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                ``float32``. If :class:`LinearOperator` implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            axis: Axis over which to apply finite difference operator.
            prepend: Flag indicating handling of the left/top/etc.
                boundary. If ``None``, there is no boundary extension.
                Values of `0` or `1` indicate respectively that zeros or
                the initial value in the array are prepended to the
                difference array.
            append: Flag indicating handling of the right/bottom/etc.
                boundary. If ``None``, there is no boundary extension.
                Values of `0` or `1` indicate respectively that zeros or
                -1 times the final value in the array are appended to the
                difference array.
            circular: If ``True``, perform circular differences, i.e.,
                include x[-1] - x[0]. If ``True``, `prepend` and `append
                must both be ``None``.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """

        if not isinstance(axis, int):
            raise TypeError(f"Expected axis to be of type int, got {type(axis)} instead.")

        if axis < 0:
            axis = len(input_shape) + axis
        if axis >= len(input_shape):
            raise ValueError(
                f"Invalid axis {axis} specified; axis must be less than "
                f"len(input_shape)={len(input_shape)}."
            )

        self.axis = axis

        if circular and (prepend is not None or append is not None):
            raise ValueError(
                "Parameter circular must be False if either prepend or append is not None."
            )
        if prepend not in [None, 0, 1]:
            raise ValueError("Parameter prepend may only take values None, 0, or 1.")
        if append not in [None, 0, 1]:
            raise ValueError("Parameter append may only take values None, 0, or 1.")

        self.prepend = prepend
        self.append = append
        self.circular = circular

        if self.circular:
            output_shape = input_shape
        else:
            output_shape = tuple(
                x + ((i == axis) * ((self.prepend is not None) + (self.append is not None) - 1))
                for i, x in enumerate(input_shape)
            )

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        prepend = None
        append = None
        if self.circular:
            # Append a copy of the initial value at the end of the array so that the difference
            # array includes the difference across the right/bottom/etc. boundary.
            ind = tuple(
                slice(0, 1) if i == self.axis else slice(None) for i in range(len(self.input_shape))
            )
            append = x[ind]
        else:
            if self.prepend == 0:
                # Prepend a 0 to the difference array by prepending a copy of the initial value
                # before the difference is computed.
                ind = tuple(
                    slice(0, 1) if i == self.axis else slice(None)
                    for i in range(len(self.input_shape))
                )
                prepend = x[ind]
            elif self.prepend == 1:
                # Prepend a copy of the initial value to the difference array by prepending a 0
                # before the difference is computed.
                prepend = 0
            if self.append == 0:
                # Append a 0 to the difference array by appending a copy of the initial value
                # before the difference is computed.
                ind = tuple(
                    slice(-1, None) if i == self.axis else slice(None)
                    for i in range(len(self.input_shape))
                )
                append = x[ind]
            elif self.append == 1:
                # Append a copy of the initial value to the difference array by appending a 0
                # before the difference is computed.
                append = 0

        return snp.diff(x, axis=self.axis, prepend=prepend, append=append)
