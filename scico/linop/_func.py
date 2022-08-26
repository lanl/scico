# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operators constructed from functions."""

from typing import Any, Callable, Optional, Sequence, Union

import scico.numpy as snp
from scico._autograd import linear_adjoint
from scico.numpy.util import indexed_shape, is_nested
from scico.typing import ArrayIndex, BlockShape, DType, JaxArray, Shape

from ._linop import LinearOperator

__all__ = ["operator_from_function", "Tranpose", "Sum", "Crop", "Pad", "Reshape", "Slice"]


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
Reshape = linop_from_function(snp.reshape, "Reshape")
Pad = linop_from_function(snp.pad, "Pad", "scico.numpy.pad")
Sum = linop_from_function(snp.sum, "Sum")


class Crop(LinearOperator):
    """A linear operator for cropping an array."""

    def __init__(
        self,
        crop_width: Union[int, Sequence],
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            crop_width: Specify the crop width using the same format as
                the `pad_width` parameter of :func:`snp.pad`.
            input_shape: Shape of input :any:`JaxArray`.
            input_dtype: `dtype` for input argument.
                Defaults to ``float32``. If this :class:`LinearOperator`
                implements complex-valued operations, this must be
                ``complex64`` for proper adjoint and gradient calculation.
            jit: If ``True``, jit the evaluation, adjoint, and gram
               functions of the :class:`LinearOperator`.
        """

        self.crop_width = crop_width
        # The crop function is defined as the adjoint of snp.pad
        pad = lambda x: snp.pad(x, pad_width=crop_width)
        # The output shape of this operator is the input shape of the corresponding
        # pad operation of which it is the adjoint. Since we don't know this output
        # shape, we assume that it can be computed by subtracting the difference in
        # output and input shapes resulting from applying the pad operator to the
        # input shape of this operator.
        tmp = pad(snp.zeros(input_shape, dtype=input_dtype))
        output_shape = tuple(2 * snp.array(input_shape) - snp.array(tmp.shape))
        pad_adjoint = linear_adjoint(pad, snp.zeros(output_shape, dtype=input_dtype))
        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            eval_fn=lambda x: pad_adjoint(x)[0],
            output_shape=output_shape,
            output_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )


class Slice(LinearOperator):
    """A linear operator for slicing an array."""

    def __init__(
        self,
        idx: ArrayIndex,
        input_shape: Union[Shape, BlockShape],
        input_dtype: DType = snp.float32,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        This operator may be applied to either a :any:`JaxArray` or a
        :class:`.BlockArray`. In the latter case, parameter `idx` must
        conform to the
        :ref:`BlockArray indexing requirements <blockarray_indexing>`.

        Args:
            idx: An array indexing expression, as generated by
                :data:`numpy.s_`, for example.
            input_shape: Shape of input :any:`JaxArray` or :class:`.BlockArray`.
            input_dtype: `dtype` for input argument.
                Defaults to ``float32``. If this :class:`LinearOperator`
                implements complex-valued operations, this must be
                ``complex64`` for proper adjoint and gradient calculation.
            jit: If ``True``, jit the evaluation, adjoint, and gram
               functions of the :class:`LinearOperator`.
        """

        output_shape: Union[Shape, BlockShape]
        if is_nested(input_shape):
            output_shape = input_shape[idx]  # type: ignore
        else:
            output_shape = indexed_shape(input_shape, idx)

        self.idx: ArrayIndex = idx
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        return x[self.idx]
