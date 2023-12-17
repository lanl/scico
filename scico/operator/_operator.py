# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Operator base class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
from jax.dtypes import result_type

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_nested, shape_to_size
from scico.typing import BlockShape, DType, Shape


def _wrap_mul_div_scalar(func: Callable) -> Callable:
    r"""Wrapper function for multiplication and division operators.

    Wrapper function for defining `__mul__`, `__rmul__`, and
    `__truediv__` between a scalar and an `Operator`.

    If one of these binary operations are called in the form
    `binop(Operator, other)` and 'b' is a scalar, specialized
    :class:`.Operator` constructors can be called.

    Args:
        func: should be either `.__mul__()`, `.__rmul__()`,
           or `.__truediv__()`.

    Raises:
        TypeError: If a binop with the form `binop(Operator, other)` is
        called and `other` is not a scalar.
    """

    @wraps(func)
    def wrapper(a, b):
        if snp.util.is_scalar_equiv(b):
            return func(a, b)

        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}.")

    return wrapper


class Operator:
    """Generic operator class."""

    def __repr__(self):
        return f"""{type(self)}
shape       : {self.shape}
matrix_shape : {self.matrix_shape}
input_dtype : {self.input_dtype}
output_dtype : {self.output_dtype}
        """

    # See https://numpy.org/doc/stable/user/c-info.beyond-basics.html#ndarray.__array_priority__
    __array_priority__ = 1

    def __init__(
        self,
        input_shape: Union[Shape, BlockShape],
        output_shape: Optional[Union[Shape, BlockShape]] = None,
        eval_fn: Optional[Callable] = None,
        input_dtype: DType = np.float32,
        output_dtype: Optional[DType] = None,
        jit: bool = False,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            output_shape: Shape of output array. Defaults to ``None``.
                If ``None``, `output_shape` is determined by evaluating
                `self.__call__` on an input array of zeros.
            eval_fn: Function used in evaluating this :class:`.Operator`.
                Defaults to ``None``. Required unless `__init__` is being
                called from a derived class with an `_eval` method.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`. If the :class:`.Operator`
                implements complex-valued operations, this must be a
                complex dtype (typically :attr:`~numpy.complex64`) for
                correct adjoint and gradient calculation.
            output_dtype: `dtype` for output argument. Defaults to
                ``None``. If ``None``, `output_dtype` is determined by
                evaluating `self.__call__` on an input array of zeros.
            jit: If ``True``, call :meth:`Operator.jit()` on this
                :class:`.Operator` to jit the forward, adjoint, and gram
                functions. Same as calling :meth:`Operator.jit` after the
                :class:`.Operator` is created.

        Raises:
            NotImplementedError: If the `eval_fn` parameter is not
               specified and the `_eval` method is not defined in a
               derived class.
        """

        #: Shape of input array or :class:`.BlockArray`.
        self.input_shape: Union[Shape, BlockShape]

        #: Size of flattened input. Sum of product of `input_shape` tuples.
        self.input_size: int

        #: Shape of output array or :class:`.BlockArray`
        self.output_shape: Union[Shape, BlockShape]

        #: Size of flattened output. Sum of product of `output_shape` tuples.
        self.output_size: int

        #: Shape Operator would take if it operated on flattened arrays.
        #: Consists of (output_size, input_size)
        self.matrix_shape: Tuple[int, int]

        #: Shape of Operator, consisting of (output_shape, input_shape).
        self.shape: Tuple[Union[Shape, BlockShape], Union[Shape, BlockShape]]

        #: Dtype of input
        self.input_dtype: DType

        #: Dtype of operator
        self.dtype: DType

        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        else:
            self.input_shape = input_shape
        self.input_dtype = input_dtype

        # Allows for dynamic creation of new Operator/LinearOperator, e.g. for adjoints
        if eval_fn:
            self._eval = eval_fn  # type: ignore
        elif not hasattr(self, "_eval"):
            raise NotImplementedError(
                "Operator is an abstract base class when the eval_fn parameter is not specified."
            )

        # If the shape isn't specified by user we can infer it using by invoking the function
        if output_shape is None or output_dtype is None:
            tmp = self(snp.zeros(self.input_shape, dtype=input_dtype))
        if output_shape is None:
            self.output_shape = tmp.shape  # type: ignore
        else:
            self.output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

        if output_dtype is None:
            self.output_dtype = tmp.dtype
        else:
            self.output_dtype = output_dtype

        # Determine the shape of the "vectorized" operator (as an element of ℝ^{n × m}
        # If the function returns a BlockArray we need to compute the size of each block,
        # then sum.
        self.input_size = shape_to_size(self.input_shape)
        self.output_size = shape_to_size(self.output_shape)

        self.shape = (self.output_shape, self.input_shape)
        self.matrix_shape = (self.output_size, self.input_size)

        if jit:
            self.jit()

    def jit(self):
        """Activate just-in-time compilation for the `_eval` method."""
        self._eval = jax.jit(self._eval)

    def __call__(self, x: Union[Operator, Array, BlockArray]) -> Union[Operator, Array, BlockArray]:
        r"""Evaluate this :class:`Operator` at the point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate this :class:`.Operator`. If `x`
               is a :class:`jax.Array` or :class:`.BlockArray`, it must
               have `shape == self.input_shape`. If `x` is a
               :class:`.Operator` or :class:`.LinearOperator`, it must
               have `x.output_shape == self.input_shape`.

        Returns:
             :class:`.Operator` evaluated at `x`.

        Raises:
            ValueError: If the `input_shape` attribute of the
                :class:`.Operator` is not equal to the input array shape,
                or to the `output_shape` attribute of another
                :class:`.Operator` with which it is composed.
        """

        if isinstance(x, Operator):
            # Compose the two operators if shapes conform
            if self.input_shape == x.output_shape:
                return Operator(
                    input_shape=x.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda z: self(x(z)),
                    input_dtype=self.input_dtype,
                    output_dtype=x.output_dtype,
                )
            raise ValueError(f"Incompatible shapes {self.shape}, {x.shape}.")

        if self.input_shape != x.shape:
            raise ValueError(
                f"Cannot evaluate {type(self)} with input_shape={self.input_shape} "
                f"on array with shape={x.shape}."
            )

        return self._eval(x)

    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            if self.shape == other.shape:
                return Operator(
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda x: self(x) + other(x),
                    input_dtype=self.input_dtype,
                    output_dtype=result_type(self.output_dtype, other.output_dtype),
                )
            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")
        raise TypeError(f"Operation __add__ not defined between {type(self)} and {type(other)}.")

    def __sub__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            if self.shape == other.shape:
                return Operator(
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda x: self(x) - other(x),
                    input_dtype=self.input_dtype,
                    output_dtype=result_type(self.output_dtype, other.output_dtype),
                )
            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")
        raise TypeError(f"Operation __sub__ not defined between {type(self)} and {type(other)}.")

    @_wrap_mul_div_scalar
    def __mul__(self, other):
        return Operator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    def __neg__(self) -> Operator:
        return -1.0 * self

    @_wrap_mul_div_scalar
    def __rmul__(self, other):
        return Operator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, other):
        return Operator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) / other,
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    def jvp(self, u, v):
        r"""Compute a Jacobian-vector product.

        Compute the product :math:`J_F(\mb{u}) \mb{v}` where :math:`F`
        represents this operator and :math:`J_F(\mb{u})` is the Jacobian
        of :math:`F` evaluated at :math:`\mb{u}`. This method is
        implemented via a call to :func:`jax.jvp`.

        Args:
            u: Value at which the Jacobian is evaluated.
            v: Vector in the Jacobian-vector product.

        Returns:
           A pair :math:`(F(\mb{u}), J_F(\mb{u}) \mb{v})`, i.e. a pair
           consisting of the operator evaluated at :math:`\mb{u}` and the
           Jacobian-vector product.
        """
        return jax.jvp(self, (u,), (v,))

    def vjp(self, u, conjugate=True):
        r"""Compute a vector-Jacobian product.

        Compute the product :math:`[J_F(\mb{u})]^T \mb{v}` where :math:`F`
        represents this operator and :math:`J_F(\mb{u})` is the Jacobian
        of :math:`F` evaluated at :math:`\mb{u}`. Instead of directly
        computing the vector-Jacobian product, this method returns a
        function, taking :math:`\mb{v}` as an argument, that returns
        the product. This method is implemented via a call to
        :func:`jax.vjp`.

        Args:
            u: Value at which the Jacobian is evaluated.
            conjugate: If ``True``, compute the product using the
               conjugate (Hermitian) transpose.

        Returns:
            A pair :math:`(F(\mb{u}), G(\cdot))` where :math:`G(\cdot)`
            is a function that computes the vector-Jacobian product, i.e.
            :math:`G(\mb{v}) = [J_F(\mb{u})]^T \mb{v}` when `conjugate`
            is ``False``, or :math:`G(\mb{v}) = [J_F(\mb{u})]^H \mb{v}`
            when `conjugate` is ``True``.
        """
        Fu, G = jax.vjp(self, u)

        if conjugate:

            def Gmap(v):
                return G(v.conj())[0].conj()

        else:

            def Gmap(v):
                return G(v)[0]

        return Fu, Gmap

    def freeze(self, argnum: int, val: Union[Array, BlockArray]) -> Operator:
        """Return a new :class:`.Operator` with fixed block argument.

        Return a new :class:`.Operator` with block argument `argnum`
        fixed to value `val`.

        Args:
            argnum: Index of block to freeze. Must be less than or equal
               to the number of blocks in an input array.
            val: Value to fix the `argnum`-th input to.

        Returns:
            A new :class:`.Operator` with one of the blocks of the input
            fixed to the specified value.

        Raises:
            ValueError: If the :class:`.Operator` does not take a
               :class:`.BlockArray` as its input, if the block index
               equals or exceeds the number of blocks, or if the shape of
               the fixed value differs from the shape of the specified
               block.
        """

        if not is_nested(self.input_shape):
            raise ValueError(
                "The freeze method can only be applied to Operators that take BlockArray inputs."
            )

        input_ndim = len(self.input_shape)
        if argnum > input_ndim - 1:
            raise ValueError(
                f"Parameter argnum to freeze must be less than the number of input arguments to "
                f"this operator ({input_ndim}); got {argnum}."
            )

        if val.shape != self.input_shape[argnum]:
            raise ValueError(
                f"Value to be frozen at position {argnum} must have shape "
                f"{self.input_shape[argnum]}, got {val.shape}."
            )

        input_shape: Union[Shape, BlockShape]
        input_shape = tuple(s for i, s in enumerate(self.input_shape) if i != argnum)  # type: ignore

        if len(input_shape) == 1:
            input_shape = input_shape[0]  # type: ignore

        def concat_args(args):
            # Create a blockarray with args and the frozen value in the correct place
            # E.g. if this operator takes a blockarray with two blocks, then
            # concat_args(args) = snp.blockarray([val, args]) if argnum = 0
            # concat_args(args) = snp.blockarray([args, val]) if argnum = 1

            if isinstance(args, (jnp.ndarray, np.ndarray)):
                # In the case that the original operator takes a blockarray with two
                # blocks, wrap in a list so we can use the same indexing as >2 block case
                args = [args]

            arg_list = []
            for i in range(input_ndim):
                if i < argnum:
                    arg_list.append(args[i])
                elif i > argnum:
                    arg_list.append(args[i - 1])
                else:
                    arg_list.append(val)
            return snp.blockarray(arg_list)

        return Operator(
            input_shape=input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(concat_args(x)),
        )
