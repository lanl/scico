# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Operator and LinearOperator base class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import operator
from functools import partial, wraps
from typing import Callable, Optional, Tuple, Union

import numpy as np

import jax
from jax.dtypes import result_type
from jax.interpreters.xla import DeviceArray

import scico.numpy as snp
from scico._autograd import linear_adjoint
from scico.array import is_complex_dtype, is_nested
from scico.blockarray import BlockArray, block_sizes
from scico.typing import BlockShape, DType, JaxArray, Shape


def _wrap_mul_div_scalar(func):
    r"""Wrapper function for defining mul, rmul, and truediv.

    Wrapper function for defining mul, rmul, and truediv between a scalar
    and an Operator.

    If one of these binary operations are called in the form
    binop(Operator, other) and 'b' is a scalar, specialized
    Operator constructors can be called.

    Args:
        func: should be either .__mul__(), .__rmul__(),
           or .__truediv__().

    Raises:
        TypeError: A binop with the form binop(Operator, other) is
        called and other is not a scalar.
    """

    @wraps(func)
    def wrapper(a, b):
        if np.isscalar(b) or isinstance(b, jax.core.Tracer):
            return func(a, b)

        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}")

    return wrapper


class Operator:
    """Generic Operator class."""

    def __repr__(self):
        return f"""{type(self)}
shape       : {self.shape}
matrix_shape : {self.matrix_shape}
input_dtype : {self.input_dtype}
output_dtype : {self.output_dtype}
        """

    # See https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.beyond-basics.html#ndarray.__array_priority__
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
            output_shape: Shape of output array.
                Defaults to ``None``. If ``None``, `output_shape` is
                determined by evaluating `self.__call__` on an input
                array of zeros.
            eval_fn: Function used in evaluating this Operator.
                Defaults to ``None``. If ``None``, then `self.__call__`
                must be defined in any derived classes.
            input_dtype: `dtype` for input argument.
                Defaults to `float32`. If Operator implements
                complex-valued operations, this must be `complex64` for
                proper adjoint and gradient calculation.
            output_dtype: `dtype` for output argument.
                Defaults to ``None``. If ``None``, `output_shape` is
                determined by evaluating `self.__call__` on an input
                array of zeros.
            jit: If ``True``, call :meth:`Operator.jit()` on this
                Operator to jit the forward, adjoint, and gram functions.
                Same as calling :meth:`Operator.jit` after the Operator
                is created.
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

        #: Shape of Operator. Consists of (output_shape, input_shape).
        self.shape: Tuple[Union[Shape, BlockShape], Union[Shape, BlockShape]]

        #: Dtype of input
        self.input_dtype: DType

        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        else:
            self.input_shape = input_shape
        self.input_dtype = input_dtype

        # Allows for dynamic creation of new Operator/LinearOperator, e.g. for adjoints
        if eval_fn:
            self._eval = eval_fn  # type: ignore

        # If the shape isn't specified by user we can infer it using by invoking the function
        if output_shape is None or output_dtype is None:
            tmp = self(snp.zeros(self.input_shape, dtype=input_dtype))
        if output_shape is None:
            self.output_shape = tmp.shape
        else:
            self.output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

        if output_dtype is None:
            self.output_dtype = tmp.dtype
        else:
            self.output_dtype = output_dtype

        # Determine the shape of the "vectorized" operator (as an element of ℝ^{n × m}
        # If the function returns a BlockArray we need to compute the size of each block,
        # then sum.
        self.input_size = int(np.sum(block_sizes(self.input_shape)))
        self.output_size = int(np.sum(block_sizes(self.output_shape)))

        self.shape = (self.output_shape, self.input_shape)
        self.matrix_shape = (self.output_size, self.input_size)

        if jit:
            self.jit()

    def jit(self):
        """Activate just-in-time compilation for the `_eval` method."""
        self._eval = jax.jit(self._eval)

    def __call__(
        self, x: Union[Operator, JaxArray, BlockArray]
    ) -> Union[Operator, JaxArray, BlockArray]:
        r"""Evaluate this Operator at the point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate this Operator. If `x` is a
               :class:`DeviceArray` or :class:`.BlockArray`, must have
               `shape == self.input_shape`. If `x` is a
               :class:`.Operator` or :class:`.LinearOperator`, must have
               `x.output_shape == self.input_shape`.
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
            raise ValueError(f"""Incompatible shapes {self.shape}, {x.shape} """)

        if isinstance(x, (np.ndarray, DeviceArray, BlockArray)):
            if self.input_shape == x.shape:
                return self._eval(x)
            raise ValueError(
                f"Cannot evaluate {type(self)} with input_shape={self.input_shape} "
                f"on array with shape={x.shape}"
            )
        # What is the context under which this gets called?
        # Currently:  in jit and grad tracers
        return self._eval(x)

    def __add__(self, other):
        if isinstance(other, Operator):
            if self.shape == other.shape:
                return Operator(
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda x: self(x) + other(x),
                    input_dtype=self.input_dtype,
                    output_dtype=result_type(self.output_dtype, other.output_dtype),
                )
            raise ValueError(f"shapes {self.shape} and {other.shape} do not match")
        raise TypeError(f"Operation __add__ not defined between {type(self)} and {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Operator):
            if self.shape == other.shape:
                return Operator(
                    input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda x: self(x) - other(x),
                    input_dtype=self.input_dtype,
                    output_dtype=result_type(self.output_dtype, other.output_dtype),
                )
            raise ValueError(f"shapes {self.shape} and {other.shape} do not match")
        raise TypeError(f"Operation __sub__ not defined between {type(self)} and {type(other)}")

    @_wrap_mul_div_scalar
    def __mul__(self, other):
        return Operator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    def __neg__(self):
        # -self = -1. * self
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

    def jvp(self, primals, tangents):
        """Computes a Jacobian-vector product.

        Args:
            primals:  Values at which the Jacobian is evaluated.
            tangents:  Vector in the Jacobian-vector product.
        """

        return jax.jvp(self, primals, tangents)

    def vjp(self, *primals):
        """Compute a vector-Jacobian product.

        Args:
            primals: Sequence of values at which the Jacobian is
               evaluated, with length equal to the number of position
               arguments of `_eval`.
        """

        primals, self_vjp = jax.vjp(self, *primals)
        return primals, self_vjp

    def freeze(self, argnum: int, val: Union[JaxArray, BlockArray]) -> Operator:
        """Return a new Operator with fixed block argument `argnum`.

        Return a new Operator with block argument `argnum` fixed to value
        `val`.

        Args:
            argnum: Index of block to freeze. Must be less than or equal
               to the number of blocks in an input array.
            val: Value to fix the `argnum`-th input to.
        """

        if not is_nested(self.input_shape):
            raise ValueError(
                "The `freeze` method can only be applied to Operators that take BlockArray inputs"
            )

        input_ndim = len(self.input_shape)
        if argnum > input_ndim - 1:
            raise ValueError(
                f"argnum to freeze must be less than the number of input arguments to "
                f"this operator ({input_ndim}); got {argnum}"
            )

        if val.shape != self.input_shape[argnum]:
            raise ValueError(
                f"value to be frozen at position {argnum} must have shape "
                f"{self.input_shape[argnum]}, got {val.shape}"
            )

        input_shape = tuple(s for i, s in enumerate(self.input_shape) if i != argnum)

        if len(input_shape) == 1:
            input_shape = input_shape[0]

        def concat_args(args):
            # Creates a blockarray with args and the frozen value in the correct place
            # Eg if this operator takes a blockarray with two blocks, then
            # concat_args(args) = BlockArray.array([val, args]) if argnum = 0
            # concat_args(args) = BlockArray.array([args, val]) if argnum = 1

            if isinstance(args, (DeviceArray, np.ndarray)):
                # In the case that the original operator takes a blcokarray with two
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
            return BlockArray.array(arg_list)

        return Operator(
            input_shape=input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(concat_args(x)),
        )


def _wrap_add_sub(func: Callable, op: Callable) -> Callable:
    r"""Wrapper function for defining __add__, __sub__.

    Wrapper function for defining __add__, __sub__ between LinearOperator
    and other objects.

    Handles shape checking and dispatching based on operand types:
    - If one of the two operands is an Operator, an Operator is returned.
    - If both operands are LinearOperators of different types, a generic
      LinearOperator is returned.
    - If both operands are LinearOperators of the same type, a special
      constructor can be called

    Args:
        func: should be either .__add__() or .__sub__().
        op: functional equivalent of func, ex. op.add for func =
           __add__.

    Raises:
        ValueError: The shape of both operators does not match.
        TypeError: One of the two operands is not an Operator
        or LinearOperator.

    """

    @wraps(func)
    def wrapper(
        a: LinearOperator, b: Union[Operator, LinearOperator]
    ) -> Union[Operator, LinearOperator]:
        if isinstance(b, Operator):
            if a.shape == b.shape:
                if isinstance(b, type(a)):
                    # same type of linop, eg convolution can have special
                    # behavior (see Conv2d.__add__)
                    return func(a, b)
                if isinstance(
                    b, LinearOperator
                ):  # LinearOperator + LinearOperator -> LinearOperator
                    return LinearOperator(
                        input_shape=a.input_shape,
                        output_shape=a.output_shape,
                        eval_fn=lambda x: op(a(x), b(x)),
                        adj_fn=lambda x: op(a(x), b(x)),
                        input_dtype=a.input_dtype,
                        output_dtype=result_type(a.output_dtype, b.output_dtype),
                    )
                # LinearOperator + Operator -> Operator
                return Operator(
                    input_shape=a.input_shape,
                    output_shape=a.output_shape,
                    eval_fn=lambda x: op(a(x), b(x)),
                    input_dtype=a.input_dtype,
                    output_dtype=result_type(a.output_dtype, b.output_dtype),
                )
            raise ValueError(f"shapes {a.shape} and {b.shape} do not match")
        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}")

    return wrapper


class LinearOperator(Operator):
    """Generic Linear Operator base class"""

    def __init__(
        self,
        input_shape: Union[Shape, BlockShape],
        output_shape: Optional[Union[Shape, BlockShape]] = None,
        eval_fn: Optional[Callable] = None,
        adj_fn: Optional[Callable] = None,
        input_dtype: DType = np.float32,
        output_dtype: Optional[DType] = None,
        jit: bool = False,
    ):
        r"""LinearOperator init method.

        Args:
            input_shape: Shape of input array.
            output_shape: Shape of output array.
                Defaults to ``None``. If ``None``, ``output_shape`` is
                determined by evaluating ``self.__call__`` on an input
                array of zeros.
            eval_fn: Function used in evaluating this LinearOperator.
                Defaults to ``None``. If ``None``, then ``self.__call__``
                must be defined in any derived classes.
            adj_fn: Function used to evaluate the adjoint of this
                LinearOperator. Defaults to ``None``. If ``None``, the
                adjoint is not set, and the :meth:`._set_adjoint`
                will be called silently at the first :meth:`.adj` call or
                can be called manually.
            input_dtype: `dtype` for input argument.
                Defaults to `float32`. If ``LinearOperator`` implements
                complex-valued operations, this must be `complex64` for
                proper adjoint and gradient calculation.
            output_dtype: `dtype` for output argument.
                Defaults to ``None``. If ``None``, ``output_shape`` is
                determined by evaluating ``self.__call__`` on an input
                array of zeros.
            jit: If ``True``, call :meth:`.jit()` on this LinearOperator
                to jit the forward, adjoint, and gram functions. Same as
                calling :meth:`.jit` after the LinearOperator is created.
        """

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            eval_fn=eval_fn,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=False,
        )

        if not hasattr(self, "_adj"):
            self._adj = None
        if not hasattr(self, "_gram"):
            self._gram = None
        if callable(adj_fn):
            self._adj = adj_fn
            self._gram = lambda x: self.adj(self(x))
        elif adj_fn is not None:
            raise TypeError(f"Parameter adj_fn must be either a Callable or None; got {adj_fn}")

        if jit:
            self.jit()

    def _set_adjoint(self):
        adj_fun = linear_adjoint(self.__call__, snp.zeros(self.input_shape, dtype=self.input_dtype))
        self._adj = lambda x: adj_fun(x)[0]
        self._gram = lambda x: self.adj(self(x))

    def jit(self):
        """Replaces the private functions :meth:`._eval`, :meth:`_adj`, :meth:`._gram`
        with jitted versions.
        """
        if (self._adj is None) or (self._gram is None):
            self._set_adjoint()

        self._eval = jax.jit(self._eval)
        self._adj = jax.jit(self._adj)
        self._gram = jax.jit(self._gram)

    @partial(_wrap_add_sub, op=operator.add)
    def __add__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) + other(x),
            adj_fn=lambda x: self.adj(x) + other.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other.output_dtype),
        )

    @partial(_wrap_add_sub, op=operator.sub)
    def __sub__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) - other(x),
            adj_fn=lambda x: self.adj(x) - other.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other.output_dtype),
        )

    @_wrap_mul_div_scalar
    def __mul__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            adj_fn=lambda x: snp.conj(other) * self.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    @_wrap_mul_div_scalar
    def __rmul__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            adj_fn=lambda x: snp.conj(other) * self.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) / other,
            adj_fn=lambda x: self.adj(x) / snp.conj(other),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    def __matmul__(self, other):
        # self @ other
        return self(other)

    def __rmatmul__(self, other):
        # other @ self
        if isinstance(other, LinearOperator):
            return other(self)

        if isinstance(other, (np.ndarray, DeviceArray)):
            # for real valued inputs: y @ self == (self.T @ y.T).T
            # for complex:  y @ self == (self.conj().T @ y.conj().T).conj().T
            # self.conj().T == self.adj
            return self.adj(other.conj().T).conj().T

        raise NotImplementedError(
            f"Operation __rmatmul__ not defined between {type(self)} and {type(other)}"
        )

    def __call__(
        self, x: Union[LinearOperator, JaxArray, BlockArray]
    ) -> Union[LinearOperator, JaxArray, BlockArray]:
        r"""Evaluate this LinearOperator at the point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate this ``LinearOperator``. If
               ``x`` is a :class:`DeviceArray` or :class:`.BlockArray`,
               must have ``shape == self.input_shape``. If ``x`` is a
               :class:`.LinearOperator`, must have
               ``x.output_shape == self.input_shape``.
        """
        if isinstance(x, LinearOperator):
            return ComposedLinearOperator(self, x)
        # Use Operator __call__ for LinearOperator @ array or LinearOperator @ Operator
        return super().__call__(x)

    def adj(
        self, y: Union[LinearOperator, JaxArray, BlockArray]
    ) -> Union[LinearOperator, JaxArray, BlockArray]:
        """Adjoint of this :class:`.LinearOperator`.

        Compute the adjoint of this :class:`.LinearOperator` applied to
        input ``y``.

        Args:
            y:  Point at which to compute adjoint. If `y` is
                :class:`DeviceArray` or :class:`.BlockArray`, must have
                ``shape == self.output_shape``. If `y` is a
                :class:`.LinearOperator`, must have
                ``y.output_shape == self.output_shape``.

        Returns:
            Result of adjoint evaluated at ``y``.
        """
        if self._adj is None:
            self._set_adjoint()

        if isinstance(y, LinearOperator):
            return ComposedLinearOperator(self.H, y)
        if self.output_dtype != y.dtype:
            raise ValueError(f"dtype error: expected {self.output_dtype}, got {y.dtype}")
        if self.output_shape != y.shape:
            raise ValueError(
                f"""Shapes do not conform: input array with shape {y.shape} does not match
                LinearOperator output_shape {self.output_shape}"""
            )
        return self._adj(y)

    @property
    def T(self) -> LinearOperator:
        """Transpose of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` that implements the
        transpose of this :class:`LinearOperator`. For a real-valued
        LinearOperator ``A`` (``A.input_dtype=np.float32` or
        ``np.float64``), the LinearOperator ``A.T`` implements the
        adjoint:  ``A.T(y) == A.adj(y)``. For a complex-valued
        LinearOperator ``A`` (``A.input_dtype``=`np.complex64` or
        ``np.complex128``), the LinearOperator ``A.T`` is not the
        adjoint. For the conjugate transpose, use ``.conj().T`` or
        :meth:`.H`.
        """
        if is_complex_dtype(self.input_dtype):
            return LinearOperator(
                input_shape=self.output_shape,
                output_shape=self.input_shape,
                eval_fn=lambda x: self.adj(x.conj()).conj(),
                adj_fn=self.__call__,
                input_dtype=self.input_dtype,
                output_dtype=self.output_dtype,
            )
        return LinearOperator(
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            eval_fn=self.adj,
            adj_fn=self.__call__,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
        )

    @property
    def H(self) -> LinearOperator:
        """Hermitian transpose of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` that is the Hermitian
        transpose of this :class:`LinearOperator`. For a real-valued
        LinearOperator ``A`` (``A.input_dtype=np.float32`` or
        ``np.float64``), the LinearOperator ``A.H`` is equivalent to
        ``A.T``. For a complex-valued LinearOperator ``A``
        (``A.input_dtype = np.complex64`` or ``np.complex128``), the
        LinearOperator ``A.H`` implements the adjoint of
        ``A : A.H @ y == A.adj(y) == A.conj().T @ y)``.

        For the non-conjugate transpose, see :meth:`.T`.
        """
        return LinearOperator(
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            eval_fn=self.adj,
            adj_fn=self.__call__,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
        )

    def conj(self) -> LinearOperator:
        """Complex conjugate of this :class:`LinearOperator`.

        Return a new :class:`.LinearOperator` ``Ac`` such that
        ``Ac(x) = conj(A)(x)``.
        """
        # A.conj() x == (A @ x.conj()).conj()
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x.conj()).conj(),
            adj_fn=lambda x: self.adj(x.conj()).conj(),
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )

    @property
    def gram_op(self) -> LinearOperator:
        """Gram operator of this :class:`LinearOperator`.

        Return a new :class:`.LinearOperator` ``G`` such that
        ``G(x) = A.adj(A(x)))``.
        """
        if self._gram is None:
            self._set_adjoint()

        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.input_shape,
            eval_fn=self.gram,
            adj_fn=self.gram,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )

    def gram(
        self, x: Union[LinearOperator, JaxArray, BlockArray]
    ) -> Union[LinearOperator, JaxArray, BlockArray]:
        """Compute ``A.adj(A(x)).``

        Args:
            x: Point at which to evaluate the gram operator. If ``x`` is
               a :class:`DeviceArray` or :class:`.BlockArray`, must have
               ``shape == self.input_shape``. If ``x`` is a
               :class:`.LinearOperator`, must have
               ``x.output_shape == self.input_shape``.

        Returns:
            Result of ``A.adj(A(x))``.
        """
        if self._gram is None:
            self._set_adjoint()
        return self._gram(x)


class ComposedLinearOperator(LinearOperator):
    """A LinearOperator formed by the composition of two LinearOperators."""

    def __init__(self, A: LinearOperator, B: LinearOperator, jit: bool = False):
        r"""ComposedLinearOperator init method.

        A ComposedLinearOperator ``AB`` implements ``AB @ x == A @ B @ x``.
        The LinearOperators ``A`` and ``B`` are stored as attributes of
        the ComposedLinearOperator.

        The LinearOperators ``A`` and ``B`` must have compatible shapes
        and dtypes: ``A.input_shape == B.output_shape`` and
        ``A.input_dtype == B.input_dtype``.

        Args:
            A: First (left) LinearOperator.
            B: Second (right) LinearOperator.
            jit: If ``True``, call :meth:`.jit()` on this LinearOperator
                to jit the forward, adjoint, and gram functions. Same as
                calling :meth:`.jit` after the LinearOperator is created.
        """
        if not isinstance(A, LinearOperator):
            raise TypeError(
                "The first argument to ComposedLinearOpeator must be a LinearOperator; "
                f"got {type(A)}"
            )
        if not isinstance(B, LinearOperator):
            raise TypeError(
                "The second argument to ComposedLinearOpeator must be a LinearOperator; "
                f"got {type(B)}"
            )
        if A.input_shape != B.output_shape:
            raise ValueError(f"Incompatable LinearOperator shapes {A.shape}, {B.shape}")
        if A.input_dtype != B.output_dtype:
            raise ValueError(
                f"Incompatable LinearOperator dtypes {A.input_dtype}, {B.output_dtype}"
            )

        self.A = A
        self.B = B

        super().__init__(
            input_shape=self.B.input_shape,
            output_shape=self.A.output_shape,
            input_dtype=self.B.input_dtype,
            output_dtype=self.A.output_dtype,
            eval_fn=lambda x: self.A(self.B(x)),
            adj_fn=lambda z: self.B.adj(self.A.adj(z)),
            jit=jit,
        )
