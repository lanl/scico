# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator base class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import operator
from functools import partial, wraps
from typing import Any, Callable, Optional, Union

import numpy as np

import jax
from jax.dtypes import result_type
from jax.interpreters.xla import DeviceArray

import scico.numpy as snp
from scico._autograd import linear_adjoint
from scico.numpy import BlockArray
from scico.numpy.util import (
    ensure_on_device,
    indexed_shape,
    is_complex_dtype,
    is_nested,
)
from scico.operator._operator import Operator, _wrap_mul_div_scalar
from scico.random import randn
from scico.typing import ArrayIndex, BlockShape, DType, JaxArray, PRNGKey, Shape


def power_iteration(A: LinearOperator, maxiter: int = 100, key: Optional[PRNGKey] = None):
    """Compute largest eigenvalue of a diagonalizable :class:`.LinearOperator`.

    Compute largest eigenvalue of a diagonalizable
    :class:`.LinearOperator` using power iteration.

    Args:
        A: :class:`.LinearOperator` used for computation. Must be
            diagonalizable.
        maxiter: Maximum number of power iterations to use.
        key: Jax PRNG key. Defaults to ``None``, in which case a new key
            is created.

    Returns:
        tuple: A tuple (`mu`, `v`) containing:

            - **mu**: Estimate of largest eigenvalue of `A`.
            - **v**: Eigenvector of `A` with eigenvalue `mu`.

    """
    v, key = randn(shape=A.input_shape, key=key, dtype=A.input_dtype)
    v = v / snp.linalg.norm(v)

    for i in range(maxiter):
        Av = A @ v
        mu = snp.sum(v.conj() * Av) / snp.linalg.norm(v) ** 2
        v = Av / snp.linalg.norm(Av)
    return mu, v


def operator_norm(A: LinearOperator, maxiter: int = 100, key: Optional[PRNGKey] = None):
    r"""Estimate the norm of a :class:`.LinearOperator`.

    Estimate the operator norm
    `induced <https://en.wikipedia.org/wiki/Matrix_norm#Matrix_norms_induced_by_vector_norms>`_
    by the :math:`\ell_2` vector norm, i.e. for :class:`.LinearOperator`
    :math:`A`,

    .. math::
       \| A \|_2 &= \max \{ \| A \mb{x} \|_2 \, : \, \| \mb{x} \|_2 \leq 1 \} \\
                 &= \sqrt{ \lambda_{ \mathrm{max} }( A^H A ) }
                 = \sigma_{\mathrm{max}}(A) \;,

    where :math:`\lambda_{\mathrm{max}}(B)` and
    :math:`\sigma_{\mathrm{max}}(B)` respectively denote the
    largest eigenvalue of :math:`B` and the largest singular value of
    :math:`B`. The value is estimated via power iteration, using
    :func:`power_iteration`, to estimate
    :math:`\lambda_{\mathrm{max}}(A^H A)`.

    Args:
        A: :class:`.LinearOperator` for which operator norm is desired.
        maxiter: Maximum number of power iterations to use. Default: 100
        key: Jax PRNG key. Defaults to ``None``, in which case a new key
            is created.

    Returns:
        float: Norm of operator :math:`A`.

    """
    return snp.sqrt(power_iteration(A.H @ A, maxiter, key)[0])


def valid_adjoint(
    A: LinearOperator,
    AT: LinearOperator,
    eps: Optional[float] = 1e-7,
    x: Optional[JaxArray] = None,
    y: Optional[JaxArray] = None,
    key: Optional[PRNGKey] = None,
) -> Union[bool, float]:
    r"""Check whether :class:`.LinearOperator` `AT` is the adjoint of `A`.

    Check whether :class:`.LinearOperator` :math:`\mathsf{AT}` is the
    adjoint of :math:`\mathsf{A}`. The test exploits the identity

    .. math::
      \mathbf{y}^T (A \mathbf{x}) = (\mathbf{y}^T A) \mathbf{x} =
      (A^T \mathbf{y})^T \mathbf{x}

    by computing :math:`\mathbf{u} = \mathsf{A}(\mathbf{x})` and
    :math:`\mathbf{v} = \mathsf{AT}(\mathbf{y})` for random
    :math:`\mathbf{x}` and :math:`\mathbf{y}` and confirming that

    .. math::
      \frac{| \mathbf{y}^T \mathbf{u} - \mathbf{v}^T \mathbf{x} |}
      {\max \left\{ | \mathbf{y}^T \mathbf{u} |,
       | \mathbf{v}^T \mathbf{x} | \right\}}
      < \epsilon \;.

    If :math:`\mathsf{A}` is a complex operator (with a complex
    `input_dtype`) then the test checks whether :math:`\mathsf{AT}` is
    the Hermitian conjugate of :math:`\mathsf{A}`, with a test as above,
    but with all the :math:`(\cdot)^T` replaced with :math:`(\cdot)^H`.

    Args:
        A: Primary :class:`.LinearOperator`.
        AT: Adjoint :class:`.LinearOperator`.
        eps: Error threshold for validation of :math:`\mathsf{AT}` as
           adjoint of :math:`\mathsf{AT}`. If ``None``, the relative
           error is returned instead of a boolean value.
        x: If not the default ``None``, use the specified array instead
           of a random array as test vector :math:`\mb{x}`. If specified,
           the array must have shape `A.input_shape`.
        y: If not the default ``None``, use the specified array instead
           of a random array as test vector :math:`\mb{y}`. If specified,
           the array must have shape `AT.input_shape`.
        key: Jax PRNG key. Defaults to ``None``, in which case a new key
           is created.

    Returns:
      Boolean value indicating whether validation passed, or relative
      error of test, depending on type of parameter `eps`.
    """

    if x is None:
        x, key = randn(shape=A.input_shape, key=key, dtype=A.input_dtype)
    else:
        if x.shape != A.input_shape:
            raise ValueError("Shape of x array not appropriate as an input for operator A")
    if y is None:
        y, key = randn(shape=AT.input_shape, key=key, dtype=AT.input_dtype)
    else:
        if y.shape != AT.input_shape:
            raise ValueError("Shape of y array not appropriate as an input for operator AT")

    u = A(x)
    v = AT(y)
    yTu = snp.dot(y.ravel().conj(), u.ravel())  # type: ignore
    vTx = snp.dot(v.ravel().conj(), x.ravel())  # type: ignore
    err = snp.abs(yTu - vTx) / max(snp.abs(yTu), snp.abs(vTx))
    if eps is None:
        return err
    return err < eps


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
        r"""
        Args:
            input_shape: Shape of input array.
            output_shape: Shape of output array.
                Defaults to ``None``. If ``None``, `output_shape` is
                determined by evaluating `self.__call__` on an input
                array of zeros.
            eval_fn: Function used in evaluating this LinearOperator.
                Defaults to ``None``. If ``None``, then `self.__call__`
                must be defined in any derived classes.
            adj_fn: Function used to evaluate the adjoint of this
                LinearOperator. Defaults to ``None``. If ``None``, the
                adjoint is not set, and the :meth:`._set_adjoint`
                will be called silently at the first :meth:`.adj` call or
                can be called manually.
            input_dtype: `dtype` for input argument.
                Defaults to ``float32``. If `LinearOperator` implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            output_dtype: `dtype` for output argument.
                Defaults to ``None``. If ``None``, `output_shape` is
                determined by evaluating `self.__call__` on an input
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
            self._adj: Optional[Callable] = None
        if not hasattr(self, "_gram"):
            self._gram: Optional[Callable] = None
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
            x: Point at which to evaluate this `LinearOperator`. If
               `x` is a :class:`DeviceArray` or :class:`.BlockArray`,
               must have `shape == self.input_shape`. If `x` is a
               :class:`.LinearOperator`, must have
               `x.output_shape == self.input_shape`.
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
        input `y`.

        Args:
            y: Point at which to compute adjoint. If `y` is
                :class:`DeviceArray` or :class:`.BlockArray`, must have
                `shape == self.output_shape`. If `y` is a
                :class:`.LinearOperator`, must have
                `y.output_shape == self.output_shape`.

        Returns:
            Result of adjoint evaluated at `y`.
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
        assert self._adj is not None
        return self._adj(y)

    @property
    def T(self) -> LinearOperator:
        """Transpose of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` that implements the
        transpose of this :class:`LinearOperator`. For a real-valued
        LinearOperator `A` (`A.input_dtype` is ``np.float32`` or
        ``np.float64``), the LinearOperator `A.T` implements the
        adjoint: `A.T(y) == A.adj(y)`. For a complex-valued
        LinearOperator `A` (`A.input_dtype` is ``np.complex64`` or
        ``np.complex128``), the LinearOperator `A.T` is not the
        adjoint. For the conjugate transpose, use `.conj().T` or
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
        LinearOperator `A` (`A.input_dtype` is ``np.float32`` or
        ``np.float64``), the LinearOperator `A.H` is equivalent to
        `A.T`. For a complex-valued LinearOperator `A`
        (`A.input_dtype` is ``np.complex64`` or ``np.complex128``), the
        LinearOperator `A.H` implements the adjoint of
        `A : A.H @ y == A.adj(y) == A.conj().T @ y)`.

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

        Return a new :class:`.LinearOperator` `Ac` such that
        `Ac(x) = conj(A)(x)`.
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

        Return a new :class:`.LinearOperator` `G` such that
        `G(x) = A.adj(A(x)))`.
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
        """Compute `A.adj(A(x)).`

        Args:
            x: Point at which to evaluate the gram operator. If `x` is
               a :class:`DeviceArray` or :class:`.BlockArray`, must have
               `shape == self.input_shape`. If `x` is a
               :class:`.LinearOperator`, must have
               `x.output_shape == self.input_shape`.

        Returns:
            Result of `A.adj(A(x))`.
        """
        if self._gram is None:
            self._set_adjoint()
        assert self._gram is not None
        return self._gram(x)


class ComposedLinearOperator(LinearOperator):
    """A LinearOperator formed by the composition of two LinearOperators."""

    def __init__(self, A: LinearOperator, B: LinearOperator, jit: bool = False):
        r"""
        A ComposedLinearOperator `AB` implements `AB @ x == A @ B @ x`.
        The LinearOperators `A` and `B` are stored as attributes of
        the ComposedLinearOperator.

        The LinearOperators `A` and `B` must have compatible shapes
        and dtypes: `A.input_shape == B.output_shape` and
        `A.input_dtype == B.input_dtype`.

        Args:
            A: First (left) LinearOperator.
            B: Second (right) LinearOperator.
            jit: If ``True``, call :meth:`.jit()` on this LinearOperator
                to jit the forward, adjoint, and gram functions. Same as
                calling :meth:`.jit` after the LinearOperator is created.
        """
        if not isinstance(A, LinearOperator):
            raise TypeError(
                "The first argument to ComposedLinearOperator must be a LinearOperator; "
                f"got {type(A)}"
            )
        if not isinstance(B, LinearOperator):
            raise TypeError(
                "The second argument to ComposedLinearOperator must be a LinearOperator; "
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
            diagonal: Diagonal elements of this linear operator.
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
    """Identity operator"""

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
                Defaults to ``float32``. If this LinearOperator implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            jit: If ``True``, jit the evaluation, adjoint, and gram
               functions of the LinearOperator.
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


def linop_from_function(f: Callable, classname: str, f_name: Optional[str] = None):
    """Make a linear operator class from a function.

    Example
    -------
    >>> Sum = linop_from_function(snp.sum, 'Sum')
    >>> H = Sum((2, 10), axis=1)
    >>> H @ snp.ones((2, 10))
    DeviceArray([10., 10.], dtype=float32)

    Args:
        f: Function from which to create a linear operator class.
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
                Defaults to ``float32``. If `LinearOperator` implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            jit: If ``True``, call :meth:`.Operator.jit` on this
                `LinearOperator` to jit the forward, adjoint, and gram
                functions. Same as calling :meth:`.Operator.jit` after
                the `LinearOperator` is created.
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
