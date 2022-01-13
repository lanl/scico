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
from functools import partial
from typing import Optional, Tuple, Union

import scico.numpy as snp
from scico import array, blockarray
from scico._generic_operators import LinearOperator, _wrap_add_sub, _wrap_mul_div_scalar
from scico.blockarray import BlockArray
from scico.random import randn
from scico.typing import ArrayIndex, BlockShape, DType, JaxArray, PRNGKey, Shape

__author__ = """\n""".join(
    ["Luke Pfister <luke.pfister@gmail.com>", "Brendt Wohlberg <brendt@ieee.org>"]
)


def power_iteration(A: LinearOperator, maxiter: int = 100, key: Optional[PRNGKey] = None):
    """Compute largest eigenvalue of a diagonalizable :class:`.LinearOperator`.

    Compute largest eigenvalue of a diagonalizable
    :class:`.LinearOperator` using power iteration.

    Args:
        A: :class:`.LinearOperator` used for computation. Must be
            diagonalizable.
        maxiter: Maximum number of power iterations to use.
        key: Jax PRNG key. Defaults to None, in which case a new key is
            created.

    Returns:
        tuple: A tuple (`mu`, `v`) containing:

            - **mu**: Estimate of largest eigenvalue of `A`.
            - **v**: Eigenvector of `A` with eigenvalue `mu`.

    """
    v, key = randn(shape=A.input_shape, key=key, dtype=A.input_dtype)
    v = v / snp.linalg.norm(v)

    for i in range(maxiter):
        Av = A @ v
        mu = snp.vdot(v, Av.ravel()) / snp.linalg.norm(v) ** 2
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
        key: Jax PRNG key. Defaults to None, in which case a new key is
            created.

    Returns:
        float : Norm of operator :math:`A`.

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
           adjoint of :math:`\mathsf{AT}`. If None, the relative error
           is returned instead of a boolean value.
        x : If not the default None, use the specified array instead of a
           random array as test vector :math:`\mb{x}`. If specified, the
           array must have shape ``A.input_shape``.
        y : If not the default None, use the specified array instead of a
           random array as test vector :math:`\mb{y}`. If specified, the
           array must have shape ``AT.input_shape``.
        key: Jax PRNG key. Defaults to None, in which case a new key is
           created.

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
    yTu = snp.dot(y.ravel().conj(), u.ravel())
    vTx = snp.dot(v.ravel().conj(), x.ravel())
    err = snp.abs(yTu - vTx) / max(snp.abs(yTu), snp.abs(vTx))
    if eps is None:
        return err
    return err < eps


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

        self.diagonal = array.ensure_on_device(diagonal)

        if input_shape is None:
            input_shape = self.diagonal.shape

        if input_dtype is None:
            input_dtype = self.diagonal.dtype

        if isinstance(diagonal, BlockArray) and array.is_nested(input_shape):
            output_shape = (snp.empty(input_shape) * diagonal).shape
        elif not isinstance(diagonal, BlockArray) and not array.is_nested(input_shape):
            output_shape = snp.broadcast_shapes(input_shape, self.diagonal.shape)
        elif isinstance(diagonal, BlockArray):
            raise ValueError(f"`diagonal` was a BlockArray but `input_shape` was not nested.")
        else:
            raise ValueError(f"`diagonal` was a not BlockArray but `input_shape` was nested.")

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


class Sum(LinearOperator):
    """A linear operator for summing along an axis or set of axes."""

    def __init__(
        self,
        sum_axis: Optional[Union[int, Tuple[int, ...]]],
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Wraps :func:`jax.numpy.sum` as a :class:`.LinearOperator`.

        Args:
            sum_axis: The axis or set of axes to sum over. If ``None``,
                sum is taken over all axes.
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument.
                Defaults to ``float32``. If this LinearOperator implements
                complex-valued operations, this must be ``complex64`` for
                proper adjoint and gradient calculation.
            jit: If ``True``, jit the evaluation, adjoint, and gram
               functions of the LinearOperator.
        """

        input_ndim = len(input_shape)
        sum_axis = array.parse_axes(sum_axis, shape=input_shape)

        self.sum_axis: Tuple[int, ...] = sum_axis
        super().__init__(input_shape=input_shape, input_dtype=input_dtype, jit=jit, **kwargs)

    def _eval(self, x: JaxArray) -> JaxArray:
        return snp.sum(x, axis=self.sum_axis)


class Slice(LinearOperator):
    """A linear operator for slicing an array."""

    def __init__(
        self,
        idx: ArrayIndex,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        This operator may be applied to either a :any:`JaxArray` or a
        :class:`.BlockArray`. In the latter case, parameter ``idx`` must
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

        if array.is_nested(input_shape):
            output_shape = blockarray.indexed_shape(input_shape, idx)
        else:
            output_shape = array.indexed_shape(input_shape, idx)

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
