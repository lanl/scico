# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator utility functions."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Union

import scico.numpy as snp
from scico.operator._operator import Operator
from scico.random import randn
from scico.typing import PRNGKey

from ._linop import LinearOperator


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
        normAv = snp.linalg.norm(Av)
        if normAv == 0.0:  # Assume that ||Av|| == 0 implies A is a zero operator
            mu = 0.0
            v = Av
            break
        mu = snp.sum(v.conj() * Av) / snp.linalg.norm(v) ** 2
        v = Av / normAv
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
    return snp.sqrt(power_iteration(A.H @ A, maxiter, key)[0].real)


def valid_adjoint(
    A: LinearOperator,
    AT: LinearOperator,
    eps: Optional[float] = 1e-7,
    x: Optional[snp.Array] = None,
    y: Optional[snp.Array] = None,
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
            raise ValueError("Shape of 'x' array not appropriate as an input for operator 'A'.")
    if y is None:
        y, key = randn(shape=AT.input_shape, key=key, dtype=AT.input_dtype)
    else:
        if y.shape != AT.input_shape:
            raise ValueError("Shape of 'y' array not appropriate as an input for operator AT.")

    u = A(x)
    v = AT(y)
    yTu = snp.sum(y.conj() * u)  # type: ignore
    vTx = snp.sum(v.conj() * x)  # type: ignore
    err = snp.abs(yTu - vTx) / max(snp.abs(yTu), snp.abs(vTx))
    if eps is None:
        return err
    return float(err) < eps


def jacobian(F: Operator, u: snp.Array, include_eval: Optional[bool] = False) -> LinearOperator:
    """Construct Jacobian linear operator for a general operator.

    For a specified :class:`.Operator`, construct a corresponding
    Jacobian :class:`LinearOperator`, the application of which is
    equivalent to multiplication by the Jacobian of the
    :class:`.Operator` at a specified input value.

    The implementation of this function is based on :meth:`.Operator.jvp`
    and :meth:`.Operator.vjp`, which are themselves based on
    :func:`jax.jvp` and :func:`jax.vjp`. For reasons of computational
    efficiency, these functions return the value of the :class:`.Operator`
    evaluated at the specified point in addition to the requested
    Jacobian-vector product. If the `include_eval` parameter of this
    function is ``True``, the constructed :class:`LinearOperator` returns
    a :class:`.BlockArray` output, the first component of which is the
    result of the :class:`.Operator` evaluation, and the second component
    of which is the requested Jacobian-vector product. If `include_eval`
    is ``False``, then the :class:`.Operator` evaluation computed by
    :func:`jax.jvp` and :func:`jax.vjp` are discarded.

    Args:
        F: :class:`.Operator` of which the Jacobian is to be computed.
        u: Input value of the :class:`.Operator` at which the Jacobian is
           to be computed.
        include_eval: Flag indicating whether the result of evaluating
           the :class:`.Operator` should be included (as the first
           component of a :class:`.BlockArray`) in the output of the
           Jacobian :class:`LinearOperator` constructed by this function.

    Returns:
      A :class:`LinearOperator` capable of computing Jacobian-vector
      products.
    """
    if include_eval:
        Fu, G = F.vjp(u, conjugate=True)

        def adj_fn(v):
            return snp.blockarray((Fu, G(v)))

        def eval_fn(v):
            return snp.blockarray(F.jvp(u, v))

    else:
        adj_fn = F.vjp(u, conjugate=True)[1]

        def eval_fn(v):
            return F.jvp(u, v)[1]

    return LinearOperator(
        F.input_shape,
        output_shape=F.output_shape,
        eval_fn=eval_fn,
        adj_fn=adj_fn,
        input_dtype=F.input_dtype,
        output_dtype=F.output_dtype,
    )
