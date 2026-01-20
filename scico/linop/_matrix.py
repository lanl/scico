# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Matrix linear operator classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import operator
from functools import partial, wraps

import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike

import scico.numpy as snp
from scico.operator._operator import Operator

from ._diag import Identity
from ._linop import LinearOperator


def _wrap_add_sub_matrix(func, op):
    @wraps(func)
    def wrapper(a, b):
        if np.isscalar(b):
            return MatrixOperator(op(a.A, b))

        if isinstance(b, MatrixOperator):
            if a.shape == b.shape:
                return MatrixOperator(op(a.A, b.A))

            raise ValueError(f"MatrixOperator shapes {a.shape} and {b.shape} do not match.")

        if isinstance(b, (jnp.ndarray, np.ndarray)):
            if a.matrix_shape == b.shape:
                return MatrixOperator(op(a.A, b))

            raise ValueError(f"Shapes {a.matrix_shape} and {b.shape} do not match.")

        if isinstance(b, Operator):
            if a.shape != b.shape:
                raise ValueError(f"Shapes {a.shape} and {b.shape} do not match.")

        if isinstance(b, LinearOperator):
            uwfunc = getattr(LinearOperator, func.__name__)._unwrapped
            return uwfunc(a, b)

        if isinstance(b, Operator):
            uwfunc = getattr(Operator, func.__name__)
            return uwfunc(a, b)

        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}.")

    return wrapper


class MatrixOperator(LinearOperator):
    """Linear operator implementing matrix multiplication."""

    def __init__(self, A: ArrayLike, input_cols: int = 0):
        """
        Args:
            A: Dense array. The action of the created
                :class:`.LinearOperator` will
                implement matrix multiplication with `A`.
            input_cols: If this parameter is set to the default of 0, the
                :class:`MatrixOperator` takes a vector (one-dimensional
                array) input. If the input is intended to be a matrix
                (two-dimensional array), this parameter should specify
                number of columns in the matrix.
        """
        self.A: snp.Array  #: Dense array implementing this matrix

        # Ensure that A is a numpy or jax array.
        if not snp.util.is_arraylike(A):
            raise TypeError(f"Expected numpy or jax array, got {type(A)}.")
        self.A = A

        # Can only do rank-2 arrays
        if A.ndim != 2:
            raise TypeError(f"Expected a two-dimensional array, got array of shape {A.shape}.")

        self.__array__ = A.__array__  # enables jnp.array(H)

        if input_cols == 0:
            input_shape = A.shape[1]
            output_shape = A.shape[0]
        else:
            input_shape = (A.shape[1], input_cols)
            output_shape = (A.shape[0], input_cols)

        super().__init__(
            input_shape=input_shape, output_shape=output_shape, input_dtype=self.A.dtype
        )

    def __call__(self, other):
        if isinstance(other, LinearOperator):
            if self.input_shape == other.output_shape:
                if isinstance(other, Identity):
                    return self

                if isinstance(other, MatrixOperator):
                    return MatrixOperator(A=self.A @ other.A)

                # must be a generic linop so return composition of the two
                return LinearOperator(
                    input_shape=other.input_shape,
                    output_shape=self.output_shape,
                    eval_fn=lambda x: self(other(x)),
                    input_dtype=self.input_dtype,
                )

            raise ValueError(
                "Cannot compute MatrixOperator-LinearOperator product, "
                f"{other.output_shape} does not match {self.input_shape}."
            )

        return self._eval(other)

    def _eval(self, other):
        return self.A @ other

    def gram(self, other):
        return self.A.conj().T @ self.A @ other

    @partial(_wrap_add_sub_matrix, op=operator.add)
    def __add__(self, other):
        pass

    @partial(_wrap_add_sub_matrix, op=operator.sub)
    def __sub__(self, other):
        pass

    def __radd__(self, other):
        # Addition is commutative
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return MatrixOperator(-self.A)

    # Could write another wrapper for mul, truediv, and rtuediv, but there is
    # no operator.__rtruediv__;  have to write that case out manually anyway.
    def __mul__(self, other):
        if np.isscalar(other):
            return MatrixOperator(other * self.A)

        if isinstance(other, MatrixOperator):
            if self.shape == other.shape:
                return MatrixOperator(self.A * other.A)

            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")

        if isinstance(other, (jnp.ndarray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(self.A * other)

            raise ValueError(f"Shapes {self.matrix_shape} and {other.shape} do not match.")

        # includes generic LinearOperator
        raise TypeError(f"Operation __mul__ not defined between {type(self)} and {type(other)}.")

    def __rmul__(self, other):
        # multiplication is commutative
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            return MatrixOperator(self.A / other)

        if isinstance(other, MatrixOperator):
            if self.shape == other.shape:
                return MatrixOperator(self.A / other.A)
            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")

        if isinstance(other, (jnp.ndarray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(self.A / other)

            raise ValueError(f"Shapes {self.matrix_shape} and {other.shape} do not match.")

        raise TypeError(
            f"Operation __truediv__ not defined between {type(self)} and {type(other)}."
        )

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return MatrixOperator(other / self.A)

        if isinstance(other, (jnp.ndarray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(other / self.A)

            raise ValueError(f"Shapes {other.shape} and {self.matrix_shape} do not match.")

        raise TypeError(
            f"Operation __truediv__ not defined between {type(other)} and {type(self)}."
        )

    def __getitem__(self, key):
        return self.A[key]

    @property
    def T(self):
        """Transpose of this :class:`.MatrixOperator`.

        Return a :class:`.MatrixOperator` corresponding to the transpose
        of this matrix.
        """
        return MatrixOperator(self.A.T)

    @property
    def H(self):
        """Hermitian (conjugate) transpose of this :class:`.MatrixOperator`.

        Return a :class:`.MatrixOperator` corresponding to the Hermitian
        (conjugate) transpose of this matrix.
        """
        return MatrixOperator(self.A.conj().T)

    def conj(self):
        """Complex conjugate of this :class:`.MatrixOperator`.

        Return a :class:`.MatrixOperator` with complex conjugated
        elements.
        """
        return MatrixOperator(A=self.A.conj())

    def adj(self, y):
        return self.A.conj().T @ y

    def to_array(self):
        """Return a :class:`numpy.ndarray` containing `self.A`."""
        return np.array(self.A)

    @property
    def gram_op(self):
        """Gram operator of this :class:`.MatrixOperator`.

        Return a new :class:`.LinearOperator` `G` such that
        `G(x) = A.adj(A(x)))`."""
        return MatrixOperator(A=self.A.conj().T @ self.A)

    def norm(self, ord=None, axis=None, keepdims=False):  # pylint: disable=W0622
        """Compute the norm of the dense matrix `self.A`.

        Call :func:`scico.numpy.linalg.norm` on the dense matrix `self.A`.
        """
        return snp.linalg.norm(self.A, ord=ord, axis=axis, keepdims=keepdims)
