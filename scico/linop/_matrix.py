# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
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

import jax
from jax.dtypes import result_type
from jax.interpreters.xla import DeviceArray

import scico.numpy as snp
from scico._generic_operators import LinearOperator
from scico.typing import JaxArray

from ._linop import Identity

__author__ = """Luke Pfister <luke.pfister@gmail.com>"""


def _wrap_add_sub_matrix(func, op):
    @wraps(func)
    def wrapper(a, b):
        if np.isscalar(b):
            return MatrixOperator(op(a.A, b))

        if isinstance(b, MatrixOperator):
            if a.shape == b.shape:
                return MatrixOperator(op(a.A, b.A))

            raise ValueError(f"MatrixOperator shapes {a.shape} and {b.shape} do not match")

        if isinstance(b, (DeviceArray, np.ndarray)):
            if a.matrix_shape == b.shape:
                return MatrixOperator(op(a.A, b))

            raise ValueError(f"Shapes {a.matrix_shape} and {b.shape} do not match")

        if isinstance(b, LinearOperator):
            if a.shape == b.shape:
                return LinearOperator(
                    input_shape=a.input_shape,
                    output_shape=a.output_shape,
                    eval_fn=lambda x: op(a(x), b(x)),
                    input_dtype=a.input_dtype,
                    output_dtype=result_type(a.output_dtype, b.output_dtype),
                )

            raise ValueError(f"Shapes {a.shape} and {b.shape} do not match")

        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}")

    return wrapper


class MatrixOperator(LinearOperator):
    """Linear operator implementing matrix multiplication."""

    def __init__(self, A: JaxArray):
        """
        Args:
            A: Dense array. The action of the created LinearOperator will
                implement matrix multiplication with `A`.
        """
        self.A: JaxArray  #: Dense array implementing this matrix

        # if A is an ndarray, make sure it gets converted to a DeviceArray
        if isinstance(A, DeviceArray):
            self.A = A
        elif isinstance(A, np.ndarray):
            self.A = jax.device_put(A)
        else:
            raise TypeError(f"Expected np.ndarray or DeviceArray, got {type(A)}")

        # Can only do rank-2 arrays
        if A.ndim != 2:
            raise TypeError(f"Expected a 2-dimensional array, got array of shape {A.shape}")

        super().__init__(input_shape=A.shape[1], output_shape=A.shape[0], input_dtype=self.A.dtype)

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
                f"{other.output_shape} does not match {self.input_shape}"
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

    # Could write another wrapper for mul, truediv, and rtuediv, bu there is
    # no operator.__rtruediv__;  have to write that case out manually anyway.
    def __mul__(self, other):
        if np.isscalar(other):
            return MatrixOperator(other * self.A)

        if isinstance(other, MatrixOperator):
            if self.shape == other.shape:
                return MatrixOperator(self.A * other.A)

            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match")

        if isinstance(other, (DeviceArray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(self.A * other)

            raise ValueError(f"Shapes {self.matrix_shape} and {other.shape} do not match")

        # includes generic LinearOperator
        raise TypeError(f"Operation __mul__ not defined between {type(self)} and {type(other)}")

    def __rmul__(self, other):
        # Multiplication is commutative
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            return MatrixOperator(self.A / other)

        if isinstance(other, MatrixOperator):
            if self.shape == other.shape:
                return MatrixOperator(self.A / other.A)
            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match")

        if isinstance(other, (DeviceArray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(self.A / other)

            raise ValueError(f"Shapes {self.matrix_shape} and {other.shape} do not match")

        raise TypeError(f"Operation __truediv__ not defined between {type(self)} and {type(other)}")

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return MatrixOperator(other / self.A)

        if isinstance(other, (DeviceArray, np.ndarray)):
            if self.matrix_shape == other.shape:
                return MatrixOperator(other / self.A)

            raise ValueError(f"Shapes {other.shape} and {self.matrix_shape} do not match")

        raise TypeError(f"Operation __truediv__ not defined between {type(other)} and {type(self)}")

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
        """Return a :class:`numpy.ndarray` containing ``self.A``."""
        return self.A.copy()

    @property
    def gram_op(self):
        """Gram operator of this :class:`.MatrixOperator`.

        Return a new :class:`.LinearOperator` ``G`` such that
        ``G(x) = A.adj(A(x)))``."""
        return MatrixOperator(A=self.A.conj().T @ self.A)

    def norm(self, ord=None, axis=None, keepdims=False):  # pylint: disable=W0622
        """Compute the norm of the dense matrix `self.A`.

        Call :func:`scico.numpy.norm` on the dense matrix `self.A`.
        """
        return snp.linalg.norm(self.A, ord=ord, axis=axis, keepdims=keepdims)
