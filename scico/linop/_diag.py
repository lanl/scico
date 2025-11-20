# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Miscellaneous linear operator definitions."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Union

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.util import broadcast_nested_shapes, is_nested
from scico.operator._operator import _wrap_mul_div_scalar
from scico.typing import BlockShape, DType, Shape

from ._linop import LinearOperator, _wrap_add_sub

__all__ = ["Diagonal", "Identity", "ScaledIdentity"]


class Diagonal(LinearOperator):
    """Diagonal linear operator."""

    def __init__(
        self,
        diagonal: Union[Array, BlockArray],
        input_shape: Optional[Union[Shape, BlockShape]] = None,
        input_dtype: Optional[DType] = None,
        **kwargs,
    ):
        r"""
        Args:
            diagonal: Diagonal elements of this :class:`LinearOperator`.
            input_shape: Shape of input array. By default, equal to
               `diagonal.shape`, but may also be set to a shape that is
               broadcast-compatible with `diagonal.shape`.
            input_dtype: `dtype` of input argument. The default,
               ``None``, means `diagonal.dtype`.
        """
        self._diagonal = diagonal

        if input_shape is None:
            input_shape = self._diagonal.shape

        if input_dtype is None:
            input_dtype = self._diagonal.dtype

        if isinstance(diagonal, BlockArray) and is_nested(input_shape):
            output_shape = broadcast_nested_shapes(input_shape, self._diagonal.shape)
        elif not isinstance(diagonal, BlockArray) and not is_nested(input_shape):
            output_shape = snp.broadcast_shapes(input_shape, self._diagonal.shape)
        elif isinstance(diagonal, BlockArray):
            raise ValueError("Argument 'diagonal' was a BlockArray but input_shape was not nested.")
        else:
            raise ValueError("Argument 'diagonal' was not a BlockArray but input_shape was nested.")

        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_shape=output_shape,
            output_dtype=input_dtype,
            **kwargs,
        )

    def _eval(self, x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        return self._diagonal * x

    def _norm(self):
        return snp.abs(self.diagonal).max()

    @property
    def diagonal(self) -> Union[Array, BlockArray]:
        """Return an array representing the diagonal component."""
        return self._diagonal

    @property
    def T(self) -> Diagonal:
        """Transpose of this :class:`Diagonal`."""
        return self

    def conj(self) -> Diagonal:
        """Complex conjugate of this :class:`Diagonal`."""
        return Diagonal(diagonal=self.diagonal.conj())

    @property
    def H(self) -> Diagonal:
        """Hermitian transpose of this :class:`Diagonal`."""
        return self.conj()

    @property
    def gram_op(self) -> Diagonal:
        """Gram operator of this :class:`Diagonal`.

        Return a new :class:`Diagonal` :code:`G` such that
        :code:`G(x) = A.adj(A(x)))`.
        """
        return Diagonal(diagonal=self.diagonal.conj() * self.diagonal)

    @_wrap_add_sub
    def __add__(self, other):
        if self.diagonal.shape == other.diagonal.shape:
            return Diagonal(diagonal=self.diagonal + other.diagonal)
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_add_sub
    def __sub__(self, other):
        if self.diagonal.shape == other.diagonal.shape:
            return Diagonal(diagonal=self.diagonal - other.diagonal)
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return Diagonal(diagonal=self.diagonal * scalar)

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return Diagonal(diagonal=self.diagonal / scalar)

    def __matmul__(self, other):
        # self @ other
        if isinstance(other, Diagonal):
            if self.shape == other.shape:
                return Diagonal(diagonal=self.diagonal * other.diagonal)
            raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")
        else:
            return self(other)

    def norm(self, ord=None):  # pylint: disable=W0622
        """Compute the matrix norm of the diagonal operator.

        Valid values of `ord` and the corresponding norm definition
        are those listed under "norm for matrices" in the
        :func:`scico.numpy.linalg.norm` documentation.
        """
        ordfunc = {
            "fro": lambda x: snp.linalg.norm(x),
            "nuc": lambda x: snp.sum(snp.abs(x)),
            -snp.inf: lambda x: snp.abs(x).min(),
            snp.inf: lambda x: snp.abs(x).max(),
        }
        mord = ord
        if mord is None:
            mord = "fro"
        elif mord in (-1, -2):
            mord = -snp.inf
        elif mord in (1, 2):
            mord = snp.inf
        if mord not in ordfunc:
            raise ValueError(f"Invalid value {ord} for argument 'ord'.")
        return ordfunc[mord](self._diagonal)


class ScaledIdentity(Diagonal):
    """Scaled identity operator."""

    def __init__(
        self,
        scalar: float,
        input_shape: Union[Shape, BlockShape],
        input_dtype: DType = snp.float32,
        **kwargs,
    ):
        """
        Args:
            scalar: Scaling of the identity.
            input_shape: Shape of input array.
            input_dtype: `dtype` of input argument.
        """
        if is_nested(input_shape):
            diagonal = scalar * snp.ones(((),) * len(input_shape), dtype=input_dtype)
        else:
            diagonal = scalar * snp.ones((), dtype=input_dtype)
        super().__init__(
            diagonal=diagonal,
            input_shape=input_shape,
            input_dtype=input_dtype,
            **kwargs,
        )

    @property
    def diagonal(self) -> Union[Array, BlockArray]:
        return self._diagonal * snp.ones(self.input_shape, dtype=self.input_dtype)

    def conj(self) -> ScaledIdentity:
        """Complex conjugate of this :class:`ScaledIdentity`."""
        return ScaledIdentity(
            scalar=self._diagonal.conj(), input_shape=self.input_shape, input_dtype=self.input_dtype
        )

    @property
    def gram_op(self) -> ScaledIdentity:
        """Gram operator of this :class:`ScaledIdentity`."""
        return ScaledIdentity(
            scalar=self._diagonal * self._diagonal.conj(),
            input_shape=self.input_shape,
            input_dtype=self.input_dtype,
        )

    @_wrap_add_sub
    def __add__(self, other):
        if self.input_shape == other.input_shape:
            return ScaledIdentity(
                scalar=self._diagonal + other._diagonal,
                input_shape=self.input_shape,
                input_dtype=self.input_dtype,
            )
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_add_sub
    def __sub__(self, other):
        if self.input_shape == other.input_shape:
            return ScaledIdentity(
                scalar=self._diagonal - other._diagonal,
                input_shape=self.input_shape,
                input_dtype=self.input_dtype,
            )
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return ScaledIdentity(
            scalar=self._diagonal * scalar,
            input_shape=self.input_shape,
            input_dtype=self.input_dtype,
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return ScaledIdentity(
            scalar=self._diagonal / scalar,
            input_shape=self.input_shape,
            input_dtype=self.input_dtype,
        )

    def __matmul__(self, other):
        # self @ other
        if isinstance(other, Diagonal):
            if self.shape != other.shape:
                raise ValueError(f"Shapes {self.shape} and {other.shape} do not match.")
            if isinstance(other, ScaledIdentity):
                return ScaledIdentity(
                    scalar=self._diagonal * other._diagonal,
                    input_shape=self.input_shape,
                    input_dtype=self.input_dtype,
                )
            else:
                return Diagonal(diagonal=self._diagonal * other.diagonal)
        else:
            return self(other)

    def norm(self, ord=None):  # pylint: disable=W0622
        """Compute the matrix norm of the identity operator.

        Valid values of `ord` and the corresponding norm definition
        are those listed under "norm for matrices" in the
        :func:`scico.numpy.linalg.norm` documentation.
        """
        N = self.input_size
        if ord is None or ord == "fro":
            return snp.abs(self._diagonal) * snp.sqrt(N)
        elif ord == "nuc":
            return snp.abs(self._diagonal) * N
        elif ord in (-snp.inf, -1, -2, 1, 2, snp.inf):
            return snp.abs(self._diagonal)
        else:
            raise ValueError(f"Invalid value {ord} for argument 'ord'.")


class Identity(ScaledIdentity):
    """Identity operator."""

    def __init__(
        self, input_shape: Union[Shape, BlockShape], input_dtype: DType = snp.float32, **kwargs
    ):
        """
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` of input argument.
        """
        super().__init__(
            scalar=1.0,
            input_shape=input_shape,
            input_dtype=input_dtype,
            **kwargs,
        )

    def _eval(self, x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        return x

    @property
    def diagonal(self) -> Union[Array, BlockArray]:
        return snp.ones(self.input_shape, dtype=self.input_dtype)

    def conj(self) -> Identity:
        """Complex conjugate of this :class:`Diagonal`."""
        return self

    @property
    def gram_op(self) -> Identity:
        """Gram operator of this :class:`Identity`."""
        return self

    def __matmul__(self, other):
        return other

    def __rmatmul__(self, x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        return x
