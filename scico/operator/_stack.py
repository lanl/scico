# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of operators classes."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from typing_extensions import TypeGuard

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_nested
from scico.typing import BlockShape, Shape

from ._operator import Operator, _wrap_mul_div_scalar


def collapse_shapes(
    shapes: Sequence[Union[Shape, BlockShape]], allow_collapse=True
) -> Tuple[Union[Shape, BlockShape], bool]:
    """Decides whether to collapse a sequence of shapes and returns the collapsed
    shape and a boolean indicating whether the shape was collapsed."""

    if is_collapsible(shapes) and allow_collapse:
        return (len(shapes), *shapes[0]), True

    if is_blockable(shapes):
        return shapes, False

    raise ValueError(
        "Combining these shapes would result in a twice-nested BlockArray, which is not supported."
    )


def is_collapsible(shapes: Sequence[Union[Shape, BlockShape]]) -> bool:
    """Return ``True`` if the a list of shapes represent arrays that can
    be stacked, i.e., they are all the same."""
    return all(s == shapes[0] for s in shapes)


def is_blockable(shapes: Sequence[Union[Shape, BlockShape]]) -> TypeGuard[Union[Shape, BlockShape]]:
    """Return ``True`` if the list of shapes represent arrays that can be
    combined into a :class:`BlockArray`, i.e., none are nested."""
    return not any(is_nested(s) for s in shapes)


class AbstractVerticalStack:
    r"""A vertical stack of operator-like objects."""

    def __init__(
        self,
        ops: List[Operator],
        op_type: Type[Operator] = Operator,
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            ops: Operator-like objects to stack.
            op_type: Type name of operator-like objects to stack.
            collapse_output: If ``True`` and the output would be a
                :class:`BlockArray` with shape ((m, n, ...), (m, n, ...),
                ...), the output is instead a :class:`jax.Array` with
                shape (S, m, n, ...) where S is the length of `ops`.
            jit: See `jit` in :class:`Operator`.
        """
        self.op_type = op_type
        self.check_if_stackable(ops)
        self.ops = ops
        self.collapse_output = collapse_output

        output_shapes = tuple(op.output_shape for op in ops)
        self.output_collapsible = is_collapsible(output_shapes)

        self.output_shape: Union[Shape, BlockShape]
        if self.output_collapsible and self.collapse_output:
            self.output_shape = (len(ops),) + output_shapes[0]  # collapse to jax array
        else:
            self.output_shape = output_shapes

    def check_if_stackable(self, ops: List[Operator]):
        """Check that input ops are suitable for stack creation."""
        if not isinstance(ops, (list, tuple)):
            raise ValueError(f"Expected a list of {self.op_type.__name__}.")

        input_shapes = [op.shape[1] for op in ops]
        if not all(input_shapes[0] == s for s in input_shapes):
            raise ValueError(
                f"Expected all {self.op_type.__name__}s to have the same input shapes, "
                f"but got {input_shapes}."
            )

        input_dtypes = [op.input_dtype for op in ops]
        if not all(input_dtypes[0] == s for s in input_dtypes):
            raise ValueError(
                f"Expected all {self.op_type.__name__}s to have the same input dtype, "
                f"but got {input_dtypes}."
            )

        if any([is_nested(op.shape[0]) for op in ops]):
            raise ValueError(f"Cannot stack {self.op_type.__name__}s with nested output shapes.")

        output_dtypes = [op.output_dtype for op in ops]
        if not np.all(output_dtypes[0] == s for s in output_dtypes):
            raise ValueError(
                f"Expected all {self.op_type.__name__}s to have the same output dtype."
            )

    def _eval(self, x: Array) -> Union[Array, BlockArray]:
        if self.output_collapsible and self.collapse_output:
            return snp.stack([op(x) for op in self.ops])
        return BlockArray([op(x) for op in self.ops])

    def scale_ops(self, scalars: Array):
        """Scale component operators.

        Return a copy of `self` with each operator scaled by the
        corresponding entry in `scalars`.

        Args:
            scalars: List or array of scalars to use.
        """
        if len(scalars) != len(self.ops):
            raise ValueError("Expected scalars to be the same length as self.ops.")

        return self.__class__(
            [a * op for a, op in zip(scalars, self.ops)], collapse_output=self.collapse_output
        )

    def __add__(self, other):
        return self.__class__(
            [op1 + op2 for op1, op2 in zip(self.ops, other.ops)],
            collapse_output=self.collapse_output,
        )

    def __sub__(self, other):
        return self.__class__(
            [op1 - op2 for op1, op2 in zip(self.ops, other.ops)],
            collapse_output=self.collapse_output,
        )

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return self.__class__(
            [scalar * op for op in self.ops], collapse_output=self.collapse_output
        )

    @_wrap_mul_div_scalar
    def __rmul__(self, scalar):
        return self.__class__(
            [scalar * op for op in self.ops], collapse_output=self.collapse_output
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return self.__class__(
            [op / scalar for op in self.ops], collapse_output=self.collapse_output
        )


class VerticalStack(AbstractVerticalStack, Operator):
    r"""A vertical stack of operators.

    Given operators :math:`A_1, A_2, \dots, A_N`, create the operator
    :math:`H` such that

    .. math::
       H(\mb{x})
       =
       \begin{pmatrix}
            A_1(\mb{x}) \\
            A_2(\mb{x}) \\
            \vdots \\
            A_N(\mb{x}) \\
       \end{pmatrix} \;.
    """

    def __init__(
        self,
        ops: List[Operator],
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            ops: Operators to stack.
            collapse_output: If ``True`` and the output would be a
                :class:`BlockArray` with shape ((m, n, ...), (m, n, ...),
                ...), the output is instead a :class:`jax.Array` with
                shape (S, m, n, ...) where S is the length of `ops`.
            jit: See `jit` in :class:`Operator`.
        """
        AbstractVerticalStack.__init__(
            self, ops=ops, op_type=Operator, collapse_output=collapse_output, jit=jit
        )
        Operator.__init__(
            self,
            input_shape=ops[0].input_shape,
            output_shape=self.output_shape,  # type: ignore
            input_dtype=ops[0].input_dtype,
            output_dtype=ops[0].output_dtype,
            jit=jit,
            **kwargs,
        )


class AbstractDiagonalStack:
    r"""A diagonal stack of operator-like objects."""

    def __init__(
        self,
        ops: List[Operator],
        collapse_input: Optional[bool] = True,
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            ops: Operators to form into a block matrix.
            collapse_input: If ``True``, inputs are expected to be
                stacked along the first dimension when possible.
            collapse_output: If ``True``, the output will be
                stacked along the first dimension when possible.
            jit: See `jit` in :class:`Operator`.
        """
        self.ops = ops

        self.input_shape: Union[Shape, BlockShape]
        self.input_shape, self.collapse_input = collapse_shapes(
            tuple(op.input_shape for op in ops),
            collapse_input,
        )
        self.output_shape: Union[Shape, BlockShape]
        self.output_shape, self.collapse_output = collapse_shapes(
            tuple(op.output_shape for op in ops),
            collapse_output,
        )

    def _eval(self, x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        result = tuple(op(x_n) for op, x_n in zip(self.ops, x))
        if self.collapse_output:
            return snp.stack(result)
        return snp.blockarray(result)


class DiagonalStack(AbstractDiagonalStack, Operator):
    r"""A diagonal stack of operators.

    Given operators :math:`A_1, A_2, \dots, A_N`, create the operator
    :math:`H` such that

    .. math::
       H
       \begin{pmatrix}
            \mb{x}_1 \\
            \mb{x}_2 \\
            \vdots \\
            \mb{x}_N \\
       \end{pmatrix}
       =
       \begin{pmatrix}
            A_1(\mb{x}_1) \\
            A_2(\mb{x}_2) \\
            \vdots \\
            A_N(\mb{x}_N) \\
       \end{pmatrix} \;.

    By default, if the inputs :math:`\mb{x}_1, \mb{x}_2, \dots,
    \mb{x}_N` all have the same (possibly nested) shape, `S`, this
    operator will work on the stack, i.e., have an input shape of `(N,
    *S)`. If the inputs have distinct shapes, `S1`, `S2`, ..., `SN`,
    this operator will work on the block concatenation, i.e.,
    have an input shape of `(S1, S2, ..., SN)`. The same holds for the
    output shape.
    """

    def __init__(
        self,
        ops: List[Operator],
        collapse_input: Optional[bool] = True,
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            ops: Operators to form into a block matrix.
            collapse_input: If ``True``, inputs are expected to be
                stacked along the first dimension when possible.
            collapse_output: If ``True``, the output will be
                stacked along the first dimension when possible.
            jit: See `jit` in :class:`Operator`.

        """
        self.ops = ops

        input_shape, self.collapse_input = collapse_shapes(
            tuple(op.input_shape for op in ops),
            collapse_input,
        )
        output_shape, self.collapse_output = collapse_shapes(
            tuple(op.output_shape for op in ops),
            collapse_output,
        )

        AbstractDiagonalStack.__init__(
            self, ops=ops, collapse_input=collapse_input, collapse_output=collapse_output, jit=jit
        )
        Operator.__init__(
            self,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=ops[0].input_dtype,
            output_dtype=ops[0].output_dtype,
            jit=jit,
            **kwargs,
        )
