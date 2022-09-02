# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of linear operators class."""

from __future__ import annotations

import operator
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from typing_extensions import TypeGuard

import scico.numpy as snp
from scico.numpy import BlockArray
from scico.numpy.util import is_nested
from scico.typing import BlockShape, JaxArray, Shape

from ._linop import LinearOperator, _wrap_add_sub, _wrap_mul_div_scalar


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
        "Combining these shapes would result in a twice-nested BlockArray, which is not supported"
    )


def is_collapsible(shapes: Sequence[Union[Shape, BlockShape]]) -> bool:
    """Return ``True`` if the a list of shapes represent arrays that
    can be stacked, i.e., they are all the same."""
    return all(s == shapes[0] for s in shapes)


def is_blockable(shapes: Sequence[Union[Shape, BlockShape]]) -> TypeGuard[Union[Shape, BlockShape]]:
    """Return ``True`` if the list of shapes represent arrays that
    can be combined into a :class:`BlockArray`, i.e., none are nested."""
    return not any(is_nested(s) for s in shapes)


class VerticalStack(LinearOperator):
    """A vertical stack of LinearOperators."""

    def __init__(
        self,
        ops: List[LinearOperator],
        collapse: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            ops: Operators to stack.
            collapse: If ``True`` and the output would be a
                :class:`BlockArray` with shape ((m, n, ...), (m, n, ...),
                ...), the output is instead a `DeviceArray` with shape
                (S, m, n, ...) where S is the length of `ops`. Defaults
                to ``True``.
            jit: see `jit` in :class:`LinearOperator`.

        """

        VerticalStack.check_if_stackable(ops)

        self.ops = ops
        self.collapse = collapse

        output_shapes = tuple(op.output_shape for op in ops)
        self.collapsible = is_collapsible(output_shapes)

        if self.collapsible and self.collapse:
            output_shape = (len(ops),) + output_shapes[0]  # collapse to DeviceArray
        else:
            output_shape = output_shapes

        super().__init__(
            input_shape=ops[0].input_shape,
            output_shape=output_shape,  # type: ignore
            input_dtype=ops[0].input_dtype,
            output_dtype=ops[0].output_dtype,
            jit=jit,
            **kwargs,
        )

    @staticmethod
    def check_if_stackable(ops: List[LinearOperator]):
        """Check that input ops are suitable for stack creation."""
        if not isinstance(ops, (list, tuple)):
            raise ValueError("Expected a list of LinearOperator")

        input_shapes = [op.shape[1] for op in ops]
        if not all(input_shapes[0] == s for s in input_shapes):
            raise ValueError(
                "Expected all LinearOperators to have the same input shapes, "
                f"but got {input_shapes}"
            )

        input_dtypes = [op.input_dtype for op in ops]
        if not all(input_dtypes[0] == s for s in input_dtypes):
            raise ValueError(
                "Expected all LinearOperators to have the same input dtype, "
                f"but got {input_dtypes}."
            )

        if any([is_nested(op.shape[0]) for op in ops]):
            raise ValueError("Cannot stack LinearOperators with nested output shapes.")

        output_dtypes = [op.output_dtype for op in ops]
        if not np.all(output_dtypes[0] == s for s in output_dtypes):
            raise ValueError("Expected all LinearOperators to have the same output dtype.")

    def _eval(self, x: JaxArray) -> Union[JaxArray, BlockArray]:
        if self.collapsible and self.collapse:
            return snp.stack([op @ x for op in self.ops])
        return BlockArray([op @ x for op in self.ops])

    def _adj(self, y: Union[JaxArray, BlockArray]) -> JaxArray:  # type: ignore
        return sum([op.adj(y_block) for y_block, op in zip(y, self.ops)])

    def scale_ops(self, scalars: JaxArray):
        """Scale component linear operators.

        Return a copy of `self` with each operator scaled by the
        corresponding entry in `scalars`.

        Args:
            scalars: List or array of scalars to use.
        """
        if len(scalars) != len(self.ops):
            raise ValueError("expected `scalars` to be the same length as `self.ops`")

        return VerticalStack([a * op for a, op in zip(scalars, self.ops)], collapse=self.collapse)

    @partial(_wrap_add_sub, op=operator.add)
    def __add__(self, other):
        # add another VerticalStack of the same shape
        return VerticalStack(
            [op1 + op2 for op1, op2 in zip(self.ops, other.ops)], collapse=self.collapse
        )

    @partial(_wrap_add_sub, op=operator.sub)
    def __sub__(self, other):
        # subtract another VerticalStack of the same shape
        return VerticalStack(
            [op1 - op2 for op1, op2 in zip(self.ops, other.ops)], collapse=self.collapse
        )

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return VerticalStack([scalar * op for op in self.ops], collapse=self.collapse)

    @_wrap_mul_div_scalar
    def __rmul__(self, scalar):
        return VerticalStack([scalar * op for op in self.ops], collapse=self.collapse)

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return VerticalStack([op / scalar for op in self.ops], collapse=self.collapse)


class DiagonalStack(LinearOperator):
    r"""A diagonal stack of linear operators.

    Given operators :math:`A_1, A_2, \dots, A_N`, creates the operator
    :math:`H` such that

    .. math::
       \begin{pmatrix}
            A_1(\mathbf{x}_1) \\
            A_2(\mathbf{x}_2) \\
            \vdots \\
            A_N(\mathbf{x}_N) \\
       \end{pmatrix}
       = H
       \begin{pmatrix}
            \mathbf{x}_1 \\
            \mathbf{x}_2 \\
            \vdots \\
            \mathbf{x}_N \\
       \end{pmatrix} \;.

    By default, if the inputs :math:`\mathbf{x}_1, \mathbf{x}_2, \dots,
    \mathbf{x}_N` all have the same (possibly nested) shape, `S`, this
    operator will work on the stack, i.e., have an input shape of `(N,
    *S)`. If the inputs have distinct shapes, `S1`, `S2`, ..., `SN`,
    this operator will work on the block concatenation, i.e.,
    have an input shape of `(S1, S2, ..., SN)`. The same holds for the
    output shape.

    """

    def __init__(
        self,
        ops: List[LinearOperator],
        allow_input_collapse: Optional[bool] = True,
        allow_output_collapse: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            op: Operators to form into a block matrix.
            allow_input_collapse: If ``True``, inputs are expected to be
                stacked along the first dimension when possible.
            allow_output_collapse: If ``True``, the output will be
                stacked along the first dimension when possible.
            jit: see `jit` in :class:`LinearOperator`.

        """
        self.ops = ops

        input_shape, self.collapse_input = collapse_shapes(
            tuple(op.input_shape for op in ops),
            allow_input_collapse,
        )

        output_shape, self.collapse_output = collapse_shapes(
            tuple(op.output_shape for op in ops),
            allow_output_collapse,
        )

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=ops[0].input_dtype,
            output_dtype=ops[0].output_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        result = tuple(op @ x_n for op, x_n in zip(self.ops, x))
        if self.collapse_output:
            return snp.stack(result)
        return snp.blockarray(result)

    def _adj(self, y: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:  # type: ignore
        result = tuple(op.T @ y_n for op, y_n in zip(self.ops, y))
        if self.collapse_input:
            return snp.stack(result)
        return snp.blockarray(result)
