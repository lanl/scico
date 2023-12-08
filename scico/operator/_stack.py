# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of operators classes."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from typing_extensions import TypeGuard

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_nested
from scico.typing import BlockShape, Shape

from ._operator import Operator


def collapse_shapes(
    shapes: Sequence[Union[Shape, BlockShape]], allow_collapse=True
) -> Tuple[Union[Shape, BlockShape], bool]:
    """Compute the collapsed representation of a sequence of shapes.

    Decide whether to collapse a sequence of shapes, returning either
    the sequence of shapes or a collapsed shape, and a boolean indicating
    whether the shape was collapsed."""

    if is_collapsible(shapes) and allow_collapse:
        return (len(shapes), *shapes[0]), True

    if is_blockable(shapes):
        return shapes, False

    raise ValueError(
        "Combining these shapes would result in a twice-nested BlockArray, which is not supported."
    )


def is_collapsible(shapes: Sequence[Union[Shape, BlockShape]]) -> bool:
    """Determine whether a sequence of shapes can be collapsed.

    Return ``True`` if the a list of shapes represent arrays that can
    be stacked, i.e., they are all the same."""
    return all(s == shapes[0] for s in shapes)


def is_blockable(shapes: Sequence[Union[Shape, BlockShape]]) -> TypeGuard[Union[Shape, BlockShape]]:
    """Determine whether a sequence of shapes could be a :class:`BlockArray` shape.

    Return ``True`` if the sequence of shapes represent arrays that can
    be combined into a :class:`BlockArray`, i.e., none are nested."""
    return not any(is_nested(s) for s in shapes)


class VerticalStack(Operator):
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
        ops: Sequence[Operator],
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
        VerticalStack.check_if_stackable(ops)

        self.ops = ops
        self.collapse_output = collapse_output

        output_shapes = tuple(op.output_shape for op in ops)
        self.output_collapsible = is_collapsible(output_shapes)

        if self.output_collapsible and self.collapse_output:
            output_shape = (len(ops),) + output_shapes[0]  # collapse to jax array
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
    def check_if_stackable(ops: Sequence[Operator]):
        """Check that input ops are suitable for stack creation."""
        if not isinstance(ops, (list, tuple)):
            raise TypeError("Expected a list of Operator.")

        input_shapes = [op.shape[1] for op in ops]
        if not all(input_shapes[0] == s for s in input_shapes):
            raise ValueError(
                "Expected all Operators to have the same input shapes, " f"but got {input_shapes}."
            )

        input_dtypes = [op.input_dtype for op in ops]
        if not all(input_dtypes[0] == s for s in input_dtypes):
            raise ValueError(
                "Expected all Operators to have the same input dtype, " f"but got {input_dtypes}."
            )

        if any([is_nested(op.shape[0]) for op in ops]):
            raise ValueError("Cannot stack Operators with nested output shapes.")

        output_dtypes = [op.output_dtype for op in ops]
        if not np.all(output_dtypes[0] == s for s in output_dtypes):
            raise ValueError("Expected all Operators to have the same output dtype.")

    def _eval(self, x: Array) -> Union[Array, BlockArray]:
        if self.output_collapsible and self.collapse_output:
            return snp.stack([op(x) for op in self.ops])
        return BlockArray([op(x) for op in self.ops])


class DiagonalStack(Operator):
    r"""A diagonal stack of operators.

    Given operators :math:`A_1, A_2, \dots, A_N`, create the operator
    :math:`H` such that

    .. math::
       H \left(
       \begin{pmatrix}
            \mb{x}_1 \\
            \mb{x}_2 \\
            \vdots \\
            \mb{x}_N \\
       \end{pmatrix} \right)
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
        ops: Sequence[Operator],
        collapse_input: Optional[bool] = True,
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            ops: Operators to stack.
            collapse_input: If ``True``, inputs are expected to be
                stacked along the first dimension when possible.
            collapse_output: If ``True``, the output will be
                stacked along the first dimension when possible.
            jit: See `jit` in :class:`Operator`.

        """
        DiagonalStack.check_if_stackable(ops)

        self.ops = ops

        input_shape, self.collapse_input = collapse_shapes(
            tuple(op.input_shape for op in ops),
            collapse_input,
        )
        output_shape, self.collapse_output = collapse_shapes(
            tuple(op.output_shape for op in ops),
            collapse_output,
        )

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=ops[0].input_dtype,
            output_dtype=ops[0].output_dtype,
            jit=jit,
            **kwargs,
        )

    @staticmethod
    def check_if_stackable(ops: Sequence[Operator]):
        """Check that input ops are suitable for stack creation."""
        if not isinstance(ops, (list, tuple)):
            raise TypeError("Expected a list of Operator.")

        if any([is_nested(op.shape[0]) for op in ops]):
            raise ValueError("Cannot stack Operators with nested output shapes.")

        output_dtypes = [op.output_dtype for op in ops]
        if not np.all(output_dtypes[0] == s for s in output_dtypes):
            raise ValueError("Expected all Operators to have the same output dtype.")

    def _eval(self, x: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        result = tuple(op(x_n) for op, x_n in zip(self.ops, x))
        if self.collapse_output:
            return snp.stack(result)
        return snp.blockarray(result)
