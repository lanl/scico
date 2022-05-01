# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of linear operators class."""

from __future__ import annotations

import operator
from functools import partial
from typing import List, Optional, Union

import numpy as np

import scico.numpy as snp
from scico.numpy import BlockArray
from scico.numpy.util import is_nested
from scico.typing import JaxArray

from ._linop import LinearOperator, _wrap_add_sub, _wrap_mul_div_scalar


class LinearOperatorStack(LinearOperator):
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
            collapse: If ``True`` and the output would be a `BlockArray`
                with shape ((m, n, ...), (m, n, ...), ...), the output is
                instead a `DeviceArray` with shape (S, m, n, ...) where S
                is the length of `ops`. Defaults to ``True``.
            jit: see `jit` in :class:`LinearOperator`.

        """

        LinearOperatorStack.check_if_stackable(ops)

        self.ops = ops
        self.collapse = collapse

        self.collapsable = all(op.output_shape == ops[0].output_shape for op in ops)

        output_shapes = tuple(op.output_shape for op in ops)
        if self.collapsable and self.collapse:
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
            raise ValueError("Expected a list of `LinearOperator`")

        input_shapes = [op.shape[1] for op in ops]
        if not all(input_shapes[0] == s for s in input_shapes):
            raise ValueError(
                "Expected all `LinearOperator`s to have the same input shapes, "
                f"but got {input_shapes}"
            )

        input_dtypes = [op.input_dtype for op in ops]
        if not all(input_dtypes[0] == s for s in input_dtypes):
            raise ValueError(
                "Expected all `LinearOperator`s to have the same input dtype, "
                f"but got {input_dtypes}."
            )

        if any([is_nested(op.shape[0]) for op in ops]):
            raise ValueError("Cannot stack `LinearOperator`s with nested output shapes.")

        output_dtypes = [op.output_dtype for op in ops]
        if not np.all(output_dtypes[0] == s for s in output_dtypes):
            raise ValueError("Expected all `LinearOperator`s to have the same output dtype.")

    def _eval(self, x: JaxArray) -> Union[JaxArray, BlockArray]:
        if self.collapsable and self.collapse:
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

        return LinearOperatorStack(
            [a * op for a, op in zip(scalars, self.ops)], collapse=self.collapse
        )

    @partial(_wrap_add_sub, op=operator.add)
    def __add__(self, other):
        # add another LinearOperatorStack of the same shape
        return LinearOperatorStack(
            [op1 + op2 for op1, op2 in zip(self.ops, other.ops)], collapse=self.collapse
        )

    @partial(_wrap_add_sub, op=operator.sub)
    def __sub__(self, other):
        # subtract another LinearOperatorStack of the same shape
        return LinearOperatorStack(
            [op1 - op2 for op1, op2 in zip(self.ops, other.ops)], collapse=self.collapse
        )

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return LinearOperatorStack([scalar * op for op in self.ops], collapse=self.collapse)

    @_wrap_mul_div_scalar
    def __rmul__(self, scalar):
        return LinearOperatorStack([scalar * op for op in self.ops], collapse=self.collapse)

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return LinearOperatorStack([op / scalar for op in self.ops], collapse=self.collapse)
