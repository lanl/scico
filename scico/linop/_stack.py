# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of linear operators classes."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.operator._stack import DiagonalStack as DStack
from scico.operator._stack import VerticalStack as VStack

from ._linop import LinearOperator


class VerticalStack(VStack, LinearOperator):
    r"""A vertical stack of linear operators.

    Given linear operators :math:`A_1, A_2, \dots, A_N`, create the
    linear operator

    .. math::
       H =
       \begin{pmatrix}
            A_1 \\
            A_2 \\
            \vdots \\
            A_N \\
       \end{pmatrix} \qquad
       \text{such that} \qquad
       H \mb{x}
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
        ops: Sequence[LinearOperator],
        collapse_output: Optional[bool] = True,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            ops: Linear operators to stack.
            collapse_output: If ``True`` and the output would be a
                :class:`BlockArray` with shape ((m, n, ...), (m, n, ...),
                ...), the output is instead a :class:`jax.Array` with
                shape (S, m, n, ...) where S is the length of `ops`.
            jit: See `jit` in :class:`LinearOperator`.
        """
        if not all(isinstance(op, LinearOperator) for op in ops):
            raise TypeError("All elements of ops must be of type LinearOperator.")

        super().__init__(ops=ops, collapse_output=collapse_output, jit=jit, **kwargs)

    def _adj(self, y: Union[Array, BlockArray]) -> Array:  # type: ignore
        return sum([op.adj(y_block) for y_block, op in zip(y, self.ops)])  # type: ignore


class DiagonalStack(DStack, LinearOperator):
    r"""A diagonal stack of linear operators.

    Given linear operators :math:`A_1, A_2, \dots, A_N`, create the
    linear operator

    .. math::
       H =
       \begin{pmatrix}
            A_1 & 0   & \ldots & 0\\
            0   & A_2 & \ldots & 0\\
            \vdots & \vdots & \ddots & \vdots\\
            0   & 0 & \ldots & A_N \\
       \end{pmatrix} \qquad
       \text{such that} \qquad
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
        ops: Sequence[LinearOperator],
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
            jit: See `jit` in :class:`LinearOperator`.

        """
        if not all(isinstance(op, LinearOperator) for op in ops):
            raise TypeError("All elements of ops must be of type LinearOperator.")

        super().__init__(
            ops=ops,
            collapse_input=collapse_input,
            collapse_output=collapse_output,
            jit=jit,
            **kwargs,
        )

    def _adj(self, y: Union[Array, BlockArray]) -> Union[Array, BlockArray]:  # type: ignore
        result = tuple(op.T @ y_n for op, y_n in zip(self.ops, y))  # type: ignore
        if self.collapse_input:
            return snp.stack(result)
        return snp.blockarray(result)
