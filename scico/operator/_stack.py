# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Stack of operators classes."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

import jax

import scico.numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_blockable, is_collapsible, is_nested
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


class DiagonalReplicated(Operator):
    r"""A diagonal stack constructed from a single operator.

    Given operator :math:`A`, create the operator :math:`H` such that

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
            A(\mb{x}_1) \\
            A(\mb{x}_2) \\
            \vdots \\
            A(\mb{x}_N) \\
       \end{pmatrix} \;.

    The application of :math:`A` to each component :math:`\mb{x}_k` is
    computed using :func:`jax.pmap` or :func:`jax.vmap`. The input shape
    for operator :math:`A` should exclude the array axis on which
    :math:`A` is replicated to form :math:`H`. For example, if :math:`A`
    has input shape `(3, 4)` and :math:`H` is constructed to replicate
    on axis 0 with 2 replicates, the input shape of :math:`H` will be
    `(2, 3, 4)`.

    Operators taking :class:`.BlockArray` input are not supported.
    """

    def __init__(
        self,
        op: Operator,
        replicates: int,
        input_axis: int = 0,
        output_axis: Optional[int] = None,
        map_type: str = "auto",
        **kwargs,
    ):
        """
        Args:
            op: Operator to replicate.
            replicates: Number of replicates of `op`.
            input_axis: Input axis over which `op` should be replicated.
            output_axis: Index of replication axis in output array.
               If ``None``, the input replication axis is used.
            map_type: If "pmap" or "vmap", apply replicated mapping using
               :func:`jax.pmap` or :func:`jax.vmap` respectively. If
               "auto", use :func:`jax.pmap` if sufficient devices are
               available for the number of replicates, otherwise use
               :func:`jax.vmap`.
        """
        if map_type not in ["auto", "pmap", "vmap"]:
            raise ValueError("Argument 'map_type' must be one of 'auto', 'pmap, or 'vmap'.")
        if input_axis < 0:
            input_axis = len(op.input_shape) + 1 + input_axis
        if input_axis < 0 or input_axis > len(op.input_shape):
            raise ValueError(
                "Argument 'input_axis' must be positive and less than the number of axes "
                "in the input shape of argument 'op'."
            )
        if is_nested(op.input_shape):
            raise ValueError("Argument 'op' may not be an Operator taking BlockArray input.")
        if is_nested(op.output_shape):
            raise ValueError("Argument 'op' may not be an Operator with BlockArray output.")
        self.op = op
        self.replicates = replicates
        self.input_axis = input_axis
        self.output_axis = self.input_axis if output_axis is None else output_axis

        if map_type == "auto":
            self.jaxmap = jax.pmap if replicates <= jax.device_count() else jax.vmap
        else:
            if map_type == "pmap" and replicates > jax.device_count():
                raise ValueError(
                    "Requested pmap mapping but number of replicates exceeds device count."
                )
            else:
                self.jaxmap = jax.pmap if map_type == "pmap" else jax.vmap

        eval_fn = self.jaxmap(op.__call__, in_axes=self.input_axis, out_axes=self.output_axis)

        input_shape = (
            op.input_shape[0 : self.input_axis] + (replicates,) + op.input_shape[self.input_axis :]
        )
        output_shape = (
            op.output_shape[0 : self.output_axis]
            + (replicates,)
            + op.output_shape[self.output_axis :]
        )

        super().__init__(
            input_shape=input_shape,  # type: ignore
            output_shape=output_shape,  # type: ignore
            eval_fn=eval_fn,
            input_dtype=op.input_dtype,
            output_dtype=op.output_dtype,
            jit=False,
            **kwargs,
        )
