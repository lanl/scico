# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Bi."""


# needed to annotate a class method that returns the encapsulating class
# see https://www.python.org/dev/peps/pep-0563/


import numpy as np

from jax.scipy.signal import convolve

from scico._generic_operators import LinearOperator, Operator
from scico.blockarray import BlockArray
from scico.linop import Convolve, ConvolveByX
from scico.typing import BlockShape, DType, JaxArray
from scico.util import is_nested


class BiConvolve(Operator):
    """BiConvolution operator.

    A BiConvolve operator accepts a :class:`.BlockArray` input with two blocks
    of equal ndims, and convolves the first block with the second.

    If `A` is a BiConvolve operator, then
    A(BlockArray.array([x, h])) = jax.scipy.signal.convolve(x, h)

    """

    def __init__(
        self,
        input_shape: BlockShape,
        input_dtype: DType = np.float32,
        mode: str = "full",
        jit: bool = True,
    ):
        r"""
        Args:
            input_shape: Shape of input BlockArray. Must correspond to a BlockArray
                with two blocks of equal ndims.
            input_dtype: `dtype` for input argument. Defaults to `float32`.
            mode:  A string indicating the size of the output.  One of "full", "valid", "same".
                Defaults to "full".
            jit:  If ``True``, jit the evaluation of this Operator.

        For more details on `mode`, see :func:`jax.scipy.signal.convolve`.
        """

        if not is_nested(input_shape):
            raise ValueError("A BlockShape is expected; got {input_shape}")
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must have two blocks; got {len(input_shape)}")
        if len(input_shape[0]) != len(input_shape[1]):
            raise ValueError(
                f"Both input blocks must have same number of dimensions; got "
                f"{len(input_shape[0]), len(input_shape[1])}"
            )

        if mode not in ["full", "valid", "same"]:
            raise ValueError(f"Invalid mode={mode}; must be one of 'full', 'valid', 'same'")
        else:
            self.mode = mode

        super().__init__(input_shape=input_shape, input_dtype=input_dtype, jit=jit)

    def _eval(self, x: BlockArray) -> JaxArray:
        return convolve(x[0], x[1], mode=self.mode)

    def freeze(self, argnum: int, val: JaxArray) -> LinearOperator:
        """Returns a new :class:`.LinearOperator` with block argument `argnum` fixed to value `val`.

        If ``argnum == 0``, a :class:`.ConvolveByX` object is returned.
        If ``argnum == 1``, a :class:`.Convolve` object is returned.

        Args:
            argnum:  Index of block to freeze. Must be 0 or 1.
            val:  Value to fix the `argnum`-th input to.
        """

        if argnum == 0:
            return ConvolveByX(
                x=val,
                input_shape=self.input_shape[1],
                input_dtype=self.input_dtype,
                output_shape=self.output_shape,
                mode=self.mode,
            )
        elif argnum == 1:
            return Convolve(
                h=val,
                input_shape=self.input_shape[0],
                input_dtype=self.input_dtype,
                output_shape=self.output_shape,
                mode=self.mode,
            )
        else:
            raise ValueError(f"argnum must be 0 or 1; got {argnum}")
