# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Fourier transform linear operator class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional

import numpy as np

import scico.numpy as snp
from scico.typing import JaxArray, Shape

from ._linop import LinearOperator

__author__ = """Luke Pfister <luke.pfister@gmail.com>"""


class DFT(LinearOperator):
    r"""N-dimensional Discrete Fourier Transform."""

    def __init__(
        self, input_shape: Shape, output_shape: Optional[Shape] = None, jit: bool = True, **kwargs
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            output_shape: Shape of transformed output. Along any axis,
                if the given output_shape is larger than the input, the
                input is padded with zeros. If None, the shape of the
                input is used.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """

        if output_shape is None:
            output_shape = input_shape

        if len(output_shape) != len(input_shape):
            raise ValueError(
                f"len(output_shape)={len(output_shape)} does not equal "
                f"len(input_shape)={len(input_shape)}"
            )

        # To satisfy mypy -- DFT shapes must be tuples, not list of tuple
        # These get set inside of super().__init__ call, but we want to have
        # more restrictive type than the general LinearOperator
        self.output_shape: Shape = output_shape
        self.input_shape: Shape = input_shape

        super().__init__(
            input_shape=input_shape,
            input_dtype=np.complex64,
            output_dtype=np.complex64,
            output_shape=output_shape,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        return snp.fft.fftn(x, s=self.output_shape)

    def inv(self, z: JaxArray, truncate: bool = True) -> JaxArray:
        """Compute the inverse of this LinearOperator.

        Compute the inverse of this LinearOperator applied to `z`.

        Args:
            z: Array to take inverse DFT.
            truncate: If `True`, the inverse DFT is truncated to be
               `input_shape`. This may be used when this DFT
               LinearOperator applies zero padding before computing the
               DFT.
        """
        y = snp.fft.ifftn(z)
        if truncate:
            for i, s in enumerate(self.input_shape):
                y = snp.take(y, indices=np.r_[:s], axis=i)

        return y
