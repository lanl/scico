# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Discrete Fourier transform linear operator class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import scico.numpy as snp
from scico.typing import JaxArray, Shape

from ._linop import LinearOperator


class DFT(LinearOperator):
    r"""Multi-dimensional discrete Fourier transform."""

    def __init__(
        self,
        input_shape: Shape,
        axes: Optional[Sequence] = None,
        axes_shape: Optional[Shape] = None,
        norm: Optional[str] = None,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            axes: Axes over which to compute the DFT. If ``None``, the
                DFT is computed over all axes.
            axes_shape: Output shape on the subset of array axes selected
                by `axes`. This parameter has the same behavior as the
                `s` parameter of :func:`numpy.fft.fftn`.
            norm: DFT normalization mode. See the `norm` parameter of
                :func:`numpy.fft.fftn`.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the LinearOperator.
        """
        if axes is not None and axes_shape is not None and len(axes) != len(axes_shape):
            raise ValueError(
                f"len(axes)={len(axes)} does not equal len(axes_shape)={len(axes_shape)}"
            )

        if axes_shape is not None:
            if axes is None:
                axes = tuple(range(len(input_shape) - len(axes_shape), len(input_shape)))
            tmp_output_shape = list(input_shape)
            for i, s in zip(axes, axes_shape):
                tmp_output_shape[i] = s
            output_shape = tuple(tmp_output_shape)
        else:
            output_shape = input_shape

        if axes is None or axes_shape is None:
            self.inv_axes_shape = None
        else:
            self.inv_axes_shape = [input_shape[i] for i in axes]

        self.axes = axes
        self.axes_shape = axes_shape
        self.norm = norm

        # To satisfy mypy -- DFT shapes must be tuples, not list of tuple
        # These get set inside of super().__init__ call, but we want to have
        # more restrictive type than the general LinearOperator
        self.input_shape: Shape
        self.output_shape: Shape

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.complex64,
            output_dtype=np.complex64,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        return snp.fft.fftn(x, s=self.axes_shape, axes=self.axes, norm=self.norm)

    def inv(self, z: JaxArray) -> JaxArray:
        """Compute the inverse of this LinearOperator.

        Compute the inverse of this LinearOperator applied to `z`.

        Args:
            z: Input array to inverse DFT.
        """
        return snp.fft.ifftn(z, s=self.inv_axes_shape, axes=self.axes, norm=self.norm)
