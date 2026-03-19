# -*- coding: utf-8 -*-
# Copyright (C) 2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Circular convolution linear operator for 3D arrays."""

from typing import Optional, Sequence, Union

import numpy as np

from jax.dtypes import result_type

try:
    from jaxdecomp.fft import pfft3d, pifft3d

    have_jaxdecomp = True
except ImportError:
    have_jaxdecomp = False

import scico.numpy as snp
from scico.typing import DType, Shape

from ._circconv import CircularConvolve
from ._linop import LinearOperator


class CircularConvolve3D(CircularConvolve):
    r"""A 3D circular convolution linear operator.

    This linear operator implements circular, three-dimensional
    convolution via pointwise multiplication in the DFT domain;
    it is variant of :class:`CircularConvolve` that is only
    applicable to three-dimensional arrays, with the advantage
    of making use of sharding-efficient FFT operations.
    """

    def __init__(
        self,
        h: snp.Array,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        h_is_dft: bool = False,
        h_center: Optional[Union[snp.Array, np.ndarray, Sequence, float, int]] = None,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            h: Array of filters.
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            h_is_dft: Flag indicating whether `h` is in the DFT domain.
            h_center: Array of length `ndims` specifying the center of
                the filter. Defaults to the upper left corner, i.e.,
                `h_center = [0, 0, ..., 0]`, may be noninteger. May be a
                ``float`` or ``int`` if `h` is one-dimensional.
            jit:  If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """
        if not have_jaxdecomp:
            raise RuntimeError("Package jaxdecomp is required for use of class CircularConvolve3D.")

        self.ndims = 3

        if h_is_dft and h_center is not None:
            raise ValueError("Argument 'h_center' must be None when h_is_dft=True.")
        self.h_center = h_center

        if h_is_dft:
            self.h_dft = h
            output_dtype = snp.dtype(input_dtype)  # cannot infer from h_dft because it is complex
        else:
            if h.shape != input_shape:
                if np.all(np.array(h.shape) > np.array(input_shape)):
                    raise ValueError("h must be smaller than or equal to input_shape on each axis.")
                # Padding required because pfft3d does not have a result shape parameter
                pad_width = [(0, wi - wh) for wh, wi in zip(h.shape, input_shape)]
                h = snp.pad(h, pad_width)
            self.h_dft = pfft3d(h)
            output_dtype = result_type(h.dtype, input_dtype)

            if self.h_center is not None:
                shift = self._dft_center_shift(input_shape)
                # transpose require due to transpose imposed bu pfft3d
                self.h_dft = self.h_dft * shift.transpose((1, 2, 0))

        self.real = output_dtype.kind != "c"

        output_shape = input_shape

        LinearOperator.__init__(
            self,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=jit,
        )

    def _eval(self, x: snp.Array) -> snp.Array:
        x = x.astype(self.input_dtype)
        x_dft = pfft3d(x)
        hx = pifft3d(
            self.h_dft * x_dft,
        )
        if self.real:
            hx = hx.real
        return hx

    def _adj(self, x: snp.Array) -> snp.Array:  # type: ignore
        x_dft = pfft3d(x)
        H_adj_x = pifft3d(
            snp.conj(self.h_dft) * x_dft,
        )
        if self.real:
            H_adj_x = H_adj_x.real
        return H_adj_x
