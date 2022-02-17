# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Circular convolution linear operators."""

import math
import operator
from functools import partial
from typing import Optional

import numpy as np

from jax.dtypes import result_type

import scico.numpy as snp
from scico._generic_operators import Operator
from scico.typing import DType, JaxArray, Shape

from ._linop import LinearOperator, _wrap_add_sub, _wrap_mul_div_scalar


class CircularConvolve(LinearOperator):
    r"""A circular convolution linear operator.

    This linear operator implements circular, n-dimensional convolution
    via pointwise multiplication in the DFT domain. In its simplest form,
    it implements a single convolution and can be represented by linear
    operator :math:`H` such that

    .. math::
       H \mb{x} = \mb{h} \ast \mb{x} \;,

    where :math:`\mb{h}` is a user-defined filter.

    More complex forms, corresponding to the case where either the input
    (as represented by parameter `input_shape`) or filter (parameter `h`)
    have additional axes that are not involved in the convolution are
    also supported. These follow numpy broadcasting rules. For example:

    Additional axes in the input :math:`\mb{x}` and not in :math:`\mb{h}`
    corresponds to the operation

    .. math::
       H \mb{x} = \left( \begin{array}{ccc}  H' & 0 & \ldots\\
                                            0 & H' & \ldots\\
                                            \vdots & \vdots & \ddots
                        \end{array} \right)
       \left( \begin{array}{c}  \mb{x}_0\\ \mb{x}_1\\ \vdots \end{array}
       \right) \;.

    Additional axes in :math:`\mb{h}` corresponds to multiple filters,
    which will be denoted by :math:`\{\mb{h}_m\}`, with corresponding
    individual linear operations being denoted by :math:`h_m \mb{x}_m =
    \mb{h}_m \ast \mb{x}_m`. The full linear operator can then be
    represented as

    .. math::
       H \mb{x} = \left( \begin{array}{c}  H_0\\ H_1\\ \vdots \end{array}
       \right) \mb{x} \;.

    if the input is singleton, and as

    .. math::
       H \mb{x} = \left( \begin{array}{ccc}  H_0 & 0 & \ldots\\
                                            0 & H_1 & \ldots\\
                                            \vdots & \vdots & \ddots
                        \end{array} \right)
       \left( \begin{array}{c}  \mb{x}_0\\ \mb{x}_1\\ \vdots \end{array}
       \right)

    otherwise.
    """

    def __init__(
        self,
        h: JaxArray,
        input_shape: Shape,
        ndims: Optional[int] = None,
        input_dtype: DType = snp.float32,
        h_is_dft: bool = False,
        h_center: Optional[JaxArray] = None,
        jit: bool = True,
        **kwargs,
    ):
        """
        Args:
            h: Array of filters.
            input_shape: Shape of input array.
            ndims: Number of (trailing) dimensions of the input and `h`
               involved in the convolution. Defaults to the number of
               dimensions in the input.
            input_dtype: `dtype` for input argument. Defaults to
               ``float32``.
            h_is_dft: Flag indicating whether `h` is in the DFT domain.
            h_center: Array of length `ndims` specifying the center of
               the filter. Defaults to the upper left corner, i.e.,
               `h_center = [0, 0, ..., 0]`, may be noninteger.
            jit:  If ``True``, jit the evaluation, adjoint, and gram
               functions of the LinearOperator.
        """

        if ndims is None:
            self.ndims = len(input_shape)
        else:
            self.ndims = ndims

        if h_is_dft and h_center is not None:
            raise ValueError("h_center is not supported when h_is_dft=True")
        self.h_center = h_center

        if h_is_dft:
            self.h_dft = h
            output_dtype = snp.dtype(input_dtype)  # cannot infer from h_dft because it is complex
        else:
            fft_shape = input_shape[-self.ndims :]
            pad = ()
            fft_axes = list(range(h.ndim - self.ndims, h.ndim))
            self.h_dft = snp.fft.fftn(h, s=fft_shape, axes=fft_axes)
            output_dtype = result_type(h.dtype, input_dtype)

            if h_center is not None:
                offset = -self.h_center
                shifts = np.ix_(
                    *tuple(
                        np.exp(-1j * k * 2 * np.pi * np.fft.fftfreq(s))
                        for k, s in zip(offset, input_shape[-self.ndims :])
                    )
                )
                # prevent accidental promotion to double
                shifts = tuple(s.astype(self.h_dft) for s in shifts)
                shift = math.prod(shifts)  # np.prod warns
                self.h_dft = self.h_dft * shift

        self.real = output_dtype.kind != "c"

        try:
            output_shape = np.broadcast_shapes(self.h_dft.shape, input_shape)
        except ValueError:
            raise ValueError(
                f"h shape after padding was {self.h_dft.shape}, needs to be compatible "
                f"for broadcasting with {input_shape}."
            )

        self.batch_axes = tuple(
            range(0, len(output_shape) - len(input_shape))
        )  # used in adjoint to undo broadcasting

        self.ifft_axes = list(range(len(output_shape) - self.ndims, len(output_shape)))
        self.x_fft_axes = list(range(len(input_shape) - self.ndims, len(input_shape)))

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        x = x.astype(self.input_dtype)
        x_dft = snp.fft.fftn(x, axes=self.x_fft_axes)
        hx = snp.fft.ifftn(
            self.h_dft * x_dft,
            axes=self.ifft_axes,
        )
        if self.real:
            hx = hx.real
        return hx

    def _adj(self, x: JaxArray) -> JaxArray:
        x_dft = snp.fft.fftn(x, axes=self.ifft_axes)
        H_adj_x = snp.fft.ifftn(
            snp.conj(self.h_dft) * x_dft,
            axes=self.ifft_axes,
            s=self.input_shape[-self.ndims :],
        )
        H_adj_x = snp.sum(H_adj_x, axis=self.batch_axes)  # adjoint of the broadcast
        if self.real:
            H_adj_x = H_adj_x.real
        return H_adj_x

    @partial(_wrap_add_sub, op=operator.add)
    def __add__(self, other):
        if self.ndims != other.ndims:
            raise ValueError(f"Incompatible ndims:  {self.ndims} != {other.ndims}")

        return CircularConvolve(
            h=self.h_dft + other.h_dft,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, other.input_dtype),
            ndims=self.ndims,
            h_is_dft=True,
        )

    @partial(_wrap_add_sub, op=operator.sub)
    def __sub__(self, other):
        if self.ndims != other.ndims:
            raise ValueError(f"Incompatible ndims:  {self.ndims} != {other.ndims}")

        return CircularConvolve(
            h=self.h_dft - other.h_dft,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, other.input_dtype),
            ndims=self.ndims,
            h_is_dft=True,
        )

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return CircularConvolve(
            h=self.h_dft * scalar,
            input_shape=self.input_shape,
            ndims=self.ndims,
            input_dtype=self.input_dtype,
            h_is_dft=True,
        )

    @_wrap_mul_div_scalar
    def __rmul__(self, scalar):
        return self * scalar

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return CircularConvolve(
            h=self.h_dft / scalar,
            input_shape=self.input_shape,
            ndims=self.ndims,
            input_dtype=self.input_dtype,
            h_is_dft=True,
        )

    def from_operator(
        H: Operator, ndims: Optional[int] = None, center: Optional[Shape] = None, jit: bool = True
    ):
        r"""Construct a CircularConvolve version of a given operator.

        Construct a CircularConvolve version of a given operator,
        which is assumed to be linear and shift invariant (LSI).

        Args:
            H: Input operator.
            ndims: Number of trailing dims over which the H acts.
            center: Location at which to place the Kronecker delta. For
              LSI inputs, this will not matter. Defaults to the center
              of H.input_shape, i.e., (n_1 // 2, n_2 // 2, ...).
            jit: If ``True``, jit the resulting `CircularConvolve`.
        """

        if ndims is None:
            ndims = len(H.input_shape)
        else:
            ndims = ndims

        if center is None:
            center = tuple(d // 2 for d in H.input_shape[-ndims:])

        # compute impulse response
        d = snp.zeros(H.input_shape, H.input_dtype)
        d = d.at[(Ellipsis,) + center].set(1.0)
        Hd = H @ d

        # build CircularConvolve
        return CircularConvolve(
            Hd,
            H.input_shape,
            ndims=ndims,
            input_dtype=H.input_dtype,
            h_center=snp.array(center),
            jit=jit,
        )


def _gradient_filters(ndim: int, axes: Shape, shape: Shape, dtype: DType = snp.float32) -> JaxArray:
    r"""Construct filters for computing gradients in the frequency domain.

    Construct a set of filters for computing gradients in the frequency domain.

    Args:
        ndim: Total number of dimensions in array in which gradients are
           to be computed.
        axes: Axes on which gradients are to be computed.
        shape: Shape of axes on which gradients are to be computed.
        dtype: Data type of output arrays.

    Returns:
        An array of frequency domain gradient operators
        :math:`\hat{G}_i`.
    """
    g = snp.zeros(
        [
            len(axes),
        ]
        + [2 if k in axes else 1 for k in range(ndim)],
        dtype,
    )

    # put a finite difference along each axis in axes
    for layer, axis in enumerate(axes):
        idx = tuple(slice(None) if k == axis else 0 for k in range(0, g.ndim - 1))
        idx = (layer,) + idx
        ref = g.at[idx]
        g = ref.set([1, -1])

    fft_axes = [ax + 1 for ax in axes]  # first axis is batch

    Gf = snp.fft.fftn(g, shape, axes=fft_axes)

    return Gf
