# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Convolution linear operator class."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import numpy as np

from jax.dtypes import result_type
from jax.scipy.signal import convolve

import scico.numpy as snp
from scico.typing import DType, Shape

from ._linop import LinearOperator, _wrap_add_sub, _wrap_mul_div_scalar


class Convolve(LinearOperator):
    """A convolution linear operator."""

    def __init__(
        self,
        h: snp.Array,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        mode: str = "full",
        jit: bool = True,
        **kwargs,
    ):
        r"""Wrap :func:`jax.scipy.signal.convolve` as a :class:`.LinearOperator`.

        Args:
            h: Convolutional filter. Must have same number of dimensions
                as `len(input_shape)`.
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            mode: A string indicating the size of the output. One of
                "full", "valid", "same". Defaults to "full".
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.

        For more details on `mode`, see :func:`jax.scipy.signal.convolve`.
        """

        self.h: snp.Array  # : Convolution kernel
        self.mode: str  # : Convolution mode

        if h.ndim != len(input_shape):
            raise ValueError(f"h.ndim = {h.ndim} must equal len(input_shape) = {len(input_shape)}.")
        self.h = h

        if mode not in ["full", "valid", "same"]:
            raise ValueError(f"Invalid mode={mode}; must be one of 'full', 'valid', 'same'.")

        self.mode = mode

        if input_dtype is None:
            input_dtype = self.h.dtype

        output_dtype = result_type(input_dtype, self.h.dtype)

        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: snp.Array) -> snp.Array:
        return convolve(x, self.h, mode=self.mode)

    @_wrap_add_sub
    def __add__(self, other):
        if self.mode != other.mode:
            raise ValueError(f"Incompatible modes:  {self.mode} != {other.mode}.")

        if self.h.shape == other.h.shape:
            return Convolve(
                h=self.h + other.h,
                input_shape=self.input_shape,
                input_dtype=result_type(self.input_dtype, other.input_dtype),
                mode=self.mode,
                output_shape=self.output_shape,
                adj_fn=lambda x: self.adj(x) + other.adj(x),
            )

        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_add_sub
    def __sub__(self, other):
        if self.mode != other.mode:
            raise ValueError(f"Incompatible modes:  {self.mode} != {other.mode}.")

        if self.h.shape == other.h.shape:
            return Convolve(
                h=self.h - other.h,
                input_shape=self.input_shape,
                input_dtype=result_type(self.input_dtype, other.input_dtype),
                mode=self.mode,
                output_shape=self.output_shape,
                adj_fn=lambda x: self.adj(x) - other.adj(x),
            )
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return Convolve(
            h=self.h * scalar,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, type(scalar)),
            mode=self.mode,
            output_shape=self.output_shape,
            adj_fn=lambda x: snp.conj(scalar) * self.adj(x),
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return Convolve(
            h=self.h / scalar,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, type(scalar)),
            mode=self.mode,
            output_shape=self.output_shape,
            adj_fn=lambda x: self.adj(x) / snp.conj(scalar),
        )


class ConvolveByX(LinearOperator):
    """A LinearOperator that performs convolution as a function of the first argument.

    The :class:`LinearOperator` `ConvolveByX(x=x)(y)` implements
    `jax.scipy.signal.convolve(x, y)`.
    """

    def __init__(
        self,
        x: snp.Array,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        mode: str = "full",
        jit: bool = True,
        **kwargs,
    ):
        r"""

        Args:
            x: Convolutional filter. Must have same number of dimensions
                as `len(input_shape)`.
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            mode: A string indicating the size of the output. One of
                "full", "valid", "same". Defaults to "full".
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.

        For more details on `mode`, see :func:`jax.scipy.signal.convolve`.
        """

        self.x: snp.Array  # : Fixed signal to convolve with
        self.mode: str  # : Convolution mode

        if x.ndim != len(input_shape):
            raise ValueError(f"x.ndim = {x.ndim} must equal len(input_shape) = {len(input_shape)}.")

        # Ensure that x is a numpy or jax array.
        if not snp.util.is_arraylike(x):
            raise TypeError(f"Expected numpy or jax array, got {type(x)}.")
        self.x = x

        if mode not in ["full", "valid", "same"]:
            raise ValueError(f"Invalid mode={mode}; must be one of 'full', 'valid', 'same'.")

        self.mode = mode

        if input_dtype is None:
            input_dtype = x.dtype

        output_dtype = result_type(input_dtype, x.dtype)

        super().__init__(
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, h: snp.Array) -> snp.Array:
        return convolve(self.x, h, mode=self.mode)

    @_wrap_add_sub
    def __add__(self, other):
        if self.mode != other.mode:
            raise ValueError(f"Incompatible modes:  {self.mode} != {other.mode}.")
        if self.x.shape == other.x.shape:
            return ConvolveByX(
                x=self.x + other.x,
                input_shape=self.input_shape,
                input_dtype=result_type(self.input_dtype, other.input_dtype),
                mode=self.mode,
                output_shape=self.output_shape,
                adj_fn=lambda x: self.adj(x) + other.adj(x),
            )
        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_add_sub
    def __sub__(self, other):
        if self.mode != other.mode:
            raise ValueError(f"Incompatible modes:  {self.mode} != {other.mode}.")

        if self.x.shape == other.x.shape:
            return ConvolveByX(
                x=self.x - other.x,
                input_shape=self.input_shape,
                input_dtype=result_type(self.input_dtype, other.input_dtype),
                mode=self.mode,
                output_shape=self.output_shape,
                adj_fn=lambda x: self.adj(x) - other.adj(x),
            )

        raise ValueError(f"Incompatible shapes: {self.shape} != {other.shape}.")

    @_wrap_mul_div_scalar
    def __mul__(self, scalar):
        return ConvolveByX(
            x=self.x * scalar,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, type(scalar)),
            mode=self.mode,
            output_shape=self.output_shape,
            adj_fn=lambda x: snp.conj(scalar) * self.adj(x),
        )

    @_wrap_mul_div_scalar
    def __truediv__(self, scalar):
        return ConvolveByX(
            x=self.x / scalar,
            input_shape=self.input_shape,
            input_dtype=result_type(self.input_dtype, type(scalar)),
            mode=self.mode,
            output_shape=self.output_shape,
            adj_fn=lambda x: self.adj(x) / snp.conj(scalar),
        )
