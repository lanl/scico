# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Total variation norms."""

from typing import Optional, Tuple

from scico import numpy as snp
from scico.linop import (
    CircularConvolve,
    Crop,
    FiniteDifference,
    LinearOperator,
    Pad,
    SingleAxisFiniteDifference,
    VerticalStack,
    linop_over_axes,
)
from scico.numpy import Array
from scico.typing import Axes, DType, Shape

from ._functional import Functional
from ._norm import L1Norm, L21Norm


class TVNorm(Functional):
    r"""Generic total variation (TV) norm.

    Generic total variation (TV) norm with approximation of the scaled
    proximal operator :cite:`kamilov-2016-parallel`
    :cite:`kamilov-2016-minimizing`.
    """

    has_eval = True
    has_prox = True

    def __init__(
        self,
        norm: Functional,
        circular: bool = True,
        ndims: Optional[int] = None,
        input_shape: Optional[Shape] = None,
        input_dtype: DType = snp.float32,
    ):
        """
        While initializers for :class:`.Functional` objects typically do
        not take `input_shape` and `input_dtype` parameters, they are
        included here because methods :meth:`__call__` and :meth:`prox`
        require instantiation of some :class:`.LinearOperator` objects,
        which do take these parameters. If these parameters are not
        provided on intialization of a :class:`TVNorm` object, then
        creation of the required :class:`.LinearOperator` objects is
        deferred until these methods are called, which can result in
        `JAX tracer <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables>`__
        errors when they are components of a jitted function.

        Args:
            norm: Norm functional from which the TV norm is composed.
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`__call__` and
                :meth:`prox`.
            input_dtype: `dtype` of input arrays of :meth:`__call__` and
                :meth:`prox`.
        """
        self.norm = norm
        self.circular = circular
        self.ndims = ndims
        self.h0 = snp.array([1.0, 1.0]) / snp.sqrt(2.0)  # lowpass filter
        self.h1 = snp.array([1.0, -1.0]) / snp.sqrt(2.0)  # highpass filter
        self.G: Optional[LinearOperator] = None
        self.WP: Optional[LinearOperator] = None

        if input_shape is not None:
            if ndims is None:
                ndims = len(input_shape)
            self.G = self._call_operator(ndims, input_shape, input_dtype)
            self.WP, self.CWT = self._prox_operators(ndims, input_shape, input_dtype)

    def _call_operator(self, ndims: int, input_shape: Shape, input_dtype: DType) -> LinearOperator:
        """Construct operator required by __call__ method."""
        axes = tuple(range(len(input_shape) - ndims, len(input_shape)))
        G = FiniteDifference(
            input_shape,
            input_dtype=input_dtype,
            axes=axes,
            circular=self.circular,
            append=None if self.circular else 0,
            jit=True,
        )
        return G

    def __call__(self, x: Array) -> float:
        """Compute the TV norm of an array.

        Args:
            x: Array for which the TV norm should be computed.

        Returns:
              TV norm of `x`.
        """
        if self.G is None or self.G.shape[1] != x.shape:
            if self.ndims is None:
                ndims = x.ndim
            else:
                ndims = self.ndims
            self.G = self._call_operator(ndims, x.shape, x.dtype)
        return self.norm(self.G @ x)

    @staticmethod
    def _shape(idx: int, ndims: int) -> Tuple:
        """Construct a shape tuple.

        Construct a tuple of size `ndims` with all unit entries except
        for index `idx`, which has a -1 entry.
        """
        return (1,) * idx + (-1,) + (1,) * (ndims - idx - 1)

    @staticmethod
    def _center(idx: int, ndims: int) -> Tuple:
        """Construct a center tuple.

        Construct a tuple of size `ndims` with all zero entries except
        for index `idx`, which has a unit entry.
        """
        return (0,) * idx + (1,) + (0,) * (ndims - idx - 1)

    def _haar_operator(self, ndims: int, input_shape: Shape, input_dtype: DType) -> LinearOperator:
        """Construct single-level shift-invariant Haar transform."""
        h0 = self.h0.astype(input_dtype)
        h1 = self.h1.astype(input_dtype)
        ConvOp = lambda h, k: CircularConvolve(
            h.reshape(TVNorm._shape(k, ndims)),
            input_shape,
            ndims=ndims,
            h_center=TVNorm._center(k, ndims),
        )
        L = VerticalStack(  # stack of lowpass filter operators for each axis
            [ConvOp(h0, k) for k in range(ndims)]
        )
        H = VerticalStack(  # stack of highpass filter operators for each axis
            [ConvOp(h1, k) for k in range(ndims)]
        )
        # single-level shift-invariant Haar transform
        return VerticalStack([L, H], jit=True)

    def _prox_operators(
        self, ndims: int, input_shape: Shape, input_dtype: DType
    ) -> Tuple[LinearOperator, LinearOperator]:
        """Construct operators required by prox method."""
        w_input_shape = (
            # circular boundary: shape of input array
            input_shape
            if self.circular
            # non-circular boundary: shape of input array on non-differenced
            #    axes and one greater for axes that are differenced
            else input_shape[0 : (len(input_shape) - ndims)]
            + tuple([n + 1 for n in input_shape[-ndims:]])
        )
        W = self._haar_operator(ndims, w_input_shape, input_dtype)
        if self.circular:
            WP, CWT = W, W.T
        else:
            pad_width = ((0, 0),) * (len(input_shape) - ndims) + ((0, 1),) * ndims
            P = Pad(input_shape, pad_width=pad_width, mode="edge", jit=True)
            WP = W @ P
            C = Crop(crop_width=pad_width, input_shape=w_input_shape, jit=True)
            CWT = C @ W.T
        return WP, CWT

    def prox(self, v: Array, lam: float = 1.0, **kwargs) -> Array:
        r"""Approximate proximal operator of the TV norm.

        Approximation of the proximal operator of the TV norm, computed
        via the methods described in :cite:`kamilov-2016-parallel`
        :cite:`kamilov-2016-minimizing`.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lam`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        if self.ndims is None:
            ndims = v.ndim
        else:
            ndims = self.ndims
        K = 2 * ndims

        if self.WP is None or self.WP.shape[1] != v.shape:
            self.WP, self.CWT = self._prox_operators(ndims, v.shape, v.dtype)

        if self.circular:
            slce = snp.s_[1]
        else:
            slce = (
                (
                    1,
                    snp.s_[:],
                )
                + (snp.s_[:],) * (v.ndim - ndims)
                + (snp.s_[:-1],) * ndims
            )
        # Apply shrinkage to highpass component of shift-invariant Haar transform
        # of padded input (or to non-boundary region thereof for non-circular
        # boundary conditions).
        WPv: Array = self.WP(v)
        WPv = WPv.at[slce].set(self.norm.prox(WPv[slce], snp.sqrt(2) * K * lam))
        u = (1.0 / K) * self.CWT(WPv)

        return u


class AnisotropicTVNorm(TVNorm):
    r"""The anisotropic total variation (TV) norm.

    The anisotropic total variation (TV) norm computed by

    .. code-block:: python

       ATV = scico.functional.AnisotropicTVNorm()
       x_norm = ATV(x)

    is equivalent to

    .. code-block:: python

       C = linop.FiniteDifference(input_shape=x.shape, circular=True)
       L1 = functional.L1Norm()
       x_norm = L1(C @ x)

    The scaled proximal operator is computed using an approximation that
    holds for small scaling parameters :cite:`kamilov-2016-parallel`.
    This does not imply that it can only be applied to problems requiring
    a small regularization parameter since most proximal algorithms
    include an additional algorithm parameter that also plays a role in
    the parameter of the proximal operator. For example, in :class:`.PGM`
    and :class:`.AcceleratedPGM`, the scaled proximal operator parameter
    is the regularization parameter divided by the `L0` algorithm
    parameter, and for :class:`.ADMM`, the scaled proximal operator
    parameters are the regularization parameters divided by the entries
    in the `rho_list` algorithm parameter.
    """

    def __init__(
        self,
        circular: bool = False,
        ndims: Optional[int] = None,
        input_shape: Optional[Shape] = None,
        input_dtype: DType = snp.float32,
    ):
        """
        Args:
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
            input_dtype: `dtype` of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
        """
        super().__init__(
            L1Norm(),
            circular=circular,
            ndims=ndims,
            input_shape=input_shape,
            input_dtype=input_dtype,
        )


class IsotropicTVNorm(TVNorm):
    r"""The isotropic total variation (TV) norm.

    The isotropic total variation (TV) norm computed by

    .. code-block:: python

       ATV = scico.functional.IsotropicTVNorm()
       x_norm = ATV(x)

    is equivalent to

    .. code-block:: python

       C = linop.FiniteDifference(input_shape=x.shape, circular=True)
       L21 = functional.L21Norm()
       x_norm = L21(C @ x)

    The scaled proximal operator is computed using an approximation that
    holds for small scaling parameters :cite:`kamilov-2016-minimizing`.
    This does not imply that it can only be applied to problems requiring
    a small regularization parameter since most proximal algorithms
    include an additional algorithm parameter that also plays a role in
    the parameter of the proximal operator. For example, in :class:`.PGM`
    and :class:`.AcceleratedPGM`, the scaled proximal operator parameter
    is the regularization parameter divided by the `L0` algorithm
    parameter, and for :class:`.ADMM`, the scaled proximal operator
    parameters are the regularization parameters divided by the entries
    in the `rho_list` algorithm parameter.
    """

    def __init__(
        self,
        circular: bool = False,
        ndims: Optional[int] = None,
        input_shape: Optional[Shape] = None,
        input_dtype: DType = snp.float32,
    ):
        r"""
        Args:
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
            input_dtype: `dtype` of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
        """
        super().__init__(
            L21Norm(),
            circular=circular,
            ndims=ndims,
            input_shape=input_shape,
            input_dtype=input_dtype,
        )


class SingleAxisFiniteSum(LinearOperator):
    r"""Two-point sum operator acting along a single axis.

    Left and right hand boundaries are handled via symmetric extension
    so that the sum operator corresponds to the matrix

    .. math::

       \left(\begin{array}{rrrrr}
        1 & 0 & 0 & \ldots & 0\\
       1 & 1 & 0 & \ldots & 0\\
       0 & 1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & 1 & 1\\
       0 & 0 & \dots & 0 & 1
       \end{array}\right) \;.
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        axis: int = -1,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            axis: Axis over which to apply finite sum operator.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """

        if not isinstance(axis, int):
            raise TypeError(f"Expected axis to be of type int, got {type(axis)} instead.")

        if axis < 0:
            axis = len(input_shape) + axis
        if axis >= len(input_shape):
            raise ValueError(
                f"Invalid axis {axis} specified; axis must be less than "
                f"len(input_shape)={len(input_shape)}."
            )

        self.axis = axis

        ndims = len(input_shape)
        self.left_pad = ((0, 0),) * axis + ((1, 0),) + ((0, 0),) * (ndims - axis - 1)
        self.right_pad = ((0, 0),) * axis + ((0, 1),) + ((0, 0),) * (ndims - axis - 1)

        output_shape = tuple(x + (i == axis) for i, x in enumerate(input_shape))

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: snp.Array) -> snp.Array:
        return snp.pad(x, self.left_pad) + snp.pad(x, self.right_pad)


class FiniteSum(VerticalStack):
    """Two-point sum operator.

    Compute two-point sums along the specified axes, returning the
    results in a :class:`jax.Array` (when possible) or :class:`BlockArray`.
    See :class:`VerticalStack` for details on how this choice is made.
    See :class:`SingleAxisFiniteSum` for boundary handling details.
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        axes: Optional[Axes] = None,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            axes: Axis or axes over which to apply sum operator. If not
                specified, or ``None``, sums are evaluated along all axes.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """
        self.axes, ops = linop_over_axes(
            SingleAxisFiniteSum,
            input_shape,
            axes=axes,
            input_dtype=input_dtype,
            jit=False,
        )
        super().__init__(
            ops,  # type: ignore
            jit=jit,
            **kwargs,
        )


class SingleAxisHaarTransform(VerticalStack):
    """Single-level shift-invariant Haar transform along a single axis.

    Compute one level of a shift-invariant Haar transform along the
    specified axis, returning the results in a :class:`jax.Array`
    consisting of sum and difference components (corresponding to lowpass
    and highpass filtered components respectively) on axis 0.
    See :class:`SingleAxisFiniteSum` for boundary handling details.
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        axis: int = -1,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            axis: Axis over which to apply Haar transform.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """
        self.axis = axis
        self.HaarL = (1.0 / 2.0) * SingleAxisFiniteSum(
            input_shape, input_dtype=input_dtype, axis=axis, jit=jit, **kwargs
        )
        self.HaarH = (1.0 / 2.0) * SingleAxisFiniteDifference(
            input_shape, input_dtype=input_dtype, axis=axis, prepend=1, append=1, jit=jit, **kwargs
        )
        super().__init__(
            (self.HaarL, self.HaarH),
            jit=jit,
            **kwargs,
        )


class HaarTransform(VerticalStack):
    """Single-level shift-invariant Haar transform.

    Compute one level of a shift-invariant Haar transform along the
    specified axes, returning the results in a :class:`jax.Array`.
    See :class:`SingleAxisHaarTransform` for details of the transform
    along each axis.
    """

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = snp.float32,
        axes: Optional[Axes] = None,
        jit: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`.
            axes: Axis or axes over which to apply Haar transform. If not
                specified, or ``None``, the transform is evaluated along
                all axes.
            jit: If ``True``, jit the evaluation, adjoint, and gram
                functions of the :class:`LinearOperator`.
        """
        self.axes, ops = linop_over_axes(
            SingleAxisHaarTransform,
            input_shape,
            axes=axes,
            input_dtype=input_dtype,
            jit=False,
        )
        super().__init__(
            ops,  # type: ignore
            jit=jit,
            **kwargs,
        )
