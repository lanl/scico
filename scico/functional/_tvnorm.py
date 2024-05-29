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
    Crop,
    FiniteDifference,
    LinearOperator,
    Pad,
    SingleAxisFiniteDifference,
    VerticalStack,
    linop_over_axes,
)
from scico.numpy import Array
from scico.numpy.util import parse_axes
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
        axes: Optional[Axes] = None,
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
            axes: Axis or axes over which to apply finite difference
                operator. If not specified, or ``None``, differences are
                evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`__call__` and
                :meth:`prox`.
            input_dtype: `dtype` of input arrays of :meth:`__call__` and
                :meth:`prox`.
        """
        self.norm = norm
        self.circular = circular
        self.axes = axes
        self.G: Optional[LinearOperator] = None
        self.WP: Optional[LinearOperator] = None

        if input_shape is not None:
            self.G = self._call_operator(input_shape, input_dtype, axes)
            self.WP, self.CWT = self._prox_operators(input_shape, input_dtype, axes)

    def _call_operator(
        self, input_shape: Shape, input_dtype: DType, axes: Optional[Axes]
    ) -> LinearOperator:
        """Construct operator required by __call__ method."""
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
            self.G = self._call_operator(x.shape, x.dtype, self.axes)
        return self.norm(self.G @ x)

    def _prox_operators(
        self, input_shape: Shape, input_dtype: DType, axes: Optional[Axes]
    ) -> Tuple[LinearOperator, LinearOperator]:
        """Construct operators required by prox method."""
        axes = parse_axes(self.axes, input_shape)
        w_input_shape = (
            # circular boundary: shape of input array
            input_shape
            if self.circular
            # non-circular boundary: shape of input array on non-differenced
            #    axes and one greater for axes that are differenced
            else tuple([s + 1 if i in axes else s for i, s in enumerate(input_shape)])  # type: ignore
        )
        W = HaarTransform(w_input_shape, input_dtype=input_dtype, axes=axes, jit=True)
        if self.circular:
            WP, CWT = W, W.T
        else:
            pad_width = [(0, 1) if i in axes else (0, 0) for i, s in enumerate(input_shape)]  # type: ignore
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
        axes = parse_axes(self.axes, v.shape)
        ndims = len(axes)
        K = 2 * ndims

        if self.WP is None or self.WP.shape[1] != v.shape:
            self.WP, self.CWT = self._prox_operators(v.shape, v.dtype, self.axes)

        if self.circular:
            slce = snp.s_[:, 1]
        else:
            slce = (
                snp.s_[:],
                snp.s_[1],
            ) + tuple([snp.s_[:-1] if i in axes else snp.s_[:] for i, s in enumerate(input_shape)])
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
        axes: Optional[Axes] = None,
        input_shape: Optional[Shape] = None,
        input_dtype: DType = snp.float32,
    ):
        """
        Args:
            circular: Flag indicating use of circular boundary conditions.
            axes: Axis or axes over which to apply finite difference
                operator. If not specified, or ``None``, differences are
                evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
            input_dtype: `dtype` of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
        """
        super().__init__(
            L1Norm(),
            circular=circular,
            axes=axes,
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
        axes: Optional[Axes] = None,
        input_shape: Optional[Shape] = None,
        input_dtype: DType = snp.float32,
    ):
        r"""
        Args:
            circular: Flag indicating use of circular boundary conditions.
            axes: Axis or axes over which to apply finite difference
                operator. If not specified, or ``None``, differences are
                evaluated along all axes.
            input_shape: Shape of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
            input_dtype: `dtype` of input arrays of :meth:`~.TVNorm.__call__` and
                :meth:`~.TVNorm.prox`.
        """
        super().__init__(
            L21Norm(),
            circular=circular,
            axes=axes,
            input_shape=input_shape,
            input_dtype=input_dtype,
        )


class SingleAxisFiniteSum(LinearOperator):
    r"""Two-point sum operator acting along a single axis.

    Boundary handling is circular,  so that the sum operator corresponds
    to the matrix

    .. math::

       \left(\begin{array}{rrrrr}
        1 & 0 & 0 & \ldots & 0\\
       1 & 1 & 0 & \ldots & 0\\
       0 & 1 & 1 & \ldots & 0\\
       \vdots & \vdots & \ddots & \ddots & \vdots\\
       0 & 0 & \ldots & 1 & 1\\
       1 & 0 & \dots & 0 & 1
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

        super().__init__(
            input_shape=input_shape,
            output_shape=input_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            jit=jit,
            **kwargs,
        )

    def _eval(self, x: snp.Array) -> snp.Array:
        return x + snp.roll(x, -1, self.axis)


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
            input_shape, input_dtype=input_dtype, axis=axis, circular=True, jit=jit, **kwargs
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
