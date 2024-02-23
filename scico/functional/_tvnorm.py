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
    FiniteDifference,
    LinearOperator,
    VerticalStack,
)
from scico.numpy import Array

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

    def __init__(self, norm: Functional, circular: bool = True, ndims: Optional[int] = None):
        """
        Args:
            norm: Norm functional from which the TV norm is composed.
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
        """
        self.norm = norm
        self.circular = circular
        self.ndims = ndims
        self.h0 = snp.array([1.0, 1.0]) / snp.sqrt(2.0)  # lowpass filter
        self.h1 = snp.array([1.0, -1.0]) / snp.sqrt(2.0)  # highpass filter
        self.G: Optional[LinearOperator] = None
        self.W: Optional[LinearOperator] = None

    def __call__(self, x: Array) -> float:
        r"""Compute the TV norm of an array.

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
            axes = tuple(range(ndims))
            self.G = FiniteDifference(
                x.shape,
                input_dtype=x.dtype,
                axes=axes,
                circular=self.circular,
                jit=True,
            )
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

        if self.W is None or self.W.shape[1] != v.shape:
            h0 = self.h0.astype(v.dtype)
            h1 = self.h1.astype(v.dtype)

            ConvOp = lambda h, k: CircularConvolve(
                h.reshape(TVNorm._shape(k, ndims)),
                v.shape,
                ndims=ndims,
                h_center=TVNorm._center(k, ndims),
            )

            C0 = VerticalStack(  # stack of lowpass filter operators for each axis
                [ConvOp(h0, k) for k in range(ndims)]
            )
            C1 = VerticalStack(  # stack of highpass filter operators for each axis
                [ConvOp(h1, k) for k in range(ndims)]
            )
            # single-level shift-invariant Haar transform
            self.W = VerticalStack([C0, C1], jit=True)

        Wv = self.W @ v

        if self.circular:
            slce = snp.s_[1]  # restrict to highpass component of shift-invariant Haar transform
        else:
            slce = (snp.s_[1], snp.s_[:],) + (
                snp.s_[:-1],
            ) * ndims  # also omit boundary samples
        Wv = Wv.at[slce].set(self.norm.prox(Wv[slce], snp.sqrt(2) * K * lam))  # apply shrinkage
        return (1.0 / K) * self.W.T @ Wv


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

    def __init__(self, circular: bool = False, ndims: Optional[int] = None):
        """
        Args:
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
        """
        super().__init__(L1Norm(), circular=circular, ndims=ndims)


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

    def __init__(self, circular: bool = False, ndims: Optional[int] = None):
        r"""
        Args:
            circular: Flag indicating use of circular boundary conditions.
            ndims: Number of (trailing) dimensions of the input over
                which to apply the finite difference operator. If
                ``None``, differences are evaluated along all axes.
        """
        super().__init__(L21Norm(), circular=circular, ndims=ndims)
