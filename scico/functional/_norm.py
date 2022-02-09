# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are norms."""

from typing import Union

from jax import jit

from scico import numpy as snp
from scico.array import no_nan_divide
from scico.blockarray import BlockArray
from scico.numpy import count_nonzero
from scico.numpy.linalg import norm
from scico.typing import JaxArray

from ._functional import Functional


class L0Norm(Functional):
    r"""The :math:`\ell_0` 'norm'.

    Counts the number of non-zero elements in an array.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return count_nonzero(x)

    @staticmethod
    @jit
    def prox(
        v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate scaled proximal operator of :math:`\ell_0` norm.

        Evaluate scaled proximal operator of :math:`\ell_0` norm using

        .. math::

            \mathrm{prox}_{\lambda\| \cdot \|_0}(\mb{v}) =
            \begin{cases}
            \mb{v},  & \text{if } \abs{\mb{v}} \geq \lambda \\
            0,  & \text{else}
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Thresholding parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return snp.where(snp.abs(v) >= lam, v, 0)


class L1Norm(Functional):
    r"""The :math:`\ell_1` norm.

    Computes

    .. math::
       \norm{\mb{x}}_1 = \sum_i \abs{x_i}^2 \;.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return snp.abs(x).sum()

    @staticmethod
    def prox(v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs) -> JaxArray:
        r"""Evaluate scaled proximal operator of :math:`\ell_1` norm.

        Evaluate scaled proximal operator of :math:`\ell_1` norm using

        .. math::
            \mathrm{prox}_{\lambda \|\cdot\|_1}(\mb{v})_i =
            \mathrm{sign}(\mb{v}_i) (\abs{\mb{v}_i} - \lambda)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{if } x \geq 0 \\
            0  & \text{else} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Thresholding parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        tmp = snp.abs(v) - lam
        tmp = 0.5 * (tmp + snp.abs(tmp))
        if snp.iscomplexobj(v):
            out = snp.exp(1j * snp.angle(v)) * tmp
        else:
            out = snp.sign(v) * tmp
        return out


class SquaredL2Norm(Functional):
    r"""Squared :math:`\ell_2` norm.

    Squared :math:`\ell_2` norm

    .. math::
       \norm{\mb{x}}^2_2 = \sum_i \abs{x_i}^2 \;.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Directly implement the squared l2 norm to avoid nondifferentiable
        # behavior of snp.norm(x) at 0.
        return (snp.abs(x) ** 2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of squared :math:`\ell_2` norm.

        Evaluate proximal operator of squared :math:`\ell_2` norm using

        .. math::
            \mathrm{prox}_{\lambda \| \cdot \|_2^2}(\mb{v})
            = \frac{\mb{v}}{1 + 2 \lambda} \;.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return v / (1.0 + 2.0 * lam)


class L2Norm(Functional):
    r""":math:`\ell_2` norm.

    .. math::
       \norm{\mb{x}}_2 = \sqrt{\sum_i \abs{x_i}^2} \;.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return norm(x)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of :math:`\ell_2` norm.

        Evaluate proximal operator of :math:`\ell_2` norm using

        .. math::
            \mathrm{prox}_{\lambda \| \cdot \|_2}(\mb{v})
            = \mb{v} \left(1 - \frac{\lambda}{\norm{v}_2} \right)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{if } x \geq 0 \\
            0  & \text{else} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        norm_v = norm(v)
        if norm_v == 0:
            return 0 * v
        return snp.maximum(1 - lam / norm_v, 0) * v


class L21Norm(Functional):
    r""":math:`\ell_{2,1}` norm.

    For a :math:`M \times N` matrix, :math:`\mb{A}`, by default,

    .. math::
           \norm{\mb{A}}_{2,1} = \sum_{n=1}^N \sqrt{\sum_{m=1}^M
           \abs{A_{m,n}}^2} \;.

    The norm generalizes to more dimensions by first computing the
    :math:`\ell_2` norm along a single (user-specified) dimension,
    followed by a sum over all remaining dimensions.

    For `BlockArray` inputs, the :math:`\ell_2` norm follows the
    reduction rules described in :class:`BlockArray`.

    A typical use case is computing the isotropic total variation norm.
    """

    has_eval = True
    has_prox = True

    def __init__(self, l2_axis: int = 0):
        r"""
        Args:
            l2_axis: Axis over which to take the l2 norm. Default: 0.
        """
        self.l2_axis = l2_axis

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        l2 = norm(x, axis=self.l2_axis)
        return snp.abs(l2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of the :math:`\ell_{2,1}` norm.

        In two dimensions,

        .. math::
            \mathrm{prox}_{\lambda \|\cdot\|_{2,1}}(\mb{v}, \lambda)_{:, n} =
             \frac{\mb{v}_{:, n}}{\|\mb{v}_{:, n}\|_2}
             (\|\mb{v}_{:, n}\|_2 - \lambda)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{if } x \geq 0 \\
            0  & \text{else} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """

        length = norm(v, axis=self.l2_axis, keepdims=True)
        direction = no_nan_divide(v, length)

        new_length = length - lam
        # set negative values to zero without `if`
        new_length = 0.5 * (new_length + snp.abs(new_length))

        return new_length * direction


class NuclearNorm(Functional):
    r"""Nuclear norm.

    Compute the nuclear norm

    .. math::
      \| X \|_* = \sum_i \sigma_i

    where :math:`\sigma_i` are the singular values of matrix :math:`X`.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Original implementation of this function was
        #   return snp.sum(snp.linalg.svd(x, compute_uv=False))
        # The implementation here is a temporary work-around due
        # to the bug reported at https://github.com/google/jax/issues/9483
        s = snp.linalg.svd(x, full_matrices=False, compute_uv=False)
        if isinstance(s, tuple):
            s = s[1]
        return snp.sum(s)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of the nuclear norm.

        Evaluate proximal operator of the nuclear norm
        :cite:`cai-2010-singular`.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """

        svdU, svdS, svdV = snp.linalg.svd(v, full_matrices=False)
        svdS = snp.maximum(0, svdS - lam)
        return svdU @ snp.diag(svdS) @ svdV
