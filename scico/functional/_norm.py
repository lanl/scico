# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are norms."""

from typing import Optional, Tuple, Union

from jax import jit, lax

from scico import numpy as snp
from scico.numpy import BlockArray, count_nonzero
from scico.numpy.linalg import norm
from scico.numpy.util import no_nan_divide
from scico.typing import JaxArray

from ._functional import Functional


class L0Norm(Functional):
    r"""The :math:`\ell_0` 'norm'.

    The :math:`\ell_0` 'norm' counts the number of non-zero elements in
    an array.
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

            \left[ \prox_{\lambda\| \cdot \|_0}(\mb{v}) \right]_i =
            \begin{cases}
            v_i  & \text{ if } \abs{v_i} \geq \lambda \\
            0  & \text{ otherwise } \;.
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
        return snp.sum(snp.abs(x))

    @staticmethod
    def prox(v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs) -> JaxArray:
        r"""Evaluate scaled proximal operator of :math:`\ell_1` norm.

        Evaluate scaled proximal operator of :math:`\ell_1` norm using

        .. math::
            \left[ \prox_{\lambda \|\cdot\|_1}(\mb{v}) \right]_i =
            \sign(v_i) (\abs{v_i} - \lambda)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{ if } x \geq 0 \\
            0  & \text{ otherwise} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Thresholding parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        tmp = snp.abs(v) - lam
        tmp = 0.5 * (tmp + snp.abs(tmp))
        if snp.util.is_complex_dtype(v.dtype):
            out = snp.exp(1j * snp.angle(v)) * tmp
        else:
            out = snp.sign(v) * tmp
        return out


class SquaredL2Norm(Functional):
    r"""The squared :math:`\ell_2` norm.

    Squared :math:`\ell_2` norm

    .. math::
       \norm{\mb{x}}^2_2 = \sum_i \abs{x_i}^2 \;.
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Directly implement the squared l2 norm to avoid nondifferentiable
        # behavior of snp.norm(x) at 0.
        return snp.sum(snp.abs(x) ** 2)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of squared :math:`\ell_2` norm.

        Evaluate proximal operator of squared :math:`\ell_2` norm using

        .. math::
            \prox_{\lambda \| \cdot \|_2^2}(\mb{v})
            = \frac{\mb{v}}{1 + 2 \lambda} \;.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return v / (1.0 + 2.0 * lam)


class L2Norm(Functional):
    r"""The :math:`\ell_2` norm.

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
            \prox_{\lambda \| \cdot \|_2}(\mb{v}) = \mb{v} \,
            \left(1 - \frac{\lambda}{\norm{\mb{v}}_2} \right)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{ if } x \geq 0 \\
            0  & \text{ otherwise} \;.
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
    r"""The :math:`\ell_{2,1}` norm.

    For a :math:`M \times N` matrix, :math:`\mb{A}`, by default,

    .. math::
           \norm{\mb{A}}_{2,1} = \sum_{n=1}^N \sqrt{\sum_{m=1}^M
           \abs{A_{m,n}}^2} \;.

    The norm generalizes to more dimensions by first computing the
    :math:`\ell_2` norm along one or more (user-specified) axes,
    followed by a sum over all remaining axes.

    For `BlockArray` inputs, the :math:`\ell_2` norm follows the
    reduction rules described in :class:`BlockArray`.

    A typical use case is computing the isotropic total variation norm.
    """

    has_eval = True
    has_prox = True

    def __init__(self, l2_axis: Union[int, Tuple] = 0):
        r"""
        Args:
            l2_axis: Axis/axes over which to take the l2 norm. Default: 0.
        """
        self.l2_axis = l2_axis

    @staticmethod
    def _l2norm(
        x: Union[JaxArray, BlockArray], axis: Union[int, Tuple], keepdims: Optional[bool] = False
    ):
        r"""Return the :math:`\ell_2` norm of an array."""
        return snp.sqrt(snp.sum(snp.abs(x) ** 2, axis=axis, keepdims=keepdims))

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        l2 = L21Norm._l2norm(x, axis=self.l2_axis)
        return snp.abs(l2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of the :math:`\ell_{2,1}` norm.

        In two dimensions,

        .. math::
            \prox_{\lambda \|\cdot\|_{2,1}}(\mb{v}, \lambda)_{:, n} =
             \frac{\mb{v}_{:, n}}{\|\mb{v}_{:, n}\|_2}
             (\|\mb{v}_{:, n}\|_2 - \lambda)_+ \;,

        where

        .. math::
            (x)_+ = \begin{cases}
            x  & \text{ if } x \geq 0 \\
            0  & \text{ otherwise} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        length = L21Norm._l2norm(v, axis=self.l2_axis, keepdims=True)
        direction = no_nan_divide(v, length)

        new_length = length - lam
        # set negative values to zero without `if`
        new_length = 0.5 * (new_length + snp.abs(new_length))

        return new_length * direction


class HuberNorm(Functional):
    r"""Huber norm.

    Compute a norm based on the Huber function :cite:`huber-1964-robust`
    :cite:`beck-2017-first` (Sec. 6.7.1). In the non-separable case the
    norm is

    .. math::
         H_{\delta}(\mb{x}) = \begin{cases}
         (1/2) \norm{ \mb{x} }_2^2  & \text{ when } \norm{ \mb{x} }_2
         \leq \delta \\
         \delta \left( \norm{ \mb{x} }_2  - (\delta / 2) \right) &
         \text{ when } \norm{ \mb{x} }_2 > \delta \;,
         \end{cases}

    where :math:`\delta` is a parameter controlling the transitions
    between :math:`\ell_1`-norm like and :math:`\ell_2`-norm like
    behavior. In the separable case the norm is

    .. math::
         H_{\delta}(\mb{x}) = \sum_i h_{\delta}(x_i) \,,

    where

    .. math::
         h_{\delta}(x) = \begin{cases}
         (1/2) \abs{ x }^2  & \text{ when } \abs{ x } \leq \delta \\
         \delta \left( \abs{ x }  - (\delta / 2) \right) &
         \text{ when } \abs{ x } > \delta \;.
         \end{cases}
    """

    has_eval = True
    has_prox = True

    def __init__(self, delta: float = 1.0, separable: bool = True):
        r"""
        Args:
            delta: Huber function parameter :math:`\delta`.
            separable: Flag indicating whether to compute separable or
               non-separable form.
        """
        self.delta = delta
        self.separable = separable

        if separable:
            self._call = self._call_sep
            self._prox = self._prox_sep
        else:
            self._call_lt_branch = lambda xl2: 0.5 * xl2**2
            self._call_gt_branch = lambda xl2: self.delta * (xl2 - self.delta / 2.0)
            self._call = self._call_nonsep
            self._prox = self._prox_nonsep

        super().__init__()

    def _call_sep(self, x: Union[JaxArray, BlockArray]) -> float:
        xabs = snp.abs(x)
        hx = snp.where(
            xabs <= self.delta, 0.5 * xabs**2, self.delta * (xabs - (self.delta / 2.0))
        )
        return snp.sum(hx)

    def _call_nonsep(self, x: Union[JaxArray, BlockArray]) -> float:
        xl2 = snp.linalg.norm(x)
        return lax.cond(xl2 <= self.delta, self._call_lt_branch, self._call_gt_branch, xl2)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self._call(x)

    def _prox_sep(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        den = snp.maximum(snp.abs(v), self.delta * (1.0 + lam))
        return (1 - ((self.delta * lam) / den)) * v

    def _prox_nonsep(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        vl2 = snp.linalg.norm(v)
        den = snp.maximum(vl2, self.delta * (1.0 + lam))
        return (1 - ((self.delta * lam) / den)) * v

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of the Huber function.

        Evaluate scaled proximal operator of the Huber function
        :cite:`beck-2017-first` (Sec. 6.7.3). The prox is

        .. math::
             \prox_{\lambda H_{\delta}} (\mb{v}) = \left( 1 -
             \frac{\lambda \delta} {\max\left\{\norm{\mb{v}}_2,
             \delta + \lambda \delta\right\} } \right) \mb{v}

        in the non-separable case, and

        .. math::
             \left[ \prox_{\lambda H_{\delta}} (\mb{v}) \right]_i =
             \left( 1 - \frac{\lambda \delta} {\max\left\{\abs{v_i},
             \delta + \lambda \delta\right\} } \right) v_i

        in the separable case.


        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return self._prox(v, lam=lam, **kwargs)


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
        return snp.sum(snp.linalg.svd(x, full_matrices=False, compute_uv=False))

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
