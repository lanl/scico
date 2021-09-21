# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are norms."""

from typing import Union

from jax import jit

from scico import numpy as snp
from scico.blockarray import BlockArray
from scico.math import safe_divide
from scico.numpy import count_nonzero
from scico.numpy.linalg import norm
from scico.typing import JaxArray

from ._functional import Functional

__author__ = """Luke Pfister <pfister@lanl.gov>, Michael McCann <mccann@lanl.gov>"""


class L0Norm(Functional):
    r"""The :math:`\ell_0` 'norm'. Calculates the number of non-zero elements in an
    array-like."""

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return count_nonzero(x)

    @staticmethod
    @jit
    def prox(x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of :math:`\ell_0` norm


        .. math::

            \mathrm{prox}(\mb{x}, \lambda) =
            \begin{cases}
            \mb{x},  & \text{if } \abs{\mb{x}} \geq \lambda \\
            0,  & \text{else}
            \end{cases}

        Args:
            x : Input array :math:`\mb{x}`
            lam : Thresholding parameter :math:`\lambda`

        """
        return snp.where(snp.abs(x) >= lam, x, 0)


class L1Norm(Functional):
    r"""The :math:`\ell_1` norm.  Computes

    .. math::
       \norm{\mb{x}}_1 = \sum_i \abs{x_i}^2
    """

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return snp.abs(x).sum()

    @staticmethod
    def prox(x: Union[JaxArray, BlockArray], lam: float) -> JaxArray:
        r"""Evaluate proximal operator of :math:`\ell_1` norm

        .. math::
            \mathrm{prox}(\mb{x}, \lambda)_i = \mathrm{sign}(\mb{x}_i)
            (\abs{\mb{x}_i} - \lambda)_+ \;

        where

        .. math::
            (x)_+ = \begin{cases}
            x,  & \text{if } x \geq 0 \\
            0,  & \text{else}
            \end{cases} \;


        """
        tmp = snp.abs(x) - lam
        tmp = 0.5 * (tmp + snp.abs(tmp))
        if snp.iscomplexobj(x):
            out = snp.exp(1j * snp.angle(x)) * tmp
        else:
            out = snp.sign(x) * tmp
        return out


class SquaredL2Norm(Functional):
    r"""Squared :math:`\ell_2` norm.

    .. math::
       \norm{\mb{x}}^2_2 = \sum_i \abs{x_i}^2


    """

    has_eval = True
    has_prox = True
    is_smooth = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Directly implement the squared l2 norm to avoid nondifferentiable
        # behavior of snp.norm(x) at 0.
        return (snp.abs(x) ** 2).sum()

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of squared :math:`\ell_2` norm:

        .. math::
            \mathrm{prox}(\mb{x}, \lambda) = \frac{\mb{x}}{1 + 2 \lambda}

        Args:
            x :  Input array :math:`\mb{x}`
            lam : Proximal parameter :math:`\lambda`
        """
        return x / (1.0 + 2 * lam)


class L2Norm(Functional):
    r""":math:`\ell_2` norm.

    .. math::
       \norm{\mb{x}}_2 = \sqrt{\sum_i \abs{x_i}^2}

    """

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return norm(x)

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of :math:`\ell_2` norm:

        .. math::
            \mathrm{prox}(\mb{x}, \lambda) = \mb{x} \right(1 - \frac{\lambda}{\norm{x}_2}\left)_+,

        where

        .. math::
            (x)_+ = \begin{cases}
            x,  & \text{if } x \geq 0 \\
            0,  & \text{else}
            \end{cases} \;

        Args:
            x :  Input array :math:`\mb{x}`
            lam : Proximal parameter :math:`\lambda`
        """
        norm_x = norm(x)
        if norm_x == 0:
            return 0 * x
        else:
            return snp.maximum(1 - lam / norm_x, 0) * x


class L21Norm(Functional):
    r""":math:`\ell_{2,1}` norm.

    For a :math:`M \times N` matrix, :math:`\mb{A}`, by default,

    .. math::
           \norm{\mb{A}}_{2,1} = \sum_{n=1}^N \sqrt{\sum_{m=1}^M \abs{A_{m,n}}^2}.

    The norm generalizes to more dimensions by first computing the :math:`\ell_2` norm along
    a single (user-specified) dimension, followed by a sum over all remaining dimensions.

    For `BlockArray` inputs, the :math:`\ell_2` norm follows the reduction rules described in :class:`BlockArray`.

    A typical use case is computing the isotropic total variation norm.
    """

    has_eval = True
    has_prox = True
    is_smooth = False

    def __init__(self, l2_axis: int = 0):
        r"""
        Args:
            l2_axis: Axis over which to take the l2 norm. Default: 0.
        """
        self.l2_axis = l2_axis

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        l2 = norm(x, axis=self.l2_axis)
        return snp.abs(l2).sum()

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of the :math:`\ell_{2,1}` norm.

        In two dimensions,

        .. math::
            \mathrm{prox}(\mb{A}, \lambda)_{:, n} = \frac{\mb{A}_{:, n}}{\|\mb{A}_{:, n}\|_2}
             (\|\mb{A}_{:, n}\|_2 - \lambda)_+ \;

        where

        .. math::
            (x)_+ = \begin{cases}
            x,  & \text{if } x \geq 0 \\
            0,  & \text{else}.
            \end{cases} \;
        """

        length = norm(x, axis=self.l2_axis, keepdims=True)
        direction = safe_divide(x, length)

        new_length = length - lam
        # set negative values to zero without `if`
        new_length = 0.5 * (new_length + snp.abs(new_length))

        return new_length * direction
