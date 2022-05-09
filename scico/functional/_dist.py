# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Distance functions."""

from typing import Callable, Union

from scico import numpy as snp
from scico.numpy import BlockArray
from scico.typing import JaxArray

from ._functional import Functional


class SetDistance(Functional):
    r"""Distance to a closed convex set.

    This functional computes the :math:`\ell_2` distance from a vector to
    a closed convex set :math:`C`

    .. math::
        d(\mb{x}) = \min_{\mb{y} \in C} \, \| \mb{x} - \mb{y} \|_2 \;.

    The set is not specified directly, but in terms of a function
    computing the projection into that set, i.e.


    .. math::
        d(\mb{x}) = \| \mb{x} - P_C(\mb{x}) \|_2 \;,

    where :math:`P_C(\mb{x})` is the projection of :math:`\mb{x}` into
    set :math:`C`.
    """

    has_eval = True
    has_prox = True

    def __init__(self, proj: Callable, args=()):
        r"""
        Args:
            proj: Function computing the projection into the convex set.
            args: Additional arguments for function `proj`.
        """
        self.proj = proj
        self.args = args

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Compute the :math:`\ell_2` distance to the set.

        Compute the distance :math:`d(\mb{x})` between :math:`\mb{x}` and
        the set :math:`C`.

        Args:
            x: Input array :math:`\mb{x}`.

        Returns:
            Euclidean distance from `x` to the projection of `x`.
        """
        y = self.proj(*((x,) + self.args))
        return snp.linalg.norm(x - y)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Proximal operator of the :math:`\ell_2` distance function.

        Compute the proximal operator of the :math:`\ell_2` distance
        function :math:`d(\mb{x})` :cite:`beck-2017-first` (Lemma 6.43).

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Scaled proximal operator evaluated at `v`.
        """
        y = self.proj(*((v,) + self.args))
        d = snp.linalg.norm(v - y)
        𝜃 = lam / d if d >= lam else 1.0
        return 𝜃 * y + (1.0 - 𝜃) * v


class SquaredSetDistance(Functional):
    r"""Squared :math:`\ell_2` distance to a closed convex set.

    This functional computes the :math:`\ell_2` distance from a vector to
    a closed convex set :math:`C`

    .. math::
        d(\mb{x}) = \min_{\mb{y} \in C} \, (1/2) \| \mb{x} - \mb{y} \|_2^2
        \;.

    The set is not specified directly, but in terms of a function
    computing the projection into that set, i.e.


    .. math::
        d(\mb{x}) = (1/2) \| \mb{x} - P_C(\mb{x}) \|_2^2 \;,

    where :math:`P_C(\mb{x})` is the projection of :math:`\mb{x}` into
    set :math:`C`.
    """

    has_eval = True
    has_prox = True

    def __init__(self, proj: Callable, args=()):
        r"""
        Args:
            proj: Function computing the projection into the convex set.
            args: Additional arguments for function `proj`.
        """
        self.proj = proj
        self.args = args

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Compute the squared :math:`\ell_2` distance to the set.

        Compute the distance :math:`d(\mb{x})` between :math:`\mb{x}` and
        the set :math:`C`.

        Args:
            x: Input array :math:`\mb{x}`.

        Returns:
            Squared :math:`\ell_2` distance from `x` to the projection of `x`.
        """
        y = self.proj(*((x,) + self.args))
        return 0.5 * snp.linalg.norm(x - y) ** 2

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Proximal operator of the squared :math:`\ell_2` distance function.

        Compute the proximal operator of the squared :math:`\ell_2` distance
        function :math:`d(\mb{x})` :cite:`beck-2017-first` (Example 6.65).

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Scaled proximal operator evaluated at `v`.
        """
        y = self.proj(*((v,) + self.args))
        𝛼 = 1.0 / (1.0 + lam)
        return 𝛼 * v + lam * 𝛼 * y
