# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are indicator functions/constraints."""

from typing import Union

import jax

from scico import numpy as snp
from scico.blockarray import BlockArray
from scico.numpy.linalg import norm
from scico.typing import JaxArray

from ._functional import Functional

__author__ = """Luke Pfister <pfister@lanl.gov>"""


class NonNegativeIndicator(Functional):
    r"""Indicator function for non-negative orthant.

    Returns 0 if all elements of input array-like are non-negative, and inf otherwise.

    .. math::
        I(\mb{x}) = \begin{cases}
        0,  & \text{if } x_i \geq 0 \text{ for each } i \\
        \infty,  & \text{else}
        \end{cases} \;

    """

    has_eval = True
    has_prox = True
    is_smooth = False

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Equivalent to
        # snp.inf if snp.any(x < 0) else 0.0
        return jax.lax.cond(snp.any(x < 0), lambda x: snp.inf, lambda x: 0.0, None)

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of indicator over non-negative orthant:

        .. math::
            [\mathrm{prox}(\mb{x}, \lambda)]_i =
            \begin{cases}
            x_i, & \text{if } x_i \geq 0 \\
            0, & \text{else}.
            \end{cases}

        Args:
            x :  Input array :math:`\mb{x}`
            lam : Proximal parameter :math:`\lambda`
        """
        return snp.maximum(x, 0)


class L2BallIndicator(Functional):
    r"""Indicator function for :math:`\ell_2` ball of given radius.

    .. math::
        I(\mb{x}) = \begin{cases}
        0,  & \text{if } \norm{\mb{x}}_2 \leq \mathrm{radius} \\
        \infty,  & \text{else}
        \end{cases} \;

    Attributes:
        radius : Radius of :math:`\ell_2` ball

    """

    has_eval = True
    has_prox = True
    is_smooth = False

    def __init__(self, radius: float = 1):
        r"""Initialize a :class:`L2BallIndicator` object.

        Args:
            radius : Radius of :math:`\ell_2` ball.  Default: 1.
        """
        self.radius = radius
        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Equivalent to
        # snp.inf if norm(x) > self.radius else 0.0
        return jax.lax.cond(norm(x) > self.radius, lambda x: snp.inf, lambda x: 0.0, None)

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal operator of indicator over :math:`\ell_2` ball:

        .. math::
            \mathrm{prox}(\mb{x}, \lambda) = \mathrm{radius} \frac{\mb{x}}{\norm{\mb{x}}_2}

        """
        return self.radius * x / norm(x)
