# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
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


class NonNegativeIndicator(Functional):
    r"""Indicator function for non-negative orthant.

    Returns 0 if all elements of input array-like are non-negative, and
    inf otherwise

    .. math::
        I(\mb{x}) = \begin{cases}
        0  & \text{if } x_i \geq 0 \text{ for each } i \\
        \infty  & \text{else} \;.
        \end{cases}

    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        if snp.iscomplexobj(x):
            raise ValueError("Not defined for complex input.")

        # Equivalent to
        # snp.inf if snp.any(x < 0) else 0.0
        return jax.lax.cond(snp.any(x < 0), lambda x: snp.inf, lambda x: 0.0, None)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaluate the scaled proximal operator of the indicator over
        the non-negative orthant, :math:`I_{>= 0}`,

        .. math::
            [\mathrm{prox}_{\lambda I_{>=0}}(\mb{v})]_i =
            \begin{cases}
            v_i\,, & \text{if } v_i \geq 0 \\
            0\,, & \text{otherwise} \;.
            \end{cases}

        Args:
            v : Input array :math:`\mb{v}`.
            lam : Proximal parameter :math:`\lambda` (has no effect).
            kwargs: Additional arguments that may be used by derived
                classes.

        """
        return snp.maximum(v, 0)


class L2BallIndicator(Functional):
    r"""Indicator function for :math:`\ell_2` ball of given radius.

    Indicator function for :math:`\ell_2` ball of given radius

    .. math::
        I(\mb{x}) = \begin{cases}
        0  & \text{if } \norm{\mb{x}}_2 \leq \mathrm{radius} \\
        \infty  & \text{else} \;.
        \end{cases}

    Attributes:
        radius: Radius of :math:`\ell_2` ball.
    """

    has_eval = True
    has_prox = True

    def __init__(self, radius: float = 1):
        r"""Initialize a :class:`L2BallIndicator` object.

        Args:
            radius: Radius of :math:`\ell_2` ball. Default: 1.
        """
        self.radius = radius
        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        # Equivalent to
        # snp.inf if norm(x) > self.radius else 0.0
        return jax.lax.cond(norm(x) > self.radius, lambda x: snp.inf, lambda x: 0.0, None)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evalulate the scaled proximal operator of the indicator over
        a :math:`\ell_2` ball with radius :math:`r` = `self.radius`,
        :math:`I_r`:

        .. math::
            \mathrm{prox}_{\lambda I_r}(\mb{v}) = r \frac{\mb{v}}{\norm{\mb{v}}_2}\;.

        """
        return self.radius * v / norm(v)
