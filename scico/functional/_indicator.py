# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are indicator functions/constraints."""

from typing import Union

import jax

from scico import numpy as snp
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm

from ._functional import Functional


class NonNegativeIndicator(Functional):
    r"""Indicator function for non-negative orthant.

    Returns 0 if all elements of input array-like are non-negative, and
    `inf` otherwise

    .. math::
        I(\mb{x}) = \begin{cases}
        0  & \text{ if } x_i \geq 0 \; \forall i \\
        \infty  & \text{ otherwise} \;.
        \end{cases}
    """

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        if snp.util.is_complex_dtype(x.dtype):
            raise ValueError("Not defined for complex input.")

        # Equivalent to snp.inf if snp.any(x < 0) else 0.0
        return jax.lax.cond(snp.any(x < 0), lambda x: snp.inf, lambda x: 0.0, None)

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""The scaled proximal operator of the non-negative indicator.

        Evaluate the scaled proximal operator of the indicator over
        the non-negative orthant, :math:`I`,

        .. math::
            [\mathrm{prox}_{\lambda I}(\mb{v})]_i =
            \begin{cases}
            v_i\, & \text{ if } v_i \geq 0 \\
            0\, & \text{ otherwise} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda` (has no effect).
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return snp.maximum(v, 0)


class L2BallIndicator(Functional):
    r"""Indicator function for :math:`\ell_2` ball of given radius.

    Indicator function for :math:`\ell_2` ball of given radius, :math:`r`

    .. math::
        I(\mb{x}) = \begin{cases}
        0  & \text{ if } \norm{\mb{x}}_2 \leq r \\
        \infty  & \text{ otherwise} \;.
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

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        # Equivalent to
        # snp.inf if norm(x) > self.radius else 0.0
        return jax.lax.cond(norm(x) > self.radius, lambda x: snp.inf, lambda x: 0.0, None)

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""The scaled proximal operator of the :math:`\ell_2` ball indicator.
        a :math:`\ell_2` ball

        Evaluate the scaled proximal operator of the indicator, :math:`I`,
        of the :math:`\ell_2` ball with radius :math:`r`

        .. math::
            \mathrm{prox}_{\lambda I}(\mb{v}) = r \frac{\mb{v}}{\norm{\mb{v}}_2}\;.
        """
        return self.radius * v / norm(v)
