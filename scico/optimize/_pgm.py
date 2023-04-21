# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal Gradient Method classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Optional, Union

import jax

import scico.numpy as snp
from scico.functional import Functional
from scico.loss import Loss
from scico.numpy import Array, BlockArray
from scico.numpy.util import ensure_on_device

from ._common import Optimizer
from ._pgmaux import (
    AdaptiveBBStepSize,
    BBStepSize,
    PGMStepSize,
    RobustLineSearchStepSize,
)


class PGM(Optimizer):
    r"""Proximal Gradient Method (PGM) base class.

    Minimize a function of the form :math:`f(\mb{x}) + g(\mb{x})`, where
    :math:`f` and the :math:`g` are instances of :class:`.Functional`.

    Uses helper :class:`StepSize` to provide an estimate of the Lipschitz
    constant :math:`L` of :math:`f`. The step size :math:`\alpha` is the
    reciprocal of :math:`L`, i.e.: :math:`\alpha = 1 / L`.
    """

    def __init__(
        self,
        f: Union[Loss, Functional],
        g: Functional,
        L0: float,
        x0: Union[Array, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        **kwargs,
    ):
        r"""

        Args:
            f: Loss or Functional object with `grad` defined.
            g: Instance of Functional with defined prox method.
            L0: Initial estimate of Lipschitz constant of f.
            x0: Starting point for :math:`\mb{x}`.
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                constant of f.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """

        #: Functional or Loss to minimize; must have grad method defined.
        self.f: Union[Loss, Functional] = f

        if g.has_prox is not True:
            raise ValueError(f"The functional g ({type(g)}) must have a prox method.")

        #: Functional to minimize; must have prox defined
        self.g: Functional = g

        if step_size is None:
            step_size = PGMStepSize()
        self.step_size: PGMStepSize = step_size
        self.step_size.internal_init(self)
        self.L: float = L0  # reciprocal of step size (estimate of Lipschitz constant of f)
        self.fixed_point_residual = snp.inf

        def x_step(v: Union[Array, BlockArray], L: float) -> Union[Array, BlockArray]:
            return self.g.prox(v - 1.0 / L * self.f.grad(v), 1.0 / L)

        self.x_step = jax.jit(x_step)

        self.x: Union[Array, BlockArray] = ensure_on_device(x0)  # current estimate of solution

        super().__init__(**kwargs)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        return snp.all(snp.isfinite(self.x))

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        return self.f.has_eval and self.g.has_eval

    def _itstat_extra_fields(self):
        """Define linearized ADMM-specific iteration statistics fields."""
        itstat_fields = {"L": "%9.3e", "Residual": "%9.3e"}
        itstat_attrib = ["L", "norm_residual()"]
        return itstat_fields, itstat_attrib

    def minimizer(self):
        """Return current estimate of the functional mimimizer."""
        return self.x

    def objective(self, x: Optional[Union[Array, BlockArray]] = None) -> float:
        r"""Evaluate the objective function :math:`f(\mb{x}) + g(\mb{x})`."""
        if x is None:
            x = self.x
        return self.f(x) + self.g(x)

    def f_quad_approx(
        self, x: Union[Array, BlockArray], y: Union[Array, BlockArray], L: float
    ) -> float:
        r"""Evaluate the quadratic approximation to function :math:`f`.

        Evaluate the quadratic approximation to function :math:`f`,
        corresponding to :math:`\hat{f}_{L}(\mb{x}, \mb{y}) = f(\mb{y}) +
        \nabla f(\mb{y})^H (\mb{x} - \mb{y}) + \frac{L}{2} \left\|\mb{x}
        - \mb{y}\right\|_2^2`.
        """
        diff_xy = x - y
        return (
            self.f(y)
            + snp.sum(snp.real(snp.conj(self.f.grad(y)) * diff_xy))
            + 0.5 * L * snp.linalg.norm(diff_xy) ** 2
        )

    def norm_residual(self) -> float:
        r"""Return the fixed point residual.

        Return the fixed point residual (see Sec. 4.3 of
        :cite:`liu-2018-first`).
        """
        return self.fixed_point_residual

    def step(self):
        """Take a single PGM step."""
        # Update reciprocal of step size using current solution.
        self.L = self.step_size.update(self.x)
        x = self.x_step(self.x, self.L)
        self.fixed_point_residual = snp.linalg.norm(self.x - x)
        self.x = x


class AcceleratedPGM(PGM):
    r"""Accelerated Proximal Gradient Method (AcceleratedPGM) base class.

    Minimize a function of the form :math:`f(\mb{x}) + g(\mb{x})`.

    Minimize a function of the form :math:`f(\mb{x}) + g(\mb{x})`, where
    :math:`f` and the :math:`g` are instances of :class:`.Functional`.
    The accelerated form of PGM is also known as FISTA
    :cite:`beck-2009-fast`.

    For documentation on inherited attributes, see :class:`.PGM`.
    """

    def __init__(
        self,
        f: Union[Loss, Functional],
        g: Functional,
        L0: float,
        x0: Union[Array, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        **kwargs,
    ):
        r"""
        Args:
            f: Loss or Functional object with `grad` defined.
            g: Instance of Functional with defined prox method.
            L0: Initial estimate of Lipschitz constant of f.
            x0: Starting point for :math:`\mb{x}`.
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                constant of f.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        x0 = ensure_on_device(x0)
        super().__init__(f=f, g=g, L0=L0, x0=x0, step_size=step_size, **kwargs)

        self.v = x0
        self.t = 1.0

    def step(self):
        """Take a single AcceleratedPGM step."""
        x_old = self.x
        # Update reciprocal of step size using current extrapolation.
        if isinstance(self.step_size, (AdaptiveBBStepSize, BBStepSize)):
            self.L = self.step_size.update(self.x)
        else:
            self.L = self.step_size.update(self.v)
        if isinstance(self.step_size, RobustLineSearchStepSize):
            # Robust line search step size uses a different extrapolation sequence.
            # Update in solution is computed while updating the reciprocal of step size.
            self.x = self.step_size.Z
            self.fixed_point_residual = snp.linalg.norm(self.x - x_old)
        else:
            self.x = self.x_step(self.v, self.L)

            self.fixed_point_residual = snp.linalg.norm(self.x - self.v)
            t_old = self.t
            self.t = 0.5 * (1 + snp.sqrt(1 + 4 * t_old**2))
            self.v = self.x + ((t_old - 1) / self.t) * (self.x - x_old)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        return snp.all(snp.isfinite(self.x)) and snp.all(snp.isfinite(self.v))
