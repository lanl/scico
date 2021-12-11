# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal Gradient Method classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import jax

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.loss import Loss
from scico.optimize.pgm import (
    AdaptiveBBStepSize,
    BBStepSize,
    PGMStepSize,
    RobustLineSearchStepSize,
)
from scico.typing import JaxArray
from scico.util import Timer, ensure_on_device

__author__ = """\n""".join(
    [
        "Luke Pfister <luke.pfister@gmail.com>",
        "Cristina Garcia-Cardona <cgarciac@lanl.gov>",
        "Thilo Balke <thilo.balke@gmail.com>",
    ]
)


class PGM:
    r"""Proximal Gradient Method (PGM) base class.

    Minimize a function of the form :math:`f(\mb{x}) + g(\mb{x})`.

    The function :math:`f` must be smooth and :math:`g` must have a
    defined prox.

    Uses helper :class:`StepSize` to provide an estimate of the Lipschitz
    constant :math:`L` of :math:`f`. The step size :math:`\alpha` is the
    reciprocal of :math:`L`, i.e.: :math:`\alpha = 1 / L`.
    """

    def __init__(
        self,
        f: Union[Loss, Functional],
        g: Functional,
        L0: float,
        x0: Union[JaxArray, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        maxiter: int = 100,
        verbose: bool = False,
        itstat: Optional[Tuple[dict, Callable]] = None,
    ):
        r"""

        Args:
            f: Loss or Functional object with `grad` defined
            g: Instance of Functional with defined prox method
            L0: Initial estimate of Lipschitz constant of f
            x0: Starting point for :math:`\mb{x}`
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                 constant of f
            maxiter: Maximum number of PGM iterations to perform.
                 Default: 100.
            verbose: Flag indicating whether iteration statistics should
                be displayed.
            itstat: A tuple (`fieldspec`, `insertfunc`), where `fieldspec`
                is a dict suitable
                for passing to the `fields` argument of the
                :class:`.diagnostics.IterationStats` initializer, and
                `insertfunc` is a function with two parameters, an
                integer and a PGM object, responsible for constructing a
                tuple ready for insertion into the
                :class:`.diagnostics.IterationStats` object. If None,
                default values are used for the tuple components.
        """

        if f.is_smooth is not True:
            raise Exception(f"The functional f ({type(f)}) must be smooth.")
        else:
            #: Functional or Loss to minimize; must have grad method defined.
            self.f: Union[Loss, Functional] = f

        if g.has_prox is not True:
            raise Exception(f"The functional g ({type(g)}) must have a proximal method.")
        else:
            #: Functional to minimize; must have prox defined
            self.g: Functional = g

        if step_size is None:
            step_size = PGMStepSize()
        self.step_size: PGMStepSize = step_size
        self.step_size.internal_init(self)
        self.L: float = L0  # reciprocal of step size (estimate of Lipschitz constant of f)
        self.itnum: int = 0
        self.maxiter: int = maxiter  # maximum number of iterations to perform
        self.timer: Timer = Timer()
        self.fixed_point_residual = snp.inf

        def x_step(v, L):
            return self.g.prox(v - 1.0 / L * self.f.grad(v), 1.0 / L)

        self.x_step = jax.jit(x_step)

        self.verbose = verbose
        if itstat:
            itstat_dict = itstat[0]
            itstat_func = itstat[1]
        elif g.has_eval:
            itstat_dict = {
                "Iter": "%d",
                "Time": "%8.2e",
                "Objective": "%8.3e",
                "L": "%8.3e",
                "Residual": "%8.3e",
            }
            itstat_func = lambda pgm: (
                pgm.itnum,
                pgm.timer.elapsed(),
                pgm.objective(self.x),
                pgm.L,
                pgm.norm_residual(),
            )
        else:
            itstat_dict = {"Iter": "%d", "Time": "%8.2e", "Residual": "%8.3e"}
            itstat_func = lambda pgm: (pgm.itnum, pgm.timer.elapsed(), pgm.norm_residual())

        self.itstat_object = IterationStats(itstat_dict, display=verbose)
        self.itstat_insert_func = itstat_func
        self.x: Union[JaxArray, BlockArray] = ensure_on_device(x0)  # current estimate of solution

    def objective(self, x) -> float:
        r"""Evaluate the objective function :math:`f(\mb{x}) + g(\mb{x})`"""
        return self.f(x) + self.g(x)

    def f_quad_approx(self, x, y, L) -> float:
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
        :cite:`liu-2018-first`)
        """
        return self.fixed_point_residual

    def step(self):
        """Take a single PGM step."""
        # Update reciprocal of step size using current solution.
        self.L = self.step_size.update(self.x)
        x = self.x_step(self.x, self.L)
        self.fixed_point_residual = snp.linalg.norm(self.x - x)
        self.x = x

    def solve(
        self,
        callback: Optional[Callable[[PGM], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        """Run the PGM algorithm.

        Run the PGM algorithm for a total of ``self.maxiter`` iterations.

        Args:
            callback: An optional callback function, taking an a single
               argument of type :class:`PGM`, that is called at the end
               of every iteration.

        Returns:
            Computed solution.
        """
        self.timer.start()
        for self.itnum in range(self.itnum, self.itnum + self.maxiter):
            self.step()
            self.itstat_object.insert(self.itstat_insert_func(self))
            if callback:
                self.timer.stop()
                callback(self)
                self.timer.start()
        self.timer.stop()
        self.itnum += 1
        return self.x


class AcceleratedPGM(PGM):
    r"""Accelerated Proximal Gradient Method (AcceleratedPGM) base class.

    Minimize a function of the form :math:`f(\mb{x}) + g(\mb{x})`.

    The function :math:`f` must be smooth and :math:`g` must have a
    defined prox. The accelerated form of PGM is also known as FISTA
    :cite:`beck-2009-fast`.

    For documentation on inherited attributes, see :class:`.PGM`.
    """

    def __init__(
        self,
        f: Union[Loss, Functional],
        g: Functional,
        L0: float,
        x0: Union[JaxArray, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        maxiter: int = 100,
        verbose: bool = False,
        itstat: Optional[Union[tuple, list]] = None,
    ):
        r"""

        Args:
            f: Loss or Functional object with `grad` defined
            g: Instance of Functional with defined prox method
            L0: Initial estimate of Lipschitz constant of f
            x0: Starting point for :math:`\mb{x}`
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                constant of f
            maxiter: Maximum number of PGM iterations to perform.
                Default: 100.
            verbose: Flag indicating whether iteration statistics should
                be displayed.
            itstat: A tuple (`fieldspec`, `insertfunc`), where `fieldspec`
                is a dict suitable for passing to the `fields` argument
                of the :class:`.diagnostics.IterationStats` initializer,
                and `insertfunc` is a function with two parameters, an
                integer and a PGM object, responsible for constructing a
                tuple ready for insertion into the
                :class:`.diagnostics.IterationStats` object. If None,
                default values are used for the tuple components.
        """
        x0 = ensure_on_device(x0)
        super().__init__(
            f=f,
            g=g,
            L0=L0,
            x0=x0,
            step_size=step_size,
            maxiter=maxiter,
            verbose=verbose,
            itstat=itstat,
        )

        self.v = x0
        self.t = 1.0

    def step(self):
        """Take a single AcceleratedPGM step."""
        x_old = self.x
        # Update reciprocal of step size using current extrapolation.
        if isinstance(self.step_size, BBStepSize) or isinstance(self.step_size, AdaptiveBBStepSize):
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
            self.t = 0.5 * (1 + snp.sqrt(1 + 4 * t_old ** 2))
            self.v = self.x + ((t_old - 1) / self.t) * (self.x - x_old)
