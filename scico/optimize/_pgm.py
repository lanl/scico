# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal Gradient Method classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, Optional, Union

import jax

import scico.numpy as snp
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.loss import Loss
from scico.numpy import BlockArray
from scico.numpy.util import ensure_on_device
from scico.typing import JaxArray
from scico.util import Timer

from ._pgmaux import (
    AdaptiveBBStepSize,
    BBStepSize,
    PGMStepSize,
    RobustLineSearchStepSize,
)


class PGM:
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
        x0: Union[JaxArray, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        maxiter: int = 100,
        itstat_options: Optional[dict] = None,
    ):
        r"""

        Args:
            f: Loss or Functional object with `grad` defined.
            g: Instance of Functional with defined prox method.
            L0: Initial estimate of Lipschitz constant of f.
            x0: Starting point for :math:`\mb{x}`.
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                constant of f.
            maxiter: Maximum number of PGM iterations to perform.
                Default: 100.
            itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and a PGM object, responsible
                for constructing a tuple ready for insertion into the
                :class:`.diagnostics.IterationStats` object. If ``None``,
                default values are used for the dict entries, otherwise
                the default dict is updated with the dict specified by
                this parameter.
        """

        #: Functional or Loss to minimize; must have grad method defined.
        self.f: Union[Loss, Functional] = f

        if g.has_prox is not True:
            raise Exception(f"The functional g ({type(g)}) must have a proximal method.")

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

        def x_step(v: Union[JaxArray, BlockArray], L: float) -> Union[JaxArray, BlockArray]:
            return self.g.prox(v - 1.0 / L * self.f.grad(v), 1.0 / L)

        self.x_step = jax.jit(x_step)

        # iteration number and time fields
        itstat_fields = {
            "Iter": "%d",
            "Time": "%8.2e",
        }
        itstat_attrib = ["itnum", "timer.elapsed()"]
        # objective function can be evaluated if 'g' function can be evaluated
        if g.has_eval:
            itstat_fields.update({"Objective": "%9.3e"})
            itstat_attrib.append("objective()")
        # step size and residual fields
        itstat_fields.update({"L": "%9.3e", "Residual": "%9.3e"})
        itstat_attrib.extend(["L", "norm_residual()"])

        # dynamically create itstat_func; see https://stackoverflow.com/questions/24733831
        itstat_return = "return(" + ", ".join(["obj." + attr for attr in itstat_attrib]) + ")"
        scope: dict[str, Callable] = {}
        exec("def itstat_func(obj): " + itstat_return, scope)

        default_itstat_options: dict[str, Union[dict, Callable, bool]] = {
            "fields": itstat_fields,
            "itstat_func": scope["itstat_func"],
            "display": False,
        }
        if itstat_options:
            default_itstat_options.update(itstat_options)
        self.itstat_insert_func: Callable = default_itstat_options.pop("itstat_func")  # type: ignore
        self.itstat_object = IterationStats(**default_itstat_options)  # type: ignore

        self.x: Union[JaxArray, BlockArray] = ensure_on_device(x0)  # current estimate of solution

    def objective(self, x: Optional[Union[JaxArray, BlockArray]] = None) -> float:
        r"""Evaluate the objective function :math:`f(\mb{x}) + g(\mb{x})`."""
        if x is None:
            x = self.x
        return self.f(x) + self.g(x)

    def f_quad_approx(
        self, x: Union[JaxArray, BlockArray], y: Union[JaxArray, BlockArray], L: float
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

    def solve(
        self,
        callback: Optional[Callable[[PGM], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        """Run the PGM algorithm.

        Run the PGM algorithm for a total of `self.maxiter` iterations.

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
        self.itstat_object.end()
        return self.x


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
        x0: Union[JaxArray, BlockArray],
        step_size: Optional[PGMStepSize] = None,
        maxiter: int = 100,
        itstat_options: Optional[dict] = None,
    ):
        r"""

        Args:
            f: Loss or Functional object with `grad` defined.
            g: Instance of Functional with defined prox method.
            L0: Initial estimate of Lipschitz constant of f.
            x0: Starting point for :math:`\mb{x}`.
            step_size: helper :class:`StepSize` to estimate the Lipschitz
                constant of f.
            maxiter: Maximum number of PGM iterations to perform.
                Default: 100.
            itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and a `PGM` object, responsible
                for constructing a tuple ready for insertion into the
                :class:`.diagnostics.IterationStats` object. If ``None``,
                default values are used for the dict entries, otherwise
                the default dict is updated with the dict specified by
                this parameter.
        """
        x0 = ensure_on_device(x0)
        super().__init__(
            f=f,
            g=g,
            L0=L0,
            x0=x0,
            step_size=step_size,
            maxiter=maxiter,
            itstat_options=itstat_options,
        )

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
