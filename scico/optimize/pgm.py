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
from scico.array import ensure_on_device
from scico.blockarray import BlockArray
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.loss import Loss
from scico.typing import JaxArray
from scico.util import Timer


class PGMStepSize:
    r"""Base class for computing the PGM step size.

    Base class for computing the reciprocal of the step size for PGM
    solvers.

    The PGM solver implemented by :class:`.PGM` addresses a general
    proximal gradient form that requires the specification of a step size
    for the gradient descent step. This class is a base class for methods
    that estimate the reciprocal of the step size (:math:`L` in PGM
    equations).

    Attributes:
        pgm (:class:`.PGM`): PGM solver object to which the solver is
           attached.
    """

    def internal_init(self, pgm: PGM):
        """Second stage initializer to be called by :meth:`.PGM.__init__`.

        Args:
            pgm: Reference to :class:`.PGM` object to which the
              :class:`.StepSize` object is to be attached.
        """
        self.pgm = pgm

    def update(self, v: Union[JaxArray, BlockArray]) -> float:
        """Hook for updating the step size in derived classes.

        Hook for updating the reciprocal of the step size in derived
        classes. The base class does not compute any update.

        Args:
            v: Current solution or current extrapolation (if accelerated
               PGM).

        Returns:
            Current reciprocal of the step size.
        """
        return self.pgm.L


class BBStepSize(PGMStepSize):
    r"""Scheme for step size estimation based on Barzilai-Borwein method.

    The Barzilai-Borwein method :cite:`barzilai-1988-stepsize` estimates
    the step size :math:`\alpha` as

    .. math::
       \mb{\Delta x} = \mb{x}_k - \mb{x}_{k-1} \; \\
       \mb{\Delta g} = \nabla f(\mb{x}_k) - \nabla f (\mb{x}_{k-1}) \; \\
       \alpha = \frac{\mb{\Delta x}^T \mb{\Delta g}}{\mb{\Delta g}^T
       \mb{\Delta g}} \;\;.

    Since the PGM solver uses the reciprocal of the step size, the value
    :math:`L = 1 / \alpha` is returned.

    When applied to complex-valued problems, only the real part of the
    inner product is used. When the inner product is negative, the
    previous iterate is used instead.

    Attributes:
        pgm (:class:`.PGM`): PGM solver object to which the solver is
           attached.
    """

    def __init__(self):
        """Initialize a :class:`BBStepSize` object."""
        self.xprev: Union[JaxArray, BlockArray] = None
        self.gradprev: Union[JaxArray, BlockArray] = None

    def update(self, v: Union[JaxArray, BlockArray]) -> float:
        """Update the reciprocal of the step size.

        Args:
            v: Current solution or current extrapolation (if accelerated
               PGM).

        Returns:
            Updated reciprocal of the step size.
        """

        if self.xprev is None:
            # Solution and gradient of previous iterate are required.
            # For first iteration these variables are stored and current estimate is returned.
            self.xprev = v
            self.gradprev = self.pgm.f.grad(self.xprev)
            L = self.pgm.L
        else:
            Δx = v - self.xprev
            gradv = self.pgm.f.grad(v)
            Δg = gradv - self.gradprev
            # Taking real part of inner products in case of complex-value problem.
            den = snp.real(snp.vdot(Δx, Δg))
            num = snp.real(snp.vdot(Δg, Δg))
            L = num / den
            # Revert to previous iterate if update results in nan or negative value.
            if snp.isnan(L) or L <= 0.0:
                L = self.pgm.L
            # Store current state and gradient for next update.
            self.xprev = v
            self.gradprev = gradv
        return L


class AdaptiveBBStepSize(PGMStepSize):
    r"""Adaptive Barzilai-Borwein method to determine step size.

    Adaptive Barzilai-Borwein method to determine step size in PGM, as
    introduced in :cite:`zhou-2006-adaptive`.

    The adaptive step size rule computes

    .. math::

       \mb{\Delta x} = \mb{x}_k - \mb{x}_{k-1} \; \\
       \mb{\Delta g} = \nabla f(\mb{x}_k) - \nabla f (\mb{x}_{k-1}) \; \\
       \alpha^{\mathrm{BB1}} = \frac{\mb{\Delta x}^T \mb{\Delta x}}
       {\mb{\Delta x}^T \mb{\Delta g}} \; \\
       \alpha^{\mathrm{BB2}} = \frac{\mb{\Delta x}^T \mb{\Delta g}}
       {\mb{\Delta g}^T \mb{\Delta g}} \;\;.

    The determination of the new steps size is made via the rule

    .. math::

        \alpha = \left\{ \begin{matrix} \alpha^{\mathrm{BB2}}  &
        \mathrm{~if~} \alpha^{\mathrm{BB2}} / \alpha^{\mathrm{BB1}}
        < \kappa \; \\
        \alpha^{\mathrm{BB1}} & \mathrm{~otherwise} \end{matrix}
        \right . \;,

    with :math:`\kappa \in (0, 1)`.

    Since the PGM solver uses the reciprocal of the step size, the value
    :math:`L = 1 / \alpha` is returned.

    When applied to complex-valued problems, only the real part of the
    inner product is used. When the inner product is negative, the
    previous iterate is used instead.

    Attributes:
        pgm (:class:`.PGM`): PGM solver object to which the solver is
           attached.
    """

    def __init__(self, kappa: float = 0.5):
        r"""Initialize a :class:`AdaptiveBBStepSize` object.

        Args:
            kappa : Threshold for step size selection :math:`\kappa`.
        """
        self.kappa: float = kappa
        self.xprev: Union[JaxArray, BlockArray] = None
        self.gradprev: Union[JaxArray, BlockArray] = None
        self.Lbb1prev: float = None
        self.Lbb2prev: float = None

    def update(self, v: Union[JaxArray, BlockArray]) -> float:
        """Update the reciprocal of the step size.

        Args:
            v: Current solution or current extrapolation (if accelerated
               PGM).

        Returns:
            Updated reciprocal of the step size.
        """

        if self.xprev is None:
            # Solution and gradient of previous iterate are required.
            # For first iteration these variables are stored and current estimate is returned.
            self.xprev = v
            self.gradprev = self.pgm.f.grad(self.xprev)
            L = self.pgm.L
        else:
            Δx = v - self.xprev
            gradv = self.pgm.f.grad(v)
            Δg = gradv - self.gradprev
            # Taking real part of inner products in case of complex-value problem.
            innerxx = snp.real(snp.vdot(Δx, Δx))
            innerxg = snp.real(snp.vdot(Δx, Δg))
            innergg = snp.real(snp.vdot(Δg, Δg))
            Lbb1 = innerxg / innerxx
            # Revert to previous iterate if computation results in nan or negative value.
            if snp.isnan(Lbb1) or Lbb1 <= 0.0:
                Lbb1 = self.Lbb1prev
            Lbb2 = innergg / innerxg
            # Revert to previous iterate if computation results in nan or negative value.
            if snp.isnan(Lbb2) or Lbb2 <= 0.0:
                Lbb2 = self.Lbb2prev
            # If possible, apply adaptive selection rule, if not, revert to previous iterate
            if Lbb1 is not None and Lbb2 is not None:
                if (Lbb1 / Lbb2) < self.kappa:
                    L = Lbb2
                else:
                    L = Lbb1
            else:
                L = self.pgm.L
            # Store current state and gradient for next update.
            self.xprev = v
            self.gradprev = gradv
            # Store current estimates of Barzilai-Borwein 1 (Lbb1) and Barzilai-Borwein 2 (Lbb2).
            self.Lbb1prev = Lbb1
            self.Lbb2prev = Lbb2

        return L


class LineSearchStepSize(PGMStepSize):
    r"""Line search for estimating the step size for PGM solvers.

    Line search for estimating the reciprocal of step size for PGM
    solvers. The line search strategy described in :cite:`beck-2009-fast`
    estimates :math:`L` such that :math:`f(\mb{x}) <= \hat{f}_{L}(\mb{x})`
    is satisfied with :math:`\hat{f}_{L}` a quadratic approximation to
    :math:`f` defined as

    .. math::
       \hat{f}_{L}(\mb{x}, \mb{y}) = f(\mb{y}) + \nabla f(\mb{y})^H
       (\mb{x} - \mb{y}) + \frac{L}{2} \left\| \mb{x} - \mb{y}
       \right\|_2^2 \;,

    with :math:`\mb{x}` the potential new update and :math:`\mb{y}` the
    current solution or current extrapolation (if accelerated PGM).

    Attributes:
        pgm (:class:`.PGM`): PGM solver object to which the solver is
           attached.
    """

    def __init__(self, gamma_u: float = 1.2, maxiter: int = 50):
        r"""Initialize a :class:`LineSearchStepSize` object.

        Args:
            gamma_u: Rate of increment in :math:`L`.
            maxiter: maximum iterations in line search.
        """
        self.gamma_u: float = gamma_u
        self.maxiter: int = maxiter

        def g_prox(v, gradv, L):
            return self.pgm.g.prox(v - 1.0 / L * gradv, 1.0 / L)

        self.g_prox = jax.jit(g_prox)

    def update(self, v: Union[JaxArray, BlockArray]) -> float:
        """Update the reciprocal of the step size.

        Args:
            v: Current solution or current extrapolation (if accelerated
               PGM).

        Returns:
            Updated reciprocal of the step size.
        """

        gradv = self.pgm.f.grad(v)
        L = self.pgm.L
        it = 0
        while it < self.maxiter:
            z = self.g_prox(v, gradv, L)
            fz = self.pgm.f(z)
            fquad = self.pgm.f_quad_approx(z, v, L)
            if fz <= fquad:
                break
            else:
                L *= self.gamma_u
            it += 1
        return L


class RobustLineSearchStepSize(LineSearchStepSize):
    r"""Robust line search for estimating the accelerated PGM step size.

    A robust line search for estimating the reciprocal of step size for
    accelerated PGM solvers.

    The robust line search strategy described in :cite:`florea-2017-robust`
    estimates :math:`L` such that :math:`f(\mb{x}) <= \hat{f}_{L}(\mb{x})`
    is satisfied with :math:`\hat{f}_{L}` a quadratic approximation to
    :math:`f` defined as

    .. math::
       \hat{f}_{L}(\mb{x}, \mb{y}) = f(\mb{y}) + \nabla f(\mb{y})^H
       (\mb{x} - \mb{y}) + \frac{L}{2} \left\| \mb{x} - \mb{y}
       \right\|_2^2 \;,

    with :math:`\mb{x}` the potential new update and :math:`\mb{y}` the
    auxiliary extrapolation state.

    Attributes:
        pgm (:class:`.PGM`): PGM solver object to which the solver is
           attached.
    """

    def __init__(self, gamma_d: float = 0.9, gamma_u: float = 2.0, maxiter: int = 50):
        r"""Initialize a :class:`RobustLineSearchStepSize` object.

        Args:
            gamma_d: Rate of decrement in :math:`L`.
            gamma_u: Rate of increment in :math:`L`.
            maxiter: maximum iterations in line search.
        """
        super(RobustLineSearchStepSize, self).__init__(gamma_u, maxiter)
        self.gamma_d: float = gamma_d
        self.Tk: float = 0.0
        # State needed for computing auxiliary extrapolation sequence in robust line search.
        self.Zrb: Union[JaxArray, BlockArray] = None
        #: Current estimate of solution in robust line search.
        self.Z: Union[JaxArray, BlockArray] = None

    def update(self, v: Union[JaxArray, BlockArray]) -> float:
        """Update the reciprocal of the step size.

        Args:
            v: Current solution or current extrapolation (if accelerated
               PGM).

        Returns:
            Updated reciprocal of the step size.
        """
        if self.Zrb is None:
            self.Zrb = self.pgm.x

        L = self.pgm.L * self.gamma_d

        it = 0
        while it < self.maxiter:
            t = (1.0 + snp.sqrt(1.0 + 4.0 * L * self.Tk)) / (2.0 * L)
            T = self.Tk + t
            # Auxiliary extrapolation sequence.
            y = (self.Tk * self.pgm.x + t * self.Zrb) / T
            # New update based on auxiliary extrapolation and current L estimate.
            z = self.pgm.x_step(y, L)
            fz = self.pgm.f(z)
            fquad = self.pgm.f_quad_approx(z, y, L)
            if fz <= fquad:
                break
            else:
                L *= self.gamma_u
            it += 1
        self.Tk = T
        self.Zrb += t * L * (z - y)
        self.Z = z

        return L


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
        self.itstat_object = IterationStats(**default_itstat_options)

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
                parameters, an integer and a PGM object, responsible
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
            self.t = 0.5 * (1 + snp.sqrt(1 + 4 * t_old ** 2))
            self.v = self.x + ((t_old - 1) / self.t) * (self.x - x_old)
