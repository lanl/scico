# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Non-linear Proximal ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, List, Optional, Union

from scico import jhvp, jvp
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.numpy import BlockArray
from scico.numpy.linalg import norm
from scico.numpy.util import ensure_on_device
from scico.typing import JaxArray
from scico.util import Timer


class NonLinearPADMM:
    r"""Non-linear proximal alternating direction method of multipliers.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + g(\mb{z}) \;
        \text{such that}\; H(\mb{x}, \mb{z}) = 0 \;,

    where :math:`f` and :math:`g` are instances of :class:`.Functional`,
    (in most cases :math:`f` will, more specifically be an an instance
    of :class:`.Loss`), and :math:`H` is a function.

    The optimization problem is solved via a variant of the proximal ADMM
    algorithm for problems with a non-linear operator constraint
    :cite:`benning-2016-preconditioned`, consisting of the
    iterations (see :meth:`step`)

    .. math::
       \begin{aligned}
       A^{(k)} &= J_{\mb{x}} H(\mb{x}^{(k)}, \mb{z}^{(k)}) \\
       \mb{x}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \mu^{-1} f} \left(
         \mb{x}^{(k)} - \mu^{-1} (A^{(k)})^T \left(2 \mb{u}^{(k)} -
         \mb{u}^{(k-1)} \right) \right) \\
       B^{(k)} &= J_{\mb{z}} H(\mb{x}^{(k+1)}, \mb{z}^{(k)}) \\
       \mb{z}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \nu^{-1} g} \left(
         \mb{z}^{(k)} - \nu^{-1} (B^{(k)})^T \left(
         H(\mb{x}^{(k+1)}, \mb{z}^{(k)}) + \mb{u}^{(k)} \right) \right) \\
       \mb{u}^{(k+1)} &=  \mb{u}^{(k)} + H(\mb{x}^{(k+1)},
         \mb{z}^{(k+1)})  \;.
       \end{aligned}

    Parameters :math:`\mu` and :math:`\nu` are required to satisfy

    .. math::
       \mu > \norm{ A^{(k)} }_2^2 \quad \text{and} \quad \nu > \norm{ B^{(k)} }_2^2

    for all :math:`A^{(k)}` and :math:`B^{(k)}`.


    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
           :class:`.Loss`).
        g (:class:`.Functional`): Functional :math:`g`.
        H (Callable): :math:`H` function.
        itnum (int): Iteration counter.
        maxiter (int): Number of linearized ADMM outer-loop iterations.
        timer (:class:`.Timer`): Iteration timer.
        rho (scalar): Penalty parameter.
        mu (scalar): First algorithm parameter.
        nu (scalar): Second algorithm parameter.
        x (array-like): Solution variable.
        z (array-like): Auxiliary variables :math:`\mb{z}` at current
          iteration.
        z_old (array-like): Auxiliary variables :math:`\mb{z}` at
          previous iteration.
        u (array-like): Scaled Lagrange multipliers :math:`\mb{u}` at
           current iteration.
        u_old (array-like): Scaled Lagrange multipliers :math:`\mb{u}` at
           previous iteration.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        H: Callable,
        rho: float,
        mu: float,
        nu: float,
        x0: Optional[Union[JaxArray, BlockArray]] = None,
        z0: Optional[Union[JaxArray, BlockArray]] = None,
        u0: Optional[Union[JaxArray, BlockArray]] = None,
        maxiter: int = 100,
        fast_dual_residual: bool = True,
        itstat_options: Optional[dict] = None,
    ):
        r"""Initialize a :class:`NonLinearPADMM` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            H: Function :math:`H`.
            rho: Penalty parameter.
            mu: First algorithm parameter.
            nu: Second algorithm parameter.
            x0: Starting point for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            z0: Starting point for :math:`\mb{z}`. If ``None``, defaults
                to an array of zeros.
            u0: Starting point for :math:`\mb{u}`. If ``None``, defaults
                to an array of zeros.
            maxiter: Number of main algorithm iterations. Default: 100.
            fast_dual_residual: Flag indicating whether to use fast
                approximation to the dual residual, or a slower but more
                accurate calculation.
            itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and a :class:`NonLinearPADMM`
                object, responsible for constructing a tuple ready for
                insertion into the :class:`.diagnostics.IterationStats`
                object. If ``None``, default values are used for the dict
                entries, otherwise the default dict is updated with the
                dict specified by this parameter.
        """
        self.f: Functional = f
        self.g: Functional = g
        self.H: Callable = H
        self.rho: float = rho
        self.mu: float = mu
        self.nu: float = nu
        self.itnum: int = 0
        self.maxiter: int = maxiter
        self.fast_dual_residual: bool = fast_dual_residual
        self.timer: Timer = Timer()

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
        # primal and dual residual fields
        itstat_fields.update({"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"})
        itstat_attrib.extend(["norm_primal_residual()", "norm_dual_residual()"])

        # dynamically create itstat_func; see https://stackoverflow.com/questions/24733831
        itstat_return = "return(" + ", ".join(["obj." + attr for attr in itstat_attrib]) + ")"
        scope: dict[str, Callable] = {}
        exec("def itstat_func(obj): " + itstat_return, scope)

        # determine itstat options and initialize IterationStats object
        default_itstat_options = {
            "fields": itstat_fields,
            "itstat_func": scope["itstat_func"],
            "display": False,
        }
        if itstat_options:
            default_itstat_options.update(itstat_options)
        self.itstat_insert_func: Callable = default_itstat_options.pop("itstat_func", None)  # type: ignore
        self.itstat_object = IterationStats(**default_itstat_options)  # type: ignore

        self.x = ensure_on_device(x0)
        self.z = ensure_on_device(z0)
        self.z_old = self.z
        self.u = ensure_on_device(u0)
        self.u_old = self.u

    def objective(
        self,
        x: Optional[Union[JaxArray, BlockArray]] = None,
        z: Optional[List[Union[JaxArray, BlockArray]]] = None,
    ) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + g(\mb{z}) \;.


        Args:
            x: Point at which to evaluate objective function. If
               ``None``, the objective is evaluated at the current
               iterate :code:`self.x`.
            z: Point at which to evaluate objective function. If
               ``None``, the objective is evaluated at the current
               iterate :code:`self.z`.

        Returns:
            scalar: Current value of the objective function.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of x and z must be supplied")
        if x is None:
            x = self.x
            z = self.z
        out = 0.0
        if self.f:
            out += self.f(x)
        out += self.g(z)
        return out

    def norm_primal_residual(
        self,
        x: Optional[Union[JaxArray, BlockArray]] = None,
        z: Optional[List[Union[JaxArray, BlockArray]]] = None,
    ) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \norm{H(\mb{x}, \mb{z})}_2 \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.x`.
            z: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.z`.

        Returns:
            Norm of primal residual.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of x and z must be supplied")
        if x is None:
            x = self.x
            z = self.z

        return norm(self.H(x, z))

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual. If the flag
        requesting a fast approximate calculation is set, it is computed
        as

        .. math::
            \norm{\mb{z}^{(k+1)} - \mb{z}^{(k)}}_2 \;,

        otherwise it is computed as

        .. math::
            \norm{A^T B ( \mb{z}^{(k+1)} - \mb{z}^{(k)} ) }_2 \;,

        where

        .. math::
            A &= J_{\mb{x}} H(\mb{x}^{(k+1)}, \mb{z}^{(k+1)}) \\
            B &= J_{\mb{z}} H(\mb{x}^{(k+1)}, \mb{z}^{(k+1)}) \;.

        Returns:
            Current norm of dual residual.
        """
        if self.fast_dual_residual:
            rsdl = self.z - self.z_old  # fast but poor approximation
        else:
            Hz = lambda z: self.H(self.x, z)
            B = lambda u: jvp(Hz, (self.z,), (u,))[1]
            Hx = lambda x: self.H(x, self.z)
            AH = jhvp(Hx, self.x)[1]
            rsdl = AH(B(self.z - self.z_old))
        return norm(rsdl)

    def step(self):
        r"""Perform a single algorithm iteration.

        Perform a single algorithm iteration.
        """
        _, AH = jhvp(self.H, self.x, self.z, jidx=0)
        proxarg = self.x - (1.0 / self.mu) * AH(2.0 * self.u - self.u_old)[0]
        self.x = self.f.prox(proxarg, (1.0 / (self.rho * self.mu)), v0=self.x)
        _, BH = jhvp(self.H, self.x, self.z, jidx=1)
        proxarg = self.z - (1.0 / self.nu) * BH(self.H(self.x, self.z) + self.u)[0]
        self.z_old = self.z
        self.z = self.g.prox(proxarg, (1.0 / (self.rho * self.nu)), v0=self.z)
        self.u_old = self.u
        self.u = self.u + self.H(self.x, self.z)

    def solve(
        self,
        callback: Optional[Callable[[NonLinearPADMM], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        r"""Initialize and run the optimization algorithm.

        Initialize and run the opimization algorithm for a total of
        `self.maxiter` iterations.

        Args:
            callback: An optional callback function, taking an a single
              argument of type :class:`NonLinearPADMM`, that is called
              at the end of every iteration.

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
