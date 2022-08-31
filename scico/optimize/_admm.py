# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import scico.numpy as snp
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import BlockArray
from scico.numpy.linalg import norm
from scico.numpy.util import ensure_on_device
from scico.typing import JaxArray
from scico.util import Timer

from ._admmaux import GenericSubproblemSolver, LinearSubproblemSolver, SubproblemSolver


class ADMM:
    r"""Basic Alternating Direction Method of Multipliers (ADMM) algorithm.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}) \;,

    where :math:`f` and the :math:`g_i` are instances of
    :class:`.Functional`, and the :math:`C_i` are
    :class:`.LinearOperator`.

    The optimization problem is solved by introducing the splitting
    :math:`\mb{z}_i = C_i \mb{x}` and solving

    .. math::
        \argmin_{\mb{x}, \mb{z}_i} \; f(\mb{x}) + \sum_{i=1}^N
        g_i(\mb{z}_i) \; \text{such that}\; C_i \mb{x} = \mb{z}_i \;,

    via an ADMM algorithm :cite:`glowinski-1975-approximation`
    :cite:`gabay-1976-dual` :cite:`boyd-2010-distributed` consisting of
    the iterations (see :meth:`step`)

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i
       \frac{\rho_i}{2} \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i
       \mb{x}}_2^2 \\
       \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) +
       \frac{\rho_i}{2}
       \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i \mb{x}^{(k+1)}}_2^2  \\
       \mb{u}_i^{(k+1)} &=  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} -
       \mb{z}^{(k+1)}_i  \; .
       \end{aligned}


    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
            :class:`.Loss`)
        g_list (list of :class:`.Functional`): List of :math:`g_i`
            functionals. Must be same length as :code:`C_list` and
            :code:`rho_list`.
        C_list (list of :class:`.LinearOperator`): List of :math:`C_i`
            operators.
        itnum (int): Iteration counter.
        maxiter (int): Number of ADMM outer-loop iterations.
        timer (:class:`.Timer`): Iteration timer.
        rho_list (list of scalars): List of :math:`\rho_i` penalty
            parameters. Must be same length as :code:`C_list` and
            :code:`g_list`.
        alpha (float): Relaxation parameter.
        u_list (list of array-like): List of scaled Lagrange multipliers
            :math:`\mb{u}_i` at current iteration.
        x (array-like): Solution.
        subproblem_solver (:class:`.SubproblemSolver`): Solver for
            :math:`\mb{x}`-update step.
        z_list (list of array-like): List of auxiliary variables
            :math:`\mb{z}_i` at current iteration.
        z_list_old (list of array-like): List of auxiliary variables
            :math:`\mb{z}_i` at previous iteration.
    """

    def __init__(
        self,
        f: Functional,
        g_list: List[Functional],
        C_list: List[LinearOperator],
        rho_list: List[float],
        alpha: float = 1.0,
        x0: Optional[Union[JaxArray, BlockArray]] = None,
        maxiter: int = 100,
        subproblem_solver: Optional[SubproblemSolver] = None,
        itstat_options: Optional[dict] = None,
    ):
        r"""Initialize an :class:`ADMM` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g_list: List of :math:`g_i` functionals. Must be same length
                 as :code:`C_list` and :code:`rho_list`.
            C_list: List of :math:`C_i` operators.
            rho_list: List of :math:`\rho_i` penalty parameters.
                Must be same length as :code:`C_list` and :code:`g_list`.
            alpha: Relaxation parameter. No relaxation for default 1.0.
            x0: Initial value for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            maxiter: Number of ADMM outer-loop iterations. Default: 100.
            subproblem_solver: Solver for :math:`\mb{x}`-update step.
                Defaults to ``None``, which implies use of an instance of
                :class:`GenericSubproblemSolver`.
            itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and an `ADMM` object, responsible
                for constructing a tuple ready for insertion into the
                :class:`.diagnostics.IterationStats` object. If ``None``,
                default values are used for the dict entries, otherwise
                the default dict is updated with the dict specified by
                this parameter.
        """
        N = len(g_list)
        if len(C_list) != N:
            raise Exception(f"len(C_list)={len(C_list)} not equal to len(g_list)={N}")
        if len(rho_list) != N:
            raise Exception(f"len(rho_list)={len(rho_list)} not equal to len(g_list)={N}")

        self.f: Functional = f
        self.g_list: List[Functional] = g_list
        self.C_list: List[LinearOperator] = C_list
        self.rho_list: List[float] = rho_list
        self.alpha: float = alpha
        self.itnum: int = 0
        self.maxiter: int = maxiter
        self.timer: Timer = Timer()
        if subproblem_solver is None:
            subproblem_solver = GenericSubproblemSolver()
        self.subproblem_solver: SubproblemSolver = subproblem_solver
        self.subproblem_solver.internal_init(self)

        # iteration number and time fields
        itstat_fields = {
            "Iter": "%d",
            "Time": "%8.2e",
        }
        itstat_attrib = ["itnum", "timer.elapsed()"]
        # objective function can be evaluated if all 'g' functions can be evaluated
        if all([_.has_eval for _ in self.g_list]):
            itstat_fields.update({"Objective": "%9.3e"})
            itstat_attrib.append("objective()")
        # primal and dual residual fields
        itstat_fields.update({"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"})
        itstat_attrib.extend(["norm_primal_residual()", "norm_dual_residual()"])

        # subproblem solver info when available
        if isinstance(self.subproblem_solver, GenericSubproblemSolver):
            itstat_fields.update({"Num FEv": "%6d", "Num It": "%6d"})
            itstat_attrib.extend(
                ["subproblem_solver.info['nfev']", "subproblem_solver.info['nit']"]
            )
        elif (
            type(self.subproblem_solver) == LinearSubproblemSolver
            and self.subproblem_solver.cg_function == "scico"
        ):
            itstat_fields.update({"CG It": "%5d", "CG Res": "%9.3e"})
            itstat_attrib.extend(
                ["subproblem_solver.info['num_iter']", "subproblem_solver.info['rel_res']"]
            )

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

        if x0 is None:
            input_shape = C_list[0].input_shape
            dtype = C_list[0].input_dtype
            x0 = snp.zeros(input_shape, dtype=dtype)
        self.x = ensure_on_device(x0)
        self.z_list, self.z_list_old = self.z_init(self.x)
        self.u_list = self.u_init(self.x)

    def objective(
        self,
        x: Optional[Union[JaxArray, BlockArray]] = None,
        z_list: Optional[List[Union[JaxArray, BlockArray]]] = None,
    ) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(\mb{z}_i) \;.

        Args:
            x: Point at which to evaluate objective function. If ``None``,
                the objective is  evaluated at the current iterate
                :code:`self.x`.
            z_list: Point at which to evaluate objective function. If
                ``None``, the objective is evaluated at the current iterate
                :code:`self.z_list`.

        Returns:
            scalar: Value of the objective function.
        """
        if (x is None) != (z_list is None):
            raise ValueError("Both or neither of x and z_list must be supplied")
        if x is None:
            x = self.x
            z_list = self.z_list
        assert z_list is not None
        out = 0.0
        if self.f:
            out += self.f(x)
        for g, z in zip(self.g_list, z_list):
            out += g(z)
        return out

    def norm_primal_residual(self, x: Optional[Union[JaxArray, BlockArray]] = None) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \left(\sum_{i=1}^N \norm{C_i \mb{x} -
            \mb{z}_i}_2^2\right)^{1/2} \;.

        Args:
            x: Point at which to evaluate primal residual.
                If ``None``, the primal residual is evaluated at the
                current iterate :code:`self.x`.

        Returns:
            Norm of primal residual.
        """
        if x is None:
            x = self.x

        out = 0.0
        for Ci, zi in zip(self.C_list, self.z_list):
            out += norm(Ci(self.x) - zi) ** 2
        return snp.sqrt(out)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \left(\sum_{i=1}^N \norm{\mb{z}^{(k)}_i -
            \mb{z}^{(k-1)}_i}_2^2\right)^{1/2} \;.

        Returns:
            Current norm of dual residual.

        """
        out = 0.0
        for zi, ziold, Ci in zip(self.z_list, self.z_list_old, self.C_list):
            out += norm(Ci.adj(zi - ziold)) ** 2
        return snp.sqrt(out)

    def z_init(
        self, x0: Union[JaxArray, BlockArray]
    ) -> Tuple[List[Union[JaxArray, BlockArray]], List[Union[JaxArray, BlockArray]]]:
        r"""Initialize auxiliary variables :math:`\mb{z}_i`.

        Initialized to

        .. math::
            \mb{z}_i = C_i \mb{x}^{(0)} \;.

        :code:`z_list` and :code:`z_list_old` are initialized to the same
        value.

        Args:
            x0: Initial value of :math:`\mb{x}`.
        """
        z_list: List[Union[JaxArray, BlockArray]] = [Ci(x0) for Ci in self.C_list]
        z_list_old = z_list.copy()
        return z_list, z_list_old

    def u_init(self, x0: Union[JaxArray, BlockArray]) -> List[Union[JaxArray, BlockArray]]:
        r"""Initialize scaled Lagrange multipliers :math:`\mb{u}_i`.

        Initialized to

        .. math::
            \mb{u}_i = \mb{0} \;.

        Note that the parameter `x0` is unused, but is provided for
        potential use in an overridden method.

        Args:
            x0: Initial value of :math:`\mb{x}`.
        """
        u_list = [snp.zeros(Ci.output_shape, dtype=Ci.output_dtype) for Ci in self.C_list]
        return u_list

    def step(self):
        r"""Perform a single ADMM iteration.

        The primary variable :math:`\mb{x}` is updated by solving the the
        optimization problem

        .. math::
            \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i
            \frac{\rho_i}{2} \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i -
            C_i \mb{x}}_2^2 \;.

        Update auxiliary variables :math:`\mb{z}_i` and scaled Lagrange
        multipliers :math:`\mb{u}_i`. The auxiliary variables are updated
        according to

        .. math::
            \begin{aligned}
            \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) +
            \frac{\rho_i}{2} \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i
            \mb{x}^{(k+1)}}_2^2  \\
            &= \mathrm{prox}_{g_i}(C_i \mb{x} + \mb{u}_i, 1 / \rho_i) \;,
            \end{aligned}

        and the scaled Lagrange multipliers are updated according to

        .. math::
            \mb{u}_i^{(k+1)} =  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} -
            \mb{z}^{(k+1)}_i \;.
        """

        self.x = self.subproblem_solver.solve(self.x)

        self.z_list_old = self.z_list.copy()

        for i, (rhoi, gi, Ci, zi, ui) in enumerate(
            zip(self.rho_list, self.g_list, self.C_list, self.z_list, self.u_list)
        ):
            if self.alpha == 1.0:
                Cix = Ci(self.x)
            else:
                Cix = self.alpha * Ci(self.x) + (1.0 - self.alpha) * zi
            zi = gi.prox(Cix + ui, 1 / rhoi, v0=zi)
            ui = ui + Cix - zi
            self.z_list[i] = zi
            self.u_list[i] = ui

    def solve(
        self,
        callback: Optional[Callable[[ADMM], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        """Run the ADMM algorithm.

        Run the ADMM algorithm for a total of `self.maxiter` iterations.

        Args:
            callback: An optional callback function, taking an a single
               argument of type :class:`ADMM`, that is called at the end
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
