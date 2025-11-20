# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm

from ._admmaux import (
    FBlockCircularConvolveSolver,
    G0BlockCircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
    MatrixSubproblemSolver,
    SubproblemSolver,
)
from ._common import Optimizer


class ADMM(Optimizer):
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
        x0: Optional[Union[Array, BlockArray]] = None,
        subproblem_solver: Optional[SubproblemSolver] = None,
        **kwargs,
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
            subproblem_solver: Solver for :math:`\mb{x}`-update step.
                Defaults to ``None``, which implies use of an instance of
                :class:`GenericSubproblemSolver`.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        N = len(g_list)
        if len(C_list) != N:
            raise ValueError(f"len(C_list)={len(C_list)} not equal to len(g_list)={N}.")
        if len(rho_list) != N:
            raise ValueError(f"len(rho_list)={len(rho_list)} not equal to len(g_list)={N}.")

        self.f: Functional = f
        self.g_list: List[Functional] = g_list
        self.C_list: List[LinearOperator] = C_list
        self.rho_list: List[float] = rho_list
        self.alpha: float = alpha

        if subproblem_solver is None:
            subproblem_solver = GenericSubproblemSolver()
        self.subproblem_solver: SubproblemSolver = subproblem_solver
        self.subproblem_solver.internal_init(self)

        if x0 is None:
            input_shape = C_list[0].input_shape
            dtype = C_list[0].input_dtype
            x0 = snp.zeros(input_shape, dtype=dtype)
        self.x = x0
        self.z_list, self.z_list_old = self.z_init(self.x)
        self.u_list = self.u_init(self.x)

        super().__init__(**kwargs)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        for v in (
            [
                self.x,
            ]
            + self.z_list
            + self.u_list
        ):
            if not snp.all(snp.isfinite(v)):
                return False
        return True

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        return (not self.f or self.f.has_eval) and all([_.has_eval for _ in self.g_list])

    def _itstat_extra_fields(self):
        """Define ADMM-specific iteration statistics fields."""
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()"]

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
        elif (
            type(self.subproblem_solver)
            in [MatrixSubproblemSolver, FBlockCircularConvolveSolver, G0BlockCircularConvolveSolver]
            and self.subproblem_solver.check_solve
        ):
            itstat_fields.update({"Slv Res": "%9.3e"})
            itstat_attrib.extend(["subproblem_solver.accuracy"])

        return itstat_fields, itstat_attrib

    def _state_variable_names(self) -> List[str]:
        # While x is in the most abstract sense not part of the algorithm
        # state, it does form part of the state in pratice due to its use
        # as an initializer for iterative solvers for the x step of the
        # ADMM algorithm.
        return ["x", "z_list", "z_list_old", "u_list"]

    def minimizer(self) -> Union[Array, BlockArray]:
        return self.x

    def objective(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
        z_list: Optional[List[Union[Array, BlockArray]]] = None,
    ) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(\mb{z}_i) \;.

        Note that this form is cheaper to compute, but may have very poor
        accuracy compared with the "true" objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}) \;.

        when the primal residual is large.

        Args:
            x: Point at which to evaluate objective function. If ``None``,
                the objective is  evaluated at the current iterate
                :code:`self.x`.
            z_list: Point at which to evaluate objective function. If
                ``None``, the objective is evaluated at the current iterate
                :code:`self.z_list`.

        Returns:
            Value of the objective function.
        """
        if (x is None) != (z_list is None):
            raise ValueError("Both or neither of arguments 'x' and 'z_list' must be supplied.")
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

    def norm_primal_residual(self, x: Optional[Union[Array, BlockArray]] = None) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \left( \sum_{i=1}^N \rho_i \left\| C_i \mb{x} -
            \mb{z}_i^{(k)} \right\|_2^2\right)^{1/2} \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
                the primal residual is evaluated at the current iterate
                :code:`self.x`.

        Returns:
            Norm of primal residual.
        """
        if x is None:
            x = self.x

        sum = 0.0
        for rhoi, Ci, zi in zip(self.rho_list, self.C_list, self.z_list):
            sum += rhoi * norm(Ci(self.x) - zi) ** 2
        return snp.sqrt(sum)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \left\| \sum_{i=1}^N \rho_i C_i^T \left( \mb{z}^{(k)}_i -
            \mb{z}^{(k-1)}_i \right) \right\|_2 \;.

        Returns:
            Norm of dual residual.

        """
        sum = 0.0
        for rhoi, zi, ziold, Ci in zip(self.rho_list, self.z_list, self.z_list_old, self.C_list):
            sum += rhoi * Ci.adj(zi - ziold)
        return norm(sum)

    def z_init(
        self, x0: Union[Array, BlockArray]
    ) -> Tuple[List[Union[Array, BlockArray]], List[Union[Array, BlockArray]]]:
        r"""Initialize auxiliary variables :math:`\mb{z}_i`.

        Initialized to

        .. math::
            \mb{z}_i = C_i \mb{x}^{(0)} \;.

        :code:`z_list` and :code:`z_list_old` are initialized to the same
        value.

        Args:
            x0: Initial value of :math:`\mb{x}`.
        """
        z_list: List[Union[Array, BlockArray]] = [Ci(x0) for Ci in self.C_list]
        z_list_old = z_list.copy()
        return z_list, z_list_old

    def u_init(self, x0: Union[Array, BlockArray]) -> List[Union[Array, BlockArray]]:
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
