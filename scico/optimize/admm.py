# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM solver and auxiliary classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import reduce
from typing import Callable, List, Optional, Union

import jax
from jax.scipy.sparse.linalg import cg as jax_cg

import scico.numpy as snp
from scico.array import ensure_on_device, is_real_dtype
from scico.blockarray import BlockArray
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.linop import CircularConvolve, Identity, LinearOperator
from scico.loss import SquaredL2Loss, WeightedSquaredL2Loss
from scico.numpy.linalg import norm
from scico.solver import cg as scico_cg
from scico.solver import minimize
from scico.typing import JaxArray
from scico.util import Timer


class SubproblemSolver:
    r"""Base class for solvers for the non-separable ADMM step.

    The ADMM solver implemented by :class:`.ADMM` addresses a general
    problem form for which one of the corresponding ADMM algorithm
    subproblems is separable into distinct subproblems for each of the
    :math:`g_i`, and another that is non-separable, involving function
    :math:`f` and a sum over :math:`\ell_2` norm terms involving all
    operators :math:`C_i`. This class is a base class for solvers of
    the latter subproblem

    ..  math::

        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the
            solver is attached.
    """

    def internal_init(self, admm: ADMM):
        """Second stage initializer to be called by :meth:`.ADMM.__init__`.

        Args:
            admm: Reference to :class:`.ADMM` object to which the
               :class:`.SubproblemSolver` object is to be attached.
        """
        self.admm = admm


class GenericSubproblemSolver(SubproblemSolver):
    """Solver for generic problem without special structure.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
           attached.
        minimize_kwargs (dict): Dictionary of arguments for
           :func:`scico.solver.minimize`.
    """

    def __init__(self, minimize_kwargs: dict = {"options": {"maxiter": 100}}):
        """Initialize a :class:`GenericSubproblemSolver` object.

        Args:
            minimize_kwargs: Dictionary of arguments for
                :func:`scico.solver.minimize`.
        """
        self.minimize_kwargs = minimize_kwargs
        self.info = {}

    def solve(self, x0: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Solve the ADMM step.

        Args:
           x0: Initial value.

        Returns:
            Computed solution.
        """

        x0 = ensure_on_device(x0)

        @jax.jit
        def obj(x):
            out = 0.0
            for rhoi, Ci, zi, ui in zip(
                self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
            ):
                out = out + 0.5 * rhoi * norm(zi - ui - Ci(x)) ** 2
            if self.admm.f is not None:
                out = out + self.admm.f(x)
            return out

        res = minimize(obj, x0, **self.minimize_kwargs)
        for attrib in ("success", "status", "message", "nfev", "njev", "nhev", "nit", "maxcv"):
            self.info[attrib] = getattr(res, attrib, None)

        return res.x


class LinearSubproblemSolver(SubproblemSolver):
    r"""Solver for quadratic functionals.

    Solver for the case in which :code:`f` is a quadratic function of
    :math:`\mb{x}`. It is a specialization of :class:`.SubproblemSolver`
    for the case where :code:`f` is an :math:`\ell_2` or weighted
    :math:`\ell_2` norm, and :code:`f.A` is a linear operator, so that
    the subproblem involves solving a linear equation. This requires that
    ``f.functional`` be an instance of either :class:`.SquaredL2Loss`
    or :class:`.WeightedSquaredL2Loss` and for the forward operator
    :code:`f.A` to be an instance of :class:`.LinearOperator`.

    In the case :class:`.WeightedSquaredL2Loss`, the
    :math:`\mb{x}`-update step is

    ..  math::

        \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2}
        \norm{\mb{y} - A x}_W^2 + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` is the weighting :class:`.Diagonal` from the
    :class:`.WeightedSquaredL2Loss` instance. This update step
    reduces to the solution of the linear system

    ..  math::

        \left(A^H W A + \sum_{i=1}^N \rho_i C_i^H C_i \right)
        \mb{x}^{(k+1)} = \;
        A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
        \mb{u}^{(k)}_i) \;.

    In the case of :class:`.SquaredL2Loss`, :math:`W` is set to
    the :class:`Identity` operator.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        cg_kwargs (dict): Dictionary of arguments for CG solver.
        cg (func): CG solver function (:func:`scico.solver.cg` or
            :func:`jax.scipy.sparse.linalg.cg`) lhs (type): Function
            implementing the linear operator needed for the
            :math:`\mb{x}` update step.
    """

    def __init__(self, cg_kwargs: Optional[dict] = None, cg_function: str = "scico"):
        """Initialize a :class:`LinearSubproblemSolver` object.

        Args:
            cg_kwargs: Dictionary of arguments for CG solver. See
                documentation for :func:`scico.solver.cg` or
                :func:`jax.scipy.sparse.linalg.cg`,
                including how to specify a preconditioner.
                Default values are the same as those of
                :func:`scico.solver.cg`, except for
                ``"tol": 1e-4`` and ``"maxiter": 100``.
            cg_function: String indicating which CG implementation to
                use. One of "jax" or "scico"; default "scico". If
                "scico", uses :func:`scico.solver.cg`. If "jax", uses
                :func:`jax.scipy.sparse.linalg.cg`. The "jax" option is
                slower on small-scale problems or problems involving
                external functions, but can be differentiated through.
                The "scico" option is faster on small-scale problems, but
                slower on large-scale problems where the forward
                operator is written entirely in jax.
        """

        default_cg_kwargs = {"tol": 1e-4, "maxiter": 100}
        if cg_kwargs:
            default_cg_kwargs.update(cg_kwargs)
        self.cg_kwargs = default_cg_kwargs
        self.cg_function = cg_function
        if cg_function == "scico":
            self.cg = scico_cg
        elif cg_function == "jax":
            self.cg = jax_cg
        else:
            raise ValueError(
                f"Parameter cg_function must be one of 'jax', 'scico'; got {cg_function}"
            )
        self.info = None

    def internal_init(self, admm):
        if admm.f is not None:
            if not isinstance(admm.f.A, LinearOperator):
                raise ValueError(
                    f"LinearSubproblemSolver requires f.A to be a scico.linop.LinearOperator; "
                    f"got {type(admm.f.A)}"
                )
            if not isinstance(admm.f, WeightedSquaredL2Loss):  # SquaredL2Loss is subclass
                raise ValueError(
                    f"LinearSubproblemSolver requires f to be a scico.loss.WeightedSquaredL2Loss"
                    f"or scico.loss.SquaredL2Loss; got {type(admm.f)}"
                )

        super().internal_init(admm)

        # Set lhs_op =  \sum_i rho_i * Ci.H @ CircularConvolve
        # Use reduce as the initialization of this sum is messy otherwise
        lhs_op = reduce(
            lambda a, b: a + b, [rhoi * Ci.gram_op for rhoi, Ci in zip(admm.rho_list, admm.C_list)]
        )
        if admm.f is not None:
            # hessian = A.T @ W @ A; W may be identity
            lhs_op = lhs_op + admm.f.hessian

        lhs_op.jit()
        self.lhs_op = lhs_op

    def compute_rhs(self) -> Union[JaxArray, BlockArray]:
        r"""Compute the right hand side of the linear equation to be solved.

        Compute

        .. math::

            A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
            \mb{u}^{(k)}_i) \;.

        Returns:
            Computed solution.
        """

        C0 = self.admm.C_list[0]
        rhs = snp.zeros(C0.input_shape, C0.input_dtype)

        if self.admm.f is not None:
            ATWy = self.admm.f.A.adj(self.admm.f.W.diagonal * self.admm.f.y)
            rhs += 2.0 * self.admm.f.scale * ATWy

        for rhoi, Ci, zi, ui in zip(
            self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
        ):
            rhs = rhs + rhoi * Ci.adj(zi - ui)
        return rhs

    def solve(self, x0: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value.

        Returns:
            Computed solution.
        """
        x0 = ensure_on_device(x0)
        rhs = self.compute_rhs()
        x, self.info = self.cg(self.lhs_op, rhs, x0, **self.cg_kwargs)
        return x


class CircularConvolveSolver(LinearSubproblemSolver):
    r"""Solver for linear operators diagonalized in the DFT domain.

    Specialization of :class:`.LinearSubproblemSolver` for the case
    where :code:`f` is an instance of :class:`.SquaredL2Loss`, the
    forward operator :code:`f.A` is either an instance of
    :class:`.Identity` or :class:`.CircularConvolve`, and the
    :code:`C_i` are all instances of :class:`.Identity` or
    :class:`.CircularConvolve`. None of the instances of
    :class:`.CircularConvolve` may sum over any of their axes.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        lhs_f (array): Left hand side, in the DFT domain, of the linear
            equation to be solved.
    """

    def __init__(self):
        """Initialize a :class:`CircularConvolveSolver` object."""

    def internal_init(self, admm):
        if admm.f is not None:
            if not isinstance(admm.f, SquaredL2Loss):
                raise ValueError(
                    "CircularConvolveSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}"
                )
            if not isinstance(admm.f.A, (CircularConvolve, Identity)):
                raise ValueError(
                    "CircularConvolveSolver requires f.A to be a scico.linop.CircularConvolve "
                    f"or scico.linop.Identity; got {type(admm.f.A)}"
                )

        super().internal_init(admm)

        self.real_result = is_real_dtype(admm.C_list[0].input_dtype)

        lhs_op_list = [
            rho * CircularConvolve.from_operator(C.gram_op)
            for rho, C in zip(admm.rho_list, admm.C_list)
        ]
        A_lhs = reduce(lambda a, b: a + b, lhs_op_list)
        if self.admm.f is not None:
            A_lhs += 2.0 * admm.f.scale * CircularConvolve.from_operator(admm.f.A.gram_op)

        self.A_lhs = A_lhs

    def solve(self, x0: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value.

        Returns:
            Computed solution.
        """
        x0 = ensure_on_device(x0)
        rhs = self.compute_rhs()
        rhs_dft = snp.fft.fftn(rhs, axes=self.A_lhs.x_fft_axes)
        x_dft = rhs_dft / self.A_lhs.h_dft
        x = snp.fft.ifftn(x_dft, axes=self.A_lhs.x_fft_axes)
        if self.real_result:
            x = x.real

        return x


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
            x0: Initial value for :math:`\mb{x}`. If None, defaults to
                an array of zeros.
            maxiter: Number of ADMM outer-loop iterations. Default: 100.
            subproblem_solver: Solver for :math:`\mb{x}`-update step.
                Defaults to ``None``, which implies use of an instance of
                :class:`GenericSubproblemSolver`.
            itstat_options: A dict of named parameters to be passed to
                the :class:`.diagnostics.IterationStats` initializer. The
                dict may also include an additional key "itstat_func"
                with the corresponding value being a function with two
                parameters, an integer and an ADMM object, responsible
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
        scope = {}
        exec("def itstat_func(obj): " + itstat_return, scope)

        # determine itstat options and initialize IterationStats object
        default_itstat_options = {
            "fields": itstat_fields,
            "itstat_func": scope["itstat_func"],
            "display": False,
        }
        if itstat_options:
            default_itstat_options.update(itstat_options)
        self.itstat_insert_func = default_itstat_options.pop("itstat_func", None)
        self.itstat_object = IterationStats(**default_itstat_options)

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
            x: Point at which to evaluate objective function. If `None`,
                the objective is  evaluated at the current iterate
                :code:`self.x`.
            z_list: Point at which to evaluate objective function. If
                `None`, the objective is evaluated at the current iterate
                :code:`self.z_list`.

        Returns:
            scalar: Current value of the objective function.
        """
        if (x is None) != (z_list is None):
            raise ValueError("Both or neither of x and z_list must be supplied")
        if x is None:
            x = self.x
            z_list = self.z_list
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
                If `None`, the primal residual is evaluated at the
                current iterate :code:`self.x`.

        Returns:
            Current value of primal residual.
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
            Current value of dual residual.

        """
        out = 0.0
        for zi, ziold, Ci in zip(self.z_list, self.z_list_old, self.C_list):
            out += norm(Ci.adj(zi - ziold)) ** 2
        return snp.sqrt(out)

    def z_init(self, x0: Union[JaxArray, BlockArray]):
        r"""Initialize auxiliary variables :math:`\mb{z}_i`.

        Initialized to

        .. math::
            \mb{z}_i = C_i \mb{x}^{(0)} \;.

        :code:`z_list` and :code:`z_list_old` are initialized to the same
        value.

        Args:
            x0: Initial value of :math:`\mb{x}`.
        """
        z_list = [Ci(x0) for Ci in self.C_list]
        z_list_old = z_list.copy()
        return z_list, z_list_old

    def u_init(self, x0: Union[JaxArray, BlockArray]):
        r"""Initialize scaled Lagrange multipliers :math:`\mb{u}_i`.

        Initialized to

        .. math::
            \mb{u}_i = C_i \mb{x}^{(0)} \;.

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

        Run the ADMM algorithm for a total of ``self.maxiter`` iterations.

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
