# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM solver and auxiliary classes."""

from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import jax
from jax.scipy.sparse.linalg import cg as jax_cg

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.linop import CircularConvolve, Identity, LinearOperator
from scico.loss import Loss, SquaredL2Loss, WeightedSquaredL2Loss
from scico.math import is_real_dtype
from scico.numpy.linalg import norm
from scico.solver import cg as scico_cg
from scico.solver import minimize
from scico.typing import JaxArray
from scico.util import ensure_on_device

__author__ = """\n""".join(["Luke Pfister <pfister@lanl.gov>", "Brendt Wohlberg <brendt@ieee.org>"])


class SubproblemSolver:
    r"""Base class for solvers for the non-separable ADMM step.

    The ADMM solver implemented by :class:`.ADMM` addresses a general problem form for
    which one of the corresponding ADMM algorithm subproblems is separable into distinct
    subproblems for each of the :math:`g_i`, and another that is non-separable, involving
    function :math:`f` and a sum over :math:`\ell_2` norm terms involving all operators
    :math:`C_i`. This class is a base class for solvers of the latter subproblem

    .. math::

        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2


    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is attached.
    """

    def internal_init(self, admm: "ADMM"):
        """Second stage initializer to be called by :meth:`.ADMM.__init__`.

        Args:
            admm: Reference to :class:`.ADMM` object to which the :class:`.SubproblemSolver`
               object is to be attached.
        """
        self.admm = admm


class GenericSubproblemSolver(SubproblemSolver):
    """Solver for generic problem without special structure that can be exploited.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is attached.
        minimize_kwargs (dict): Dictionary of arguments for :func:`scico.solver.minimize`.
    """

    def __init__(self, minimize_kwargs: dict = {"options": {"maxiter": 100}}):
        """Initialize a :class:`GenericSubproblemSolver` object.

        Args:
            minimize_kwargs : Dictionary of arguments for :func:`scico.solver.minimize`.
        """
        self.minimize_kwargs = minimize_kwargs

    def solve(self, x0: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Solve the ADMM step.

        Args:
           x0: Starting point.

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

        return res.x


class LinearSubproblemSolver(SubproblemSolver):
    r"""Solver for the case where :code:`f` is a quadratic function of :math:`\mb{x}`.

    Specialization of :class:`.SubproblemSolver` for the case where :code:`f` is an
    :math:`\ell_2` or weighted :math:`\ell_2` norm, and :code:`f.A` is a linear
    operator, so that the subproblem involves solving a linear equation. This requires
    that ``f.functional`` be an instance of either :class:`.SquaredL2Loss` or
    :class:`.WeightedSquaredL2Loss` and for the forward operator :code:`f.A` to be an
    instance of :class:`.LinearOperator`.

    In the case :class:`.WeightedSquaredL2Loss`, the :math:`\mb{x}`-update step is

    .. math::
         \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2} \norm{\mb{y} - A x}_W^2 +
         \sum_i \frac{\rho_i}{2} \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` is the weighting :class:`.LinearOperator` from the :class:`.WeightedSquaredL2Loss`
    instance. This update step reduces to the solution of the linear system

    .. math::
        \left(A^* W A + \sum_{i=1}^N \rho_i C_i^* C_i \right) \mb{x}^{(k+1)} = \;
        A^* W \mb{y} + \sum_{i=1}^N \rho_i C_i^* ( \mb{z}^{(k)}_i - \mb{u}^{(k)}_i) \;.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is attached.
        cg_kwargs (dict): Dictionary of arguments for CG solver.
        cg (func): CG solver function (:func:`scico.solver.cg` or
           :func:`jax.scipy.sparse.linalg.cg`)
        lhs (type): Function implementing the linear operator needed for the :math:`\mb{x}`
           update step
    """

    def __init__(self, cg_kwargs: dict = {"maxiter": 100}, cg_function: str = "scico"):
        """Initialize a :class:`LinearSubproblemSolver` object.

        Args:
            cg_kwargs : Dictionary of arguments for CG solver. See :func:`scico.solver.cg` or
                :func:`jax.scipy.sparse.linalg.cg`, documentation, including how to specify
                a preconditioner.
            cg_function: String indicating which CG implementation to use. One of "jax" or
                "scico"; default "scico".  If "scico", uses :func:`scico.solver.cg`.  If
                "jax", uses :func:`jax.scipy.sparse.linalg.cg`.  The "jax" option is slower
                on small-scale problems or problems involving external functions, but
                can be differentiated through.  The "scico" option is faster on small-scale
                problems, but slower on large-scale problems where the forward operator is
                written entirely in jax.
        """
        self.cg_kwargs = cg_kwargs
        if cg_function == "scico":
            self.cg = scico_cg
        elif cg_function == "jax":
            self.cg = jax_cg
        else:
            raise ValueError(
                f"Parameter cg_function must be one of 'jax', 'scico'; got {cg_function}"
            )

    def internal_init(self, admm):
        if admm.f is not None:
            if not isinstance(admm.f.A, LinearOperator):
                raise ValueError(
                    "LinearSubproblemSolver requires f.A to be a scico.linop.LinearOperator; "
                    f"got {type(admm.f.A)}"
                )
            if not admm.f.is_quadratic:
                raise ValueError("LinearSubproblemSolver requires f.is_quadratic == True")

        super().internal_init(admm)

        # set lhs_op =  \sum_i rho_i * Ci.H @ CircularConvolve
        # use reduce as the initialization of this sum is messy otherwise
        lhs_op = reduce(
            lambda a, b: a + b, [rhoi * Ci.gram_op for rhoi, Ci in zip(admm.rho_list, admm.C_list)]
        )
        if admm.f is not None:
            # hessian = A.T @ W @ A; W may be identity
            lhs_op = lhs_op + 2.0 * admm.f.scale * admm.f.hessian

        lhs_op.jit()
        self.lhs_op = lhs_op

    def compute_rhs(self) -> Union[JaxArray, BlockArray]:
        r"""Compute the right hand side of the linear equation to be solved.

        Compute

        .. math::

            A^* W \mb{y} + \sum_{i=1}^N \rho_i C_i^* ( \mb{z}^{(k)}_i - \mb{u}^{(k)}_i) \;.

        Returns:
            Computed solution.
        """

        C0 = self.admm.C_list[0]
        rhs = snp.zeros(C0.input_shape, C0.input_dtype)

        if self.admm.f is not None:
            if isinstance(self.admm.f, WeightedSquaredL2Loss):
                ATWy = self.admm.f.A.adj(self.admm.f.weight_op @ self.admm.f.y)
                rhs += 2.0 * self.admm.f.scale * ATWy
            else:
                ATy = self.admm.f.A.adj(self.admm.f.y)
                rhs += 2.0 * self.admm.f.scale * ATy

        for rhoi, Ci, zi, ui in zip(
            self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
        ):
            rhs = rhs + rhoi * Ci.adj(zi - ui)
        return rhs

    def solve(self, x0: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Starting point.

        Returns:
            Computed solution.
        """
        x0 = ensure_on_device(x0)
        rhs = self.compute_rhs()
        x, self.info = self.cg(self.lhs_op, rhs, x0=x0, **self.cg_kwargs)
        return x


class CircularConvolveSolver(LinearSubproblemSolver):
    r"""Solver for linear operators diagonalized in the DFT domain.

    Specialization of :class:`.LinearSubproblemSolver` for the case where :code:`f` is an
    instance of :class:`.SquaredL2Loss`, the forward operator :code:`f.A` is either
    an instance of :class:`.Identity` or :class:`.CircularConvolve`, and the :code:`C_i` are
    all instances of :class:`.Identity` or :class:`.CircularConvolve`. None of the instances of
    :class:`.CircularConvolve` may sum over any of their axes.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is attached.
        lhs_f (array): Left hand side, in the DFT domain, of the linear equation to be solved.
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
            if not (isinstance(admm.f.A, CircularConvolve) or isinstance(admm.f.A, Identity)):
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
            x0: Starting point.

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
    r"""Basic Alternating Direction Method of Multipliers (ADMM)
    algorithm :cite:`boyd-2010-distributed`.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}),

    where :math:`f` is an instance of :class:`.Loss`, the :math:`g_i` are :class:`.Functional`,
    and the :math:`C_i` are :class:`.LinearOperator`.

    The optimization problem is solved by introducing the splitting :math:`\mb{z}_i = C_i \mb{x}`
    and solving

    .. math::
        \argmin_{\mb{x}, \mb{z}_i} \; f(\mb{x}) + \sum_{i=1}^N g_i(\mb{z}_i) \;
       \text{such that}\; C_i \mb{x} = \mb{z}_i \;,

    via an ADMM algorithm consisting of the iterations

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i \frac{\rho_i}{2}
       \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \\
        \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) + \frac{\rho_i}{2}
        \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i \mb{x}^{(k+1)}}_2^2  \\
       \mb{u}_i^{(k+1)} &=  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} - \mb{z}^{(k+1)}_i  \; .
       \end{aligned}

    For documentation on minimization with respect to :math:`\mb{x}`, see :meth:`x_step`.

    For documentation on minimization with respect to :math:`\mb{z}_i` and
    :math:`\mb{u}_i`, see :meth:`z_and_u_step`.


    Attributes:
        f (:class:`.Loss`): Loss function
        g_list (list of :class:`.Functional`): List of :math:`g_i`
            functionals. Must be same length as :code:`C_list` and :code:`rho_list`
        C_list (list of :class:`.LinearOperator`): List of :math:`C_i` operators
        maxiter (int): Number of ADMM outer-loop iterations.
        rho_list (list of scalars): List of :math:`\rho_i` penalty parameters.
            Must be same length as :code:`C_list` and :code:`g_list`
        u_list (list of array-like): List of scaled Lagrange multipliers
            :math:`\mb{u}_i` at current iteration.
        x (array-like): Solution
        subproblem_solver (:class:`.SubproblemSolver`): Solver for :math:`\mb{x}`-update step.
        z_list (list of array-like): List of auxiliary variables :math:`\mb{z}_i`
            at current iteration.
        z_list_old (list of array-like): List of auxiliary variables :math:`\mb{z}_i`
            at previous iteration.
    """

    def __init__(
        self,
        f: Loss,
        g_list: List[Functional],
        C_list: List[LinearOperator],
        rho_list: List[float],
        x0: Optional[Union[JaxArray, BlockArray]] = None,
        maxiter: int = 100,
        subproblem_solver: Optional[SubproblemSolver] = None,
        verbose: bool = False,
        itstat: Optional[Tuple[dict, Callable]] = None,
    ):
        r"""Initialize an :class:`ADMM` object.

        Args:
            f : Loss function
            g_list : List of :math:`g_i`
                functionals. Must be same length as :code:`C_list` and :code:`rho_list`
            C_list : List of :math:`C_i` operators
            rho_list : List of :math:`\rho_i` penalty parameters.
                Must be same length as :code:`C_list` and :code:`g_list`
            x0 : Starting point for :math:`\mb{x}`.  If None, defaults to an array of zeros.
            maxiter : Number of ADMM outer-loop iterations. Default: 100.
            subproblem_solver : Solver for :math:`\mb{x}`-update step. Defaults to ``None``, which
                implies use of an instance of :class:`GenericSubproblemSolver`.
            verbose: Flag indicating whether iteration statistics should be displayed.
            itstat: A tuple (`fieldspec`, `insertfunc`), where `fieldspec` is a dict suitable
                for passing to the `fields` argument of the :class:`.diagnostics.IterationStats`
                initializer, and `insertfunc` is a function with two parameters, an integer
                and an ADMM object, responsible for constructing a tuple ready for insertion into
                the :class:`.diagnostics.IterationStats` object. If None, default values are
                used for the tuple components.
        """
        N = len(g_list)
        if len(C_list) != N:
            raise Exception(f"len(C_list)={len(C_list)} not equal to len(g_list)={N}")
        if len(rho_list) != N:
            raise Exception(f"len(rho_list)={len(rho_list)} not equal to len(g_list)={N}")

        self.f: Loss = f
        self.g_list: List[Functional] = g_list
        self.C_list: List[LinearOperator] = C_list
        self.rho_list: List[float] = rho_list
        self.maxiter: int = maxiter
        # ToDo: a None value should imply automatic selection of the solver
        if subproblem_solver is None:
            subproblem_solver = GenericSubproblemSolver()
        self.subproblem_solver: SubproblemSolver = subproblem_solver
        self.subproblem_solver.internal_init(self)

        self.verbose: bool = verbose

        if itstat:
            itstat_dict = itstat[0]
            itstat_func = itstat[1]
        elif itstat is None:
            if all([_.has_eval for _ in self.g_list]):
                itstat_dict = {
                    "Iter": "%d",
                    "Objective": "%8.3e",
                    "Primal Rsdl": "%8.3e",
                    "Dual Rsdl": "%8.3e",
                }

                def itstat_func(i, obj):
                    return (
                        i,
                        obj.objective(),
                        obj.norm_primal_residual(),
                        obj.norm_dual_residual(),
                    )

            else:
                # at least one 'g' can't be evaluated, so drop objective from the default itstat
                itstat_dict = {"Iter": "%d", "Primal Rsdl": "%8.3e", "Dual Rsdl": "%8.3e"}

                def itstat_func(i, admm):
                    return (
                        i,
                        admm.norm_primal_residual(),
                        admm.norm_dual_residual(),
                    )

        self.itstat_object = IterationStats(itstat_dict, display=verbose)
        self.itstat_insert_func = itstat_func

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

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(\mb{z}_i)


        Args:
            x: Point at which to evaluate objective function. If `None`, the objective is
                evaluated at the current iterate :code:`self.x`
            z_list: Point at which to evaluate objective function. If `None`, the objective is
                evaluated at the current iterate :code:`self.z_list`


        Returns:
            scalar: Current value of the objective function
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

        .. math::
            \left(\sum_{i=1}^N \norm{C_i \mb{x} - \mb{z}_i}_2^2\right)^{1/2}

        Args:
            x: Point at which to evaluate primal residual.
               If `None`, the primal residual is evaluated at the current iterate :code:`self.x`

        Returns:
            Current value of primal residual
        """
        if x is None:
            x = self.x

        out = 0.0
        for Ci, zi in zip(self.C_list, self.z_list):
            out += norm(Ci(self.x) - zi) ** 2
        return snp.sqrt(out)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        .. math::
            \left(\sum_{i=1}^N \norm{\mb{z}^{(k)} - \mb{z}^{(k-1)}}_2^2\right)^{1/2}

        Returns:
            Current value of dual residual

        """
        out = 0.0
        for zi, ziold, Ci in zip(self.z_list, self.z_list_old, self.C_list):
            out += norm(Ci.adj(zi - ziold)) ** 2
        return snp.sqrt(out)

    def z_init(self, x0: Union[JaxArray, BlockArray]):
        r"""Initialize auxiliary variables :math:`\mb{z}`.

        Initializes to

        .. math::
            \mb{z}_i = C_i \mb{x}_0

        :code:`z_list` and :code:`z_list_old` are initialized to the same value.

        Args:
            x0: Starting point for :math:`\mb{x}`
        """
        z_list = [Ci(x0) for Ci in self.C_list]
        z_list_old = z_list.copy()
        return z_list, z_list_old

    def u_init(self, x0: Union[JaxArray, BlockArray]):
        r"""Initialize scaled Lagrange multipliers :math:`\mb{u}`.

        Initializes to

        .. math::
            \mb{u}_i = C_i \mb{x}_0


        Args:
            x0: Starting point for :math:`\mb{x}`
        """
        u_list = [snp.zeros(Ci.output_shape, dtype=Ci.output_dtype) for Ci in self.C_list]
        return u_list

    def x_step(self, x):
        r"""Update :math:`\mb{x}` by solving the optimization problem.

        .. math::
            \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i \frac{\rho_i}{2}
            \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2

        """
        return self.subproblem_solver.solve(x)

    def z_and_u_step(self, u_list, z_list):
        r""" Update the auxiliary variables :math:`\mb{z}` and scaled Lagrange multipliers
        :math:`\mb{u}`.

        The auxiliary variables are updated according to

        .. math::
            \begin{aligned}
            \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) + \frac{\rho_i}{2}
            \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i \mb{x}^{(k+1)}}_2^2  \\
            &= \mathrm{prox}_{g_i}(C_i \mb{x} + \mb{u}_i, 1 / \rho_i)
            \end{aligned}

        while the scaled Lagrange multipliers are updated according to

        .. math::
            \mb{u}_i^{(k+1)} =  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} - \mb{z}^{(k+1)}_i

        """
        z_list_old = z_list.copy()

        # unpack the arrays that will be changing to prevent side-effects
        z_list = self.z_list
        u_list = self.u_list

        for i, (rhoi, fi, Ci, zi, ui) in enumerate(
            zip(self.rho_list, self.g_list, self.C_list, z_list, u_list)
        ):
            Cix = Ci(self.x)
            zi = fi.prox(Cix + ui, 1 / rhoi)
            ui = ui + Cix - zi
            z_list[i] = zi
            u_list[i] = ui
        return u_list, z_list, z_list_old

    def step(self):
        """Perform a single ADMM iteration.

        Equivalent to calling :meth:`.x_step` followed by :meth:`.z_and_u_step`.

        """
        self.x = self.x_step(self.x)
        self.u_list, self.z_list, self.z_list_old = self.z_and_u_step(self.u_list, self.z_list)

    def solve(self) -> Union[JaxArray, BlockArray]:
        r"""Initialize and run the ADMM algorithm for a total of ``self.maxiter`` iterations.

        Returns:
            Computed solution.
        """
        for itnum in range(self.maxiter):
            self.x = self.x_step(self.x)
            self.u_list, self.z_list, self.z_list_old = self.z_and_u_step(self.u_list, self.z_list)
            self.itstat_object.insert(self.itstat_insert_func(itnum, self))
        return self.x
