# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM auxiliary classes (base class, generic, linear, and matrix
subproblem solvers)."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import reduce
from typing import Any, Optional, Union

import jax
from jax import Device
from jax.scipy.sparse.linalg import cg as jax_cg
from jax.sharding import Sharding

import scico.numpy as snp
import scico.optimize.admm as soa
from scico.linop import (
    Diagonal,
    LinearOperator,
    MatrixOperator,
)
from scico.loss import SquaredL2Loss
from scico.numpy import Array, BlockArray
from scico.solver import MatrixATADSolver
from scico.solver import cg as scico_cg
from scico.solver import minimize


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

    def internal_init(self, admm: soa.ADMM):
        """Second stage initializer to be called by :meth:`.ADMM.__init__`.

        Args:
            admm: Reference to :class:`.ADMM` object to which the
               :class:`.SubproblemSolver` object is to be attached.
        """
        self.admm = admm


class GenericSubproblemSolver(SubproblemSolver):
    """Solver for generic problem without special structure.

    Note that this solver is only suitable for small-scale problems.

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
        self.info: dict = {}

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
           x0: Initial value.

        Returns:
            Computed solution.
        """

        @jax.jit
        def obj(x):
            out = 0.0
            for rhoi, Ci, zi, ui in zip(
                self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
            ):
                out += 0.5 * rhoi * snp.sum(snp.abs(zi - ui - Ci(x)) ** 2)
            if self.admm.f is not None:
                out += self.admm.f(x)
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
    :code:`f.functional` be an instance of :class:`.SquaredL2Loss` and
    for the forward operator :code:`f.A` to be an instance of
    :class:`.LinearOperator`.

    The :math:`\mb{x}`-update step is

    ..  math::

        \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2}
        \norm{\mb{y} - A \mb{x}}_W^2 + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` a weighting :class:`.Diagonal` operator
    or an :class:`.Identity` operator (i.e., no weighting).
    This update step reduces to the solution of the linear system

    ..  math::

        \left(A^H W A + \sum_{i=1}^N \rho_i C_i^H C_i \right)
        \mb{x}^{(k+1)} = \;
        A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
        \mb{u}^{(k)}_i) \;.


    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        cg_kwargs (dict): Dictionary of arguments for CG solver.
        cg (func): CG solver function (:func:`scico.solver.cg` or
            :func:`jax.scipy.sparse.linalg.cg`) lhs (type): Function
            implementing the linear operator needed for the
            :math:`\mb{x}` update step.
    """

    def __init__(
        self,
        cg_kwargs: Optional[dict[str, Any]] = None,
        cg_function: str = "scico",
        device: Optional[Union[Device, Sharding]] = None,
    ):
        """Initialize a :class:`LinearSubproblemSolver` object.

        Args:
            cg_kwargs: Dictionary of arguments for CG solver. See
                documentation for :func:`scico.solver.cg` or
                :func:`jax.scipy.sparse.linalg.cg`,
                including how to specify a preconditioner.
                Default values are the same as those of
                :func:`scico.solver.cg`, except for
                `"tol": 1e-4` and `"maxiter": 100`.
            cg_function: String indicating which CG implementation to
                use. One of "jax" or "scico"; default "scico". If
                "scico", uses :func:`scico.solver.cg`. If "jax", uses
                :func:`jax.scipy.sparse.linalg.cg`. The "jax" option is
                slower on small-scale problems or problems involving
                external functions, but can be differentiated through.
                The "scico" option is faster on small-scale problems, but
                slower on large-scale problems where the forward
                operator is written entirely in jax.
           device: Device or sharding for array constructed in
                :meth:`compute_rhs`.
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
                f"Argument 'cg_function' must be one of 'jax', 'scico'; got {cg_function}."
            )
        self.info = None
        self.device = device

    def internal_init(self, admm: soa.ADMM):
        if admm.f is not None:
            if not isinstance(admm.f, SquaredL2Loss):
                raise TypeError(
                    "LinearSubproblemSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}."
                )
            if not isinstance(admm.f.A, LinearOperator):
                raise TypeError(
                    "LinearSubproblemSolver requires f.A to be a scico.linop.LinearOperator; "
                    f"got {type(admm.f.A)}."
                )

        super().internal_init(admm)  # call method of SubproblemSolver via GenericSubproblemSolver

        # Set lhs_op =  \sum_i rho_i * Ci.H @ Ci
        # Use reduce as the initialization of this sum is messy otherwise
        lhs_op = reduce(
            lambda a, b: a + b, [rhoi * Ci.gram_op for rhoi, Ci in zip(admm.rho_list, admm.C_list)]
        )
        if admm.f is not None:
            # hessian = A.T @ W @ A; W may be identity
            lhs_op += admm.f.hessian

        self.lhs_op = lhs_op

    def compute_rhs(self) -> Union[Array, BlockArray]:
        r"""Compute the right hand side of the linear equation to be solved.

        Compute

        .. math::

            A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
            \mb{u}^{(k)}_i) \;.

        Returns:
            Computed solution.
        """

        C0 = self.admm.C_list[0]
        rhs = snp.zeros(C0.input_shape, C0.input_dtype, device=self.device)

        if self.admm.f is not None:
            ATWy = self.admm.f.A.adj(self.admm.f.W.diagonal * self.admm.f.y)  # type: ignore
            rhs += 2.0 * self.admm.f.scale * ATWy  # type: ignore

        for rhoi, Ci, zi, ui in zip(
            self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
        ):
            rhs += rhoi * Ci.adj(zi - ui)
        return rhs

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value.

        Returns:
            Computed solution.
        """
        rhs = self.compute_rhs()
        x, self.info = self.cg(self.lhs_op, rhs, x0, **self.cg_kwargs)  # type: ignore
        return x


class MatrixSubproblemSolver(LinearSubproblemSolver):
    r"""Solver for quadratic functionals involving matrix operators.

    Solver for the case in which :math:`f` is a quadratic function of
    :math:`\mb{x}`, and :math:`A` and all the :math:`C_i` are diagonal
    or matrix operators. It is a specialization of
    :class:`.LinearSubproblemSolver`.

    As for :class:`.LinearSubproblemSolver`, the :math:`\mb{x}`-update
    step is

    ..  math::

        \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2}
        \norm{\mb{y} - A \mb{x}}_W^2 + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` is a weighting :class:`.Diagonal` operator
    or an :class:`.Identity` operator (i.e., no weighting).
    This update step reduces to the solution of the linear system

    ..  math::

        \left(A^H W A + \sum_{i=1}^N \rho_i C_i^H C_i \right)
        \mb{x}^{(k+1)} = \;
        A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
        \mb{u}^{(k)}_i) \;,

    which is solved by factorization of the left hand side of the
    equation, using :class:`.MatrixATADSolver`.


    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        solve_kwargs (dict): Dictionary of arguments for solver
            :class:`.MatrixATADSolver` initialization.
    """

    def __init__(
        self,
        check_solve: bool = False,
        solve_kwargs: Optional[dict[str, Any]] = None,
        device: Optional[Union[Device, Sharding]] = None,
    ):
        """Initialize a :class:`MatrixSubproblemSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
            solve_kwargs: Dictionary of arguments for solver
                :class:`.MatrixATADSolver` initialization.
            device: Device or sharding for arrays constructed in
                :meth:`internal_init` and :meth:`compute_rhs`.
        """
        self.check_solve = check_solve
        default_solve_kwargs = {"cho_factor": False}
        if solve_kwargs:
            default_solve_kwargs.update(solve_kwargs)
        self.solve_kwargs = default_solve_kwargs
        self.device = device

    def internal_init(self, admm: soa.ADMM):
        if admm.f is not None:
            if not isinstance(admm.f, SquaredL2Loss):
                raise TypeError(
                    "MatrixSubproblemSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}."
                )
            if not isinstance(admm.f.A, (Diagonal, MatrixOperator)):
                raise TypeError(
                    "MatrixSubproblemSolver requires f.A to be a Diagonal or MatrixOperator; "
                    f"got {type(admm.f.A)}."
                )
        for i, Ci in enumerate(admm.C_list):
            if not isinstance(Ci, (Diagonal, MatrixOperator)):
                raise TypeError(
                    "MatrixSubproblemSolver requires C[{i}] to be a Diagonal or MatrixOperator; "
                    f"got {type(Ci)}."
                )

        SubproblemSolver.internal_init(self, admm)

        if admm.f is None:
            C0 = self.admm.C_list[0]
            A = snp.zeros(C0.input_shape[0], dtype=C0.input_dtype, device=self.device)
            W = None
        else:
            A = admm.f.A
            W = 2.0 * self.admm.f.scale * admm.f.W  # type: ignore

        Csum = reduce(
            lambda a, b: a + b, [rhoi * Ci.gram_op for rhoi, Ci in zip(admm.rho_list, admm.C_list)]
        )
        self.solver = MatrixATADSolver(A, Csum, W, **self.solve_kwargs)

    def solve(self, x0: Array) -> Array:
        """Solve the ADMM step.

        Args:
            x0: Initial value (ignored).

        Returns:
            Computed solution.
        """
        rhs = self.compute_rhs()
        x = self.solver.solve(rhs)
        if self.check_solve:
            self.accuracy = self.solver.accuracy(x, rhs)

        return x
