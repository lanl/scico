# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM auxiliary classes."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import reduce
from typing import Any, Optional, Union

import jax
from jax.scipy.sparse.linalg import cg as jax_cg

import scico.numpy as snp
import scico.optimize.admm as soa
from scico.linop import CircularConvolve, Identity, LinearOperator
from scico.loss import SquaredL2Loss
from scico.numpy import BlockArray
from scico.numpy.linalg import norm
from scico.numpy.util import ensure_on_device, is_real_dtype
from scico.solver import cg as scico_cg
from scico.solver import minimize
from scico.typing import JaxArray


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
    `f.functional` be an instance of :class:`.SquaredL2Loss` and for
    the forward operator :code:`f.A` to be an instance of
    :class:`.LinearOperator`.

    The :math:`\mb{x}`-update step is

    ..  math::

        \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2}
        \norm{\mb{y} - A \mb{x}}_W^2 + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` a weighting :class:`.Diagonal` operator
    or an :class:`.Identity` operator (i.e. no weighting).
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

    def __init__(self, cg_kwargs: Optional[dict[str, Any]] = None, cg_function: str = "scico"):
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

    def internal_init(self, admm: soa.ADMM):
        if admm.f is not None:
            if not isinstance(admm.f, SquaredL2Loss):
                raise ValueError(
                    "LinearSubproblemSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}"
                )
            if not isinstance(admm.f.A, LinearOperator):
                raise ValueError(
                    f"LinearSubproblemSolver requires f.A to be a scico.linop.LinearOperator; "
                    f"got {type(admm.f.A)}"
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
            ATWy = self.admm.f.A.adj(self.admm.f.W.diagonal * self.admm.f.y)  # type: ignore
            rhs += 2.0 * self.admm.f.scale * ATWy  # type: ignore

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
        x, self.info = self.cg(self.lhs_op, rhs, x0, **self.cg_kwargs)  # type: ignore
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

    def internal_init(self, admm: soa.ADMM):
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
