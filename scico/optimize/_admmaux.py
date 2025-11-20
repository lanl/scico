# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
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
from scico.functional import ZeroFunctional
from scico.linop import (
    CircularConvolve,
    ComposedLinearOperator,
    Diagonal,
    Identity,
    LinearOperator,
    MatrixOperator,
)
from scico.loss import SquaredL2Loss
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_real_dtype
from scico.solver import ConvATADSolver, MatrixATADSolver
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
                f"Argument 'cg_function' must be one of 'jax', 'scico'; got {cg_function}."
            )
        self.info = None

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

        super().internal_init(admm)

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
        rhs = snp.zeros(C0.input_shape, C0.input_dtype)

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

    def __init__(self, check_solve: bool = False, solve_kwargs: Optional[dict[str, Any]] = None):
        """Initialize a :class:`MatrixSubproblemSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
            solve_kwargs: Dictionary of arguments for solver
                :class:`.MatrixATADSolver` initialization.
        """
        self.check_solve = check_solve
        default_solve_kwargs = {"cho_factor": False}
        if solve_kwargs:
            default_solve_kwargs.update(solve_kwargs)
        self.solve_kwargs = default_solve_kwargs

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

        super().internal_init(admm)

        if admm.f is None:
            A = snp.zeros(admm.C_list[0].input_shape[0], dtype=admm.C_list[0].input_dtype)
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


class CircularConvolveSolver(LinearSubproblemSolver):
    r"""Solver for linear operators diagonalized in the DFT domain.

    Specialization of :class:`.LinearSubproblemSolver` for the case
    where :code:`f` is ``None``, or an instance of
    :class:`.SquaredL2Loss` with a forward operator :code:`f.A` that is
    either an instance of :class:`.Identity` or
    :class:`.CircularConvolve`, and the :code:`C_i` are all shift
    invariant linear operators, examples of which include instances of
    :class:`.Identity` as well as some instances (depending on
    initializer parameters) of :class:`.CircularConvolve` and
    :class:`.FiniteDifference`. None of the instances of
    :class:`.CircularConvolve` may sum over any of their axes.

    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        lhs_f (array): Left hand side, in the DFT domain, of the linear
            equation to be solved.
    """

    def __init__(self, ndims: Optional[int] = None):
        """Initialize a :class:`CircularConvolveSolver` object.

        Args:
            ndims: Number of trailing dimensions of the input and kernel
                involved in the :class:`.CircularConvolve` convolutions.
                In most cases this value is automatically determined from
                the optimization problem specification, but this is not
                possible when :code:`f` is ``None`` and none of the
                :code:`C_i` are of type :class:`.CircularConvolve`. When
                not ``None``, this parameter overrides the automatic
                mechanism.
        """
        self.ndims = ndims

    def internal_init(self, admm: soa.ADMM):
        if admm.f is None:
            is_cc = [isinstance(C, CircularConvolve) for C in admm.C_list]
            if any(is_cc):
                auto_ndims = admm.C_list[is_cc.index(True)].ndims
            else:
                auto_ndims = None
        else:
            if not isinstance(admm.f, SquaredL2Loss):
                raise TypeError(
                    "CircularConvolveSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}."
                )
            if not isinstance(admm.f.A, (CircularConvolve, Identity)):
                raise TypeError(
                    "CircularConvolveSolver requires f.A to be a scico.linop.CircularConvolve "
                    f"or scico.linop.Identity; got {type(admm.f.A)}."
                )
            auto_ndims = admm.f.A.ndims if isinstance(admm.f.A, CircularConvolve) else None

        if self.ndims is None:
            self.ndims = auto_ndims
        super().internal_init(admm)

        self.real_result = is_real_dtype(admm.C_list[0].input_dtype)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        lhs_op_list = [
            rho * CircularConvolve.from_operator(C.gram_op, ndims=self.ndims)
            for rho, C in zip(admm.rho_list, admm.C_list)
        ]
        A_lhs = reduce(lambda a, b: a + b, lhs_op_list)
        if self.admm.f is not None:
            A_lhs += (
                2.0
                * admm.f.scale
                * CircularConvolve.from_operator(admm.f.A.gram_op, ndims=self.ndims)
            )

        self.A_lhs = A_lhs

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value (unused, has no effect).

        Returns:
            Computed solution.
        """
        rhs = self.compute_rhs()
        rhs_dft = snp.fft.fftn(rhs, axes=self.A_lhs.x_fft_axes)
        x_dft = rhs_dft / self.A_lhs.h_dft
        x = snp.fft.ifftn(x_dft, axes=self.A_lhs.x_fft_axes)
        if self.real_result:
            x = x.real

        return x


class FBlockCircularConvolveSolver(LinearSubproblemSolver):
    r"""Solver for linear operators block-diagonalized in the DFT domain.

    Specialization of :class:`.LinearSubproblemSolver` for the case where
    :code:`f` is an instance of :class:`.SquaredL2Loss`, the forward
    operator :code:`f.A` is a composition of a :class:`.Sum` operator and
    a :class:`.CircularConvolve` operator. The former must sum over the
    first axis of its input, and the latter must be initialized so that
    it convolves a set of filters, indexed by the first axis, with an
    input array that has the same number of axes as the filter array, and
    has an initial axis of the same length as that of the filter array.
    The :math:`C_i` must all be shift invariant linear operators,
    examples of which include instances of :class:`.Identity` as well as
    some instances (depending on initializer parameters) of
    :class:`.CircularConvolve` and :class:`.FiniteDifference`. None of
    the instances of :class:`.CircularConvolve` may be summed over any of
    their axes.

    The solver is based on the frequency-domain approach proposed in
    :cite:`wohlberg-2014-efficient`. We have :math:`f = \omega
    \norm{A \mb{x} - \mb{y}}_2^2`, where typically :math:`\omega = 1/2`,
    and :math:`A` is a block-row operator with circulant blocks, i.e. it
    can be written as

    .. math::

       A = \left( \begin{array}{cccc} A_1 & A_2 & \ldots & A_{K}
           \end{array} \right) \;,

    where all of the :math:`A_k` are circular convolution operators. The
    complete functional to be minimized is

    .. math::

       \omega \norm{A \mb{x} - \mb{y}}_2^2 + \sum_{i=1}^N g_i(C_i \mb{x})
       \;,

    where the :math:`C_i` are either identity or circular convolutions,
    and the ADMM x-step is

    .. math::

       \mb{x}^{(j+1)} = \argmin_{\mb{x}} \; \omega \norm{A \mb{x}
       - \mb{y}}_2^2 + \sum_i \frac{\rho_i}{2} \norm{C_i \mb{x} -
       (\mb{z}^{(j)}_i - \mb{u}^{(j)}_i)}_2^2 \;.

    This subproblem is most easily solved in the DFT transform domain,
    where the circular convolutions become diagonal operators. Denoting
    the frequency-domain versions of variables with a circumflex (e.g.
    :math:`\hat{\mb{x}}` is the frequency-domain version of
    :math:`\mb{x}`), the solution of the subproblem can be written as

    .. math::

       \left( \hat{A}^H \hat{A} + \frac{1}{2 \omega} \sum_i \rho_i
       \hat{C}_i^H \hat{C}_i \right) \hat{\mathbf{x}} = \hat{A}^H
       \hat{\mb{y}} + \frac{1}{2 \omega} \sum_i \rho_i \hat{C}_i^H
       (\hat{\mb{z}}_i - \hat{\mb{u}}_i) \;.

    This linear equation is computational expensive to solve because
    the left hand side includes the term :math:`\hat{A}^H \hat{A}`,
    which corresponds to the outer product of :math:`\hat{A}^H`
    and :math:`\hat{A}`. A computationally efficient solution is possible,
    however, by exploiting the Woodbury matrix identity

    .. math::

       (D + U G V)^{-1} = D^{-1} - D^{-1} U (G^{-1} + V D^{-1} U)^{-1}
       V D^{-1} \;.

    Setting

    .. math::

       D &= \frac{1}{2 \omega} \sum_i \rho_i \hat{C}_i^H \hat{C}_i \\
       U &= \hat{A}^H \\
       G &= I \\
       V &= \hat{A}

    we have

    .. math::

       (D + \hat{A}^H \hat{A})^{-1} = D^{-1} - D^{-1} \hat{A}^H
       (I + \hat{A} D^{-1} \hat{A}^H)^{-1} \hat{A} D^{-1}

    which can be simplified to

    .. math::

       (D + \hat{A}^H \hat{A})^{-1} = D^{-1} (I - \hat{A}^H E^{-1}
       \hat{A} D^{-1})

    by defining :math:`E = I + \hat{A} D^{-1} \hat{A}^H`. The right
    hand side is much cheaper to compute because the only matrix
    inversions involve :math:`D`, which is diagonal, and :math:`E`,
    which is a weighted inner product of :math:`\hat{A}^H` and
    :math:`\hat{A}`.
    """

    def __init__(self, ndims: Optional[int] = None, check_solve: bool = False):
        """Initialize a :class:`FBlockCircularConvolveSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
        """
        self.ndims = ndims
        self.check_solve = check_solve
        self.accuracy: Optional[float] = None

    def internal_init(self, admm: soa.ADMM):
        if admm.f is None:
            raise ValueError("FBlockCircularConvolveSolver does not allow f to be None.")
        else:
            if not isinstance(admm.f, SquaredL2Loss):
                raise TypeError(
                    "FBlockCircularConvolveSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}."
                )
            if not isinstance(admm.f.A, ComposedLinearOperator):
                raise TypeError(
                    "FBlockCircularConvolveSolver requires f.A to be a composition of Sum "
                    f"and CircularConvolve linear operators; got {type(admm.f.A)}."
                )
        super().internal_init(admm)

        assert isinstance(self.admm.f, SquaredL2Loss)
        assert isinstance(self.admm.f.A, ComposedLinearOperator)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        c_gram_list = [
            rho * CircularConvolve.from_operator(C.gram_op, ndims=self.ndims)
            for rho, C in zip(admm.rho_list, admm.C_list)
        ]
        D = reduce(lambda a, b: a + b, c_gram_list) / (2.0 * self.admm.f.scale)
        self.solver = ConvATADSolver(self.admm.f.A, D)

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value (unused, has no effect).

        Returns:
            Computed solution.
        """
        assert isinstance(self.admm.f, SquaredL2Loss)

        rhs = self.compute_rhs() / (2.0 * self.admm.f.scale)
        x = self.solver.solve(rhs)
        if self.check_solve:
            self.accuracy = self.solver.accuracy(x, rhs)

        return x


class G0BlockCircularConvolveSolver(SubproblemSolver):
    r"""Solver for linear operators block-diagonalized in the DFT
    domain.

    Specialization of :class:`.LinearSubproblemSolver` for the case
    where :math:`f = 0` (i.e, :code:`f` is a :class:`.ZeroFunctional`),
    :math:`g_1` is an instance of :class:`.SquaredL2Loss`, :math:`C_1`
    is a composition of a :class:`.Sum` operator an a
    :class:`.CircularConvolve` operator.  The former must sum over the
    first axis of its input, and the latter must be initialized so
    that it convolves a set of filters, indexed by the first axis,
    with an input array that has the same number of axes as the filter
    array, and has an initial axis of the same length as that of the
    filter array.  The other :math:`C_i` must all be shift invariant
    linear operators, examples of which include instances of
    :class:`.Identity` as well as some instances (depending on
    initializer parameters) of :class:`.CircularConvolve` and
    :class:`.FiniteDifference`.  None of these instances of
    :class:`.CircularConvolve` may be summed over any of their axes.

    The solver is based on the frequency-domain approach proposed in
    :cite:`wohlberg-2014-efficient`.  We have :math:`g_1 = \omega
    \norm{B A \mb{x} - \mb{y}}_2^2`, where typically :math:`\omega =
    1/2`, :math:`B` is the identity or a diagonal operator, and
    :math:`A` is a block-row operator with circulant blocks, i.e. it
    can be written as

    .. math::

       A = \left( \begin{array}{cccc} A_1 & A_2 & \ldots & A_{K}
           \end{array} \right) \;,

    where all of the :math:`A_k` are circular convolution operators. The
    complete functional to be minimized is

    .. math::

       \sum_{i=1}^N g_i(C_i \mb{x}) \;,

    where

    .. math::

       g_1(\mb{z}) &= \omega \norm{B \mb{z} - \mb{y}}_2^2\\
       C_1 &= A \;,

    and the other :math:`C_i` are either identity or circular
    convolutions. The ADMM x-step is

    .. math::

       \mb{x}^{(j+1)} = \argmin_{\mb{x}} \; \rho_1 \omega \norm{
       A \mb{x} - (\mb{z}^{(j)}_1 - \mb{u}^{(j)}_1)}_2^2 + \sum_{i=2}^N
       \frac{\rho_i}{2} \norm{C_i \mb{x} - (\mb{z}^{(j)}_i -
       \mb{u}^{(j)}_i)}_2^2 \;.

    This subproblem is most easily solved in the DFT transform domain,
    where the circular convolutions become diagonal operators. Denoting
    the frequency-domain versions of variables with a circumflex (e.g.
    :math:`\hat{\mb{x}}` is the frequency-domain version of
    :math:`\mb{x}`), the solution of the subproblem can be written as

    .. math::

       \left( \hat{A}^H \hat{A} + \frac{1}{2 \omega \rho_1} \sum_{i=2}^N
       \rho_i \hat{C}_i^H \hat{C}_i \right) \hat{\mathbf{x}} =
       \hat{A}^H (\hat{\mb{z}}_1 - \hat{\mb{u}}_1) +
       \frac{1}{2 \omega \rho_1} \sum_{i=2}^N \rho_i
       \hat{C}_i^H (\hat{\mb{z}}_i - \hat{\mb{u}}_i) \;.

    This linear equation is computational expensive to solve because
    the left hand side includes the term :math:`\hat{A}^H \hat{A}`,
    which corresponds to the outer product of :math:`\hat{A}^H`
    and :math:`\hat{A}`. A computationally efficient solution is possible,
    however, by exploiting the Woodbury matrix identity

    .. math::

       (D + U G V)^{-1} = D^{-1} - D^{-1} U (G^{-1} + V D^{-1} U)^{-1}
       V D^{-1} \;.

    Setting

    .. math::

       D &= \frac{1}{2 \omega \rho_1} \sum_{i=2}^N \rho_i \hat{C}_i^H
            \hat{C}_i \\
       U &= \hat{A}^H \\
       G &= I \\
       V &= \hat{A}

    we have

    .. math::

       (D + \hat{A}^H \hat{A})^{-1} = D^{-1} - D^{-1} \hat{A}^H
       (I + \hat{A} D^{-1} \hat{A}^H)^{-1} \hat{A} D^{-1}

    which can be simplified to

    .. math::

       (D + \hat{A}^H \hat{A})^{-1} = D^{-1} (I - \hat{A}^H E^{-1}
       \hat{A} D^{-1})

    by defining :math:`E = I + \hat{A} D^{-1} \hat{A}^H`. The right
    hand side is much cheaper to compute because the only matrix
    inversions involve :math:`D`, which is diagonal, and :math:`E`,
    which is a weighted inner product of :math:`\hat{A}^H` and
    :math:`\hat{A}`.
    """

    def __init__(self, ndims: Optional[int] = None, check_solve: bool = False):
        """Initialize a :class:`G0BlockCircularConvolveSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
        """
        self.ndims = ndims
        self.check_solve = check_solve
        self.accuracy: Optional[float] = None

    def internal_init(self, admm: soa.ADMM):
        if admm.f is not None and not isinstance(admm.f, ZeroFunctional):
            raise ValueError(
                "G0BlockCircularConvolveSolver requires f to be None or a ZeroFunctional"
            )
        if not isinstance(admm.g_list[0], SquaredL2Loss):
            raise TypeError(
                "G0BlockCircularConvolveSolver requires g_1 to be a scico.loss.SquaredL2Loss; "
                f"got {type(admm.g_list[0])}."
            )
        if not isinstance(admm.C_list[0], ComposedLinearOperator):
            raise TypeError(
                "G0BlockCircularConvolveSolver requires C_1 to be a composition of Sum "
                f"and CircularConvolve linear operators; got {type(admm.C_list[0])}."
            )

        super().internal_init(admm)

        assert isinstance(self.admm.g_list[0], SquaredL2Loss)
        assert isinstance(self.admm.C_list[0], ComposedLinearOperator)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        c_gram_list = [
            rho * CircularConvolve.from_operator(C.gram_op, ndims=self.ndims)
            for rho, C in zip(admm.rho_list[1:], admm.C_list[1:])
        ]
        D = reduce(lambda a, b: a + b, c_gram_list) / (
            2.0 * self.admm.g_list[0].scale * admm.rho_list[0]
        )
        self.solver = ConvATADSolver(self.admm.C_list[0], D)

    def compute_rhs(self) -> Union[Array, BlockArray]:
        r"""Compute the right hand side of the linear equation to be solved.

        Compute

        .. math::

            C_1^H  ( \mb{z}^{(k)}_1 - \mb{u}^{(k)}_1) +
            \frac{1}{2 \omega \rho_1}\sum_{i=2}^N \rho_i C_i^H
            ( \mb{z}^{(k)}_i - \mb{u}^{(k)}_i) \;.

        Returns:
            Right hand side of the linear equation.
        """
        assert isinstance(self.admm.g_list[0], SquaredL2Loss)

        C0 = self.admm.C_list[0]
        rhs = snp.zeros(C0.input_shape, C0.input_dtype)
        omega = self.admm.g_list[0].scale
        omega_list = [
            2.0 * omega,
        ] + [
            1.0,
        ] * (len(self.admm.C_list) - 1)
        for omegai, rhoi, Ci, zi, ui in zip(
            omega_list, self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
        ):
            rhs += omegai * rhoi * Ci.adj(zi - ui)
        return rhs

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value (unused, has no effect).

        Returns:
            Computed solution.
        """
        assert isinstance(self.admm.g_list[0], SquaredL2Loss)

        rhs = self.compute_rhs() / (2.0 * self.admm.g_list[0].scale * self.admm.rho_list[0])
        x = self.solver.solve(rhs)
        if self.check_solve:
            self.accuracy = self.solver.accuracy(x, rhs)

        return x
