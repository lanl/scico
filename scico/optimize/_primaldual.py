# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Primal-dual solvers."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Union

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator, jacobian, operator_norm
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm
from scico.operator import Operator
from scico.typing import PRNGKey

from ._common import Optimizer


class PDHG(Optimizer):
    r"""Primal–dual hybrid gradient (PDHG) algorithm.

    |

    Primal–dual hybrid gradient (PDHG) is a family of algorithms
    :cite:`esser-2010-general` that includes the Chambolle-Pock
    primal-dual algorithm :cite:`chambolle-2010-firstorder`. The form
    implemented here is a minor variant :cite:`pock-2011-diagonal` of the
    original Chambolle-Pock algorithm.

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + g(C \mb{x}) \;,

    where :math:`f` and :math:`g` are instances of :class:`.Functional`,
    (in most cases :math:`f` will, more specifically be an an instance
    of :class:`.Loss`), and :math:`C` is an instance of
    :class:`.Operator` or :class:`.LinearOperator`.

    When `C` is a :class:`.LinearOperator`, the algorithm iterations are

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \mathrm{prox}_{\tau f} \left( \mb{x}^{(k)} -
       \tau C^T \mb{z}^{(k)} \right) \\
       \mb{z}^{(k+1)} &= \mathrm{prox}_{\sigma g^*} \left( \mb{z}^{(k)}
       + \sigma C((1 + \alpha) \mb{x}^{(k+1)} - \alpha \mb{x}^{(k)}
       \right) \;,
       \end{aligned}

    where :math:`g^*` denotes the convex conjugate of :math:`g`.
    Parameters :math:`\tau > 0` and :math:`\sigma > 0` are also required
    to satisfy

    .. math::
       \tau \sigma < \| C \|_2^{-2} \;,

    and it is required that :math:`\alpha \in [0, 1]`.

    When `C` is a non-linear :class:`.Operator`, a non-linear PDHG variant
    :cite:`valkonen-2014-primal` is used, with the same iterations except
    for :math:`\mb{x}` update

    .. math::
       \mb{x}^{(k+1)} = \mathrm{prox}_{\tau f} \left( \mb{x}^{(k)} -
       \tau [J_x C(\mb{x}^{(k)})]^T \mb{z}^{(k)} \right) \;.


    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
          :class:`.Loss`).
        g (:class:`.Functional`): Functional :math:`g`.
        C (:class:`.Operator`): :math:`C` operator.
        tau (scalar): First algorithm parameter.
        sigma (scalar): Second algorithm parameter.
        alpha (scalar): Relaxation parameter.
        x (array-like): Primal variable :math:`\mb{x}` at current
          iteration.
        x_old (array-like): Primal variable :math:`\mb{x}` at previous
          iteration.
        z (array-like): Dual variable :math:`\mb{z}` at current
          iteration.
        z_old (array-like): Dual variable :math:`\mb{z}` at previous
          iteration.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        C: Operator,
        tau: float,
        sigma: float,
        alpha: float = 1.0,
        x0: Optional[Union[Array, BlockArray]] = None,
        z0: Optional[Union[Array, BlockArray]] = None,
        **kwargs,
    ):
        r"""Initialize a :class:`PDHG` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            C: Operator :math:`C`.
            tau: First algorithm parameter.
            sigma: Second algorithm parameter.
            alpha: Relaxation parameter.
            x0: Starting point for :math:`\mb{x}`. If ``None``, defaults
               to an array of zeros.
            z0: Starting point for :math:`\mb{z}`. If ``None``, defaults
               to an array of zeros.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.f: Functional = f
        self.g: Functional = g
        self.C: Operator = C
        self.tau: float = tau
        self.sigma: float = sigma
        self.alpha: float = alpha

        if x0 is None:
            input_shape = C.input_shape
            dtype = C.input_dtype
            x0 = snp.zeros(input_shape, dtype=dtype)
        self.x = x0
        self.x_old = self.x
        if z0 is None:
            input_shape = C.output_shape
            dtype = C.output_dtype
            z0 = snp.zeros(input_shape, dtype=dtype)
        self.z = z0
        self.z_old = self.z

        super().__init__(**kwargs)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        return snp.all(snp.isfinite(self.x)) and snp.all(snp.isfinite(self.z))

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        return self.f.has_eval and self.g.has_eval

    def _itstat_extra_fields(self):
        """Define linearized ADMM-specific iteration statistics fields."""
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()"]
        return itstat_fields, itstat_attrib

    def _state_variable_names(self) -> List[str]:
        return ["x", "x_old", "z", "z_old"]

    def minimizer(self) -> Union[Array, BlockArray]:
        return self.x

    def objective(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
    ) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + g(C \mb{x}) \;.

        Args:
            x: Point at which to evaluate objective function. If ``None``,
                the objective is evaluated at the current iterate
                :code:`self.x`

        Returns:
            scalar: Value of the objective function.
        """
        if x is None:
            x = self.x
        return self.f(x) + self.g(self.C(x))

    def norm_primal_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \tau^{-1} \norm{\mb{x}^{(k)} - \mb{x}^{(k-1)}}_2 \;.

        Returns:
            Current norm of primal residual.
        """

        return norm(self.x - self.x_old) / self.tau  # type: ignore

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \sigma^{-1} \norm{\mb{z}^{(k)} - \mb{z}^{(k-1)}}_2 \;.

        Returns:
            Current norm of dual residual.

        """
        return norm(self.z - self.z_old) / self.sigma

    def step(self):
        """Perform a single iteration."""
        self.x_old = self.x
        self.z_old = self.z
        if isinstance(self.C, LinearOperator):
            proxarg = self.x - self.tau * self.C.conj().T(self.z)
        else:
            proxarg = self.x - self.tau * self.C.vjp(self.x, conjugate=True)[1](self.z)
        self.x = self.f.prox(proxarg, self.tau, v0=self.x)
        proxarg = self.z + self.sigma * self.C(
            (1.0 + self.alpha) * self.x - self.alpha * self.x_old
        )
        self.z = self.g.conj_prox(proxarg, self.sigma, v0=self.z)

    @staticmethod
    def estimate_parameters(
        C: Operator,
        x: Optional[Union[Array, BlockArray]] = None,
        ratio: float = 1.0,
        factor: Optional[float] = 1.01,
        maxiter: int = 100,
        key: Optional[PRNGKey] = None,
    ):
        r"""Estimate `tau` and `sigma` parameters of :class:`PDHG`.

        Find values of the `tau` and `sigma` parameters of :class:`PDHG`
        that respect the constraint

        .. math::
           \tau \sigma < \| C \|_2^{-2} \quad \text{or} \quad
           \tau \sigma < \| J_x C(\mb{x}) \|_2^{-2} \;,

        depending on whether :math:`C` is a :class:`.LinearOperator` or
        not.

        Args:
            C: Operator :math:`C`.
            x: Value of :math:`\mb{x}` at which to evaluate the Jacobian
               of :math:`C` (when it is not a :class:`.LinearOperator`).
               If ``None``, defaults to an array of zeros.
            ratio: Desired ratio between return :math:`\tau` and
               :math:`\sigma` values (:math:`\sigma = \mathrm{ratio}
               \tau`).
            factor: Safety factor with which to multiply :math:`\| C
               \|_2^{-2}` to ensure strict inequality compliance. If
               ``None``, the value is set to 1.0.
            maxiter: Maximum number of power iterations to use in operator
               norm estimation (see :func:`.operator_norm`). Default: 100.
            key: Jax PRNG key to use in operator norm estimation (see
               :func:`.operator_norm`). Defaults to ``None``, in which
               case a new key is created.

        Returns:
            A tuple (`tau`, `sigma`) representing the estimated parameter
            values.
        """
        if x is None:
            x = snp.zeros(C.input_shape, dtype=C.input_dtype)
        if factor is None:
            factor = 1.0
        if isinstance(C, LinearOperator):
            J = C
        else:
            J = jacobian(C, x)
        Cnrm = operator_norm(J, maxiter=maxiter, key=key)
        tau = snp.sqrt(factor / ratio) / Cnrm
        sigma = ratio * tau
        return (tau, sigma)
