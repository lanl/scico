# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linearized ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm

from ._common import Optimizer


class LinearizedADMM(Optimizer):
    r"""Linearized alternating direction method of multipliers algorithm.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + g(C \mb{x}) \;,

    where :math:`f` and :math:`g` are instances of :class:`.Functional`,
    (in most cases :math:`f` will, more specifically be an an instance
    of :class:`.Loss`), and :math:`C` is an instance of
    :class:`.LinearOperator`.

    The optimization problem is solved by introducing the splitting
    :math:`\mb{z} = C \mb{x}` and solving

    .. math::
        \argmin_{\mb{x}, \mb{z}} \; f(\mb{x}) + g(\mb{z}) \;
       \text{such that}\; C \mb{x} = \mb{z} \;,

    via a linearized ADMM algorithm :cite:`yang-2012-linearized`
    :cite:`parikh-2014-proximal` (Sec. 4.4.2) consisting of the
    iterations (see :meth:`step`)

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \mathrm{prox}_{\mu f} \left( \mb{x}^{(k)} -
       (\mu / \nu) C^T \left(C \mb{x}^{(k)} - \mb{z}^{(k)} + \mb{u}^{(k)}
       \right) \right) \\
       \mb{z}^{(k+1)} &= \mathrm{prox}_{\nu g} \left(C \mb{x}^{(k+1)} +
       \mb{u}^{(k)} \right) \\
       \mb{u}^{(k+1)} &=  \mb{u}^{(k)} + C \mb{x}^{(k+1)} -
       \mb{z}^{(k+1)}  \;.
       \end{aligned}

    Parameters :math:`\mu` and :math:`\nu` are required to satisfy

    .. math::
       0 < \mu < \nu \| C \|_2^{-2} \;.


    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
           :class:`.Loss`).
        g (:class:`.Functional`): Functional :math:`g`.
        C (:class:`.LinearOperator`): :math:`C` operator.
        mu (scalar): First algorithm parameter.
        nu (scalar): Second algorithm parameter.
        u (array-like): Scaled Lagrange multipliers :math:`\mb{u}` at
           current iteration.
        x (array-like): Solution variable.
        z (array-like): Auxiliary variables :math:`\mb{z}` at current
          iteration.
        z_old (array-like): Auxiliary variables :math:`\mb{z}` at
          previous iteration.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        C: LinearOperator,
        mu: float,
        nu: float,
        x0: Optional[Union[Array, BlockArray]] = None,
        **kwargs,
    ):
        r"""Initialize a :class:`LinearizedADMM` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            C: Operator :math:`C`.
            mu: First algorithm parameter.
            nu: Second algorithm parameter.
            x0: Starting point for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.f: Functional = f
        self.g: Functional = g
        self.C: LinearOperator = C
        self.mu: float = mu
        self.nu: float = nu

        if x0 is None:
            input_shape = C.input_shape
            dtype = C.input_dtype
            x0 = snp.zeros(input_shape, dtype=dtype)
        self.x = x0
        self.z, self.z_old = self.z_init(self.x)
        self.u = self.u_init(self.x)

        super().__init__(**kwargs)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        return (
            snp.all(snp.isfinite(self.x))
            and snp.all(snp.isfinite(self.z))
            and snp.all(snp.isfinite(self.u))
        )

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        return self.f.has_eval and self.g.has_eval

    def _itstat_extra_fields(self):
        """Define linearized ADMM-specific iteration statistics fields."""
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()"]
        return itstat_fields, itstat_attrib

    def _state_variable_names(self) -> List[str]:
        return ["x", "z", "z_old", "u"]

    def minimizer(self) -> Union[Array, BlockArray]:
        return self.x

    def objective(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
        z: Optional[Union[Array, BlockArray]] = None,
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
            scalar: Value of the objective function.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of arguments 'x' and 'z' must be supplied.")
        if x is None:
            x = self.x
            z = self.z
        return self.f(x) + self.g(z)

    def norm_primal_residual(self, x: Optional[Union[Array, BlockArray]] = None) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \norm{C \mb{x} - \mb{z}}_2 \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.x`.

        Returns:
            Norm of primal residual.
        """
        if x is None:
            x = self.x

        return norm(self.C(self.x) - self.z)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \norm{\mb{z}^{(k)} - \mb{z}^{(k-1)}}_2 \;.

        Returns:
            Current norm of dual residual.
        """
        return norm(self.C.adj(self.z - self.z_old))

    def z_init(
        self, x0: Union[Array, BlockArray]
    ) -> Tuple[Union[Array, BlockArray], Union[Array, BlockArray]]:
        r"""Initialize auxiliary variable :math:`\mb{z}`.

        Initialized to

        .. math::
            \mb{z} = C \mb{x}^{(0)} \;.

        :code:`z` and :code:`z_old` are initialized to the same value.

        Args:
            x0: Starting point for :math:`\mb{x}`.
        """
        z = self.C(x0)
        z_old = z
        return z, z_old

    def u_init(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        r"""Initialize scaled Lagrange multiplier :math:`\mb{u}`.

        Initialized to

        .. math::
            \mb{u} = \mb{0} \;.

        Note that the parameter `x0` is unused, but is provided for
        potential use in an overridden method.

        Args:
            x0: Starting point for :math:`\mb{x}`.
        """
        u = snp.zeros(self.C.output_shape, dtype=self.C.output_dtype)
        return u

    def step(self):
        r"""Perform a single linearized ADMM iteration.

        The primary variable :math:`\mb{x}` is updated by computing

        .. math::
            \mb{x}^{(k+1)} = \mathrm{prox}_{\mu f} \left( \mb{x}^{(k)} -
            (\mu / \nu) A^T \left(A \mb{x}^{(k)} - \mb{z}^{(k)} +
            \mb{u}^{(k)} \right) \right) \;.

        The auxiliary variable is updated according to

        .. math::
            \mb{z}^{(k+1)} = \mathrm{prox}_{\nu g} \left(A \mb{x}^{(k+1)}
            + \mb{u}^{(k)} \right) \;,

        and the scaled Lagrange multiplier is updated according to

        .. math::
            \mb{u}^{(k+1)} =  \mb{u}^{(k)} + C \mb{x}^{(k+1)} -
            \mb{z}^{(k+1)} \;.
        """
        proxarg = self.x - (self.mu / self.nu) * self.C.conj().T(self.C(self.x) - self.z + self.u)
        self.x = self.f.prox(proxarg, self.mu, v0=self.x)

        self.z_old = self.z
        Cx = self.C(self.x)
        self.z = self.g.prox(Cx + self.u, self.nu, v0=self.z)
        self.u = self.u + Cx - self.z
