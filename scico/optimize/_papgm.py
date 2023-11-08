"""Proximal Averaged Accelerated Projected Gradient Method."""

from typing import List, Optional, Tuple, Union

import scico.numpy as snp
from scico.functional import Functional
from scico.loss import Loss
from scico.numpy import Array, BlockArray

from ._common import Optimizer


class AcceleratedPAPGM(Optimizer):
    r"""Accelerated Proximal Averaged Projected Gradient Method (AcceleratedPAPGM) base class.

    Minimize a function of the form :math:`f(\mb{x}) + \sum_{i=1}^N \rho_i g_i(\mb{x})`,

    where :math:`f` and the :math:`g` are instances of :class:`.Functional`,
    `rho_i` are positive and non-zero and sum upto 1.
    This modifies FISTA to handle the case of composite prior minimization.
    :cite:`yaoliang-2013-nips`.

    """

    def __init__(
        self,
        f: Union[Loss, Functional],
        g_list: List[Functional],
        rho_list: List[float],
        L0: float,
        x0: Union[Array, BlockArray],
        **kwargs,
    ):
        r"""
        Args:
            f: (:class:`.Functional`): Functional :math:`f` (usually a
            :class:`.Loss`)
            g_list: (list of :class:`.Functional`): List of :math:`g_i`
            functionals. Must be same length as :code:`rho_list`.
            rho_list: (list of scalars): List of :math:`\rho_i` penalty
            parameters. Must be same length as :code:`g_list` and sum to 1.
            L0: (float): Initial estimate of Lipschitz constant of f.
            x0: (array-like): Starting point for :math:`\mb{x}`.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.f: Union[Loss, Functional] = f
        self.g_list: List[Functional] = g_list
        self.rho_list: List[float] = rho_list
        self.x: Union[Array, BlockArray] = x0
        self.fixed_point_residual: float = snp.inf
        self.v: Union[Array, BlockArray] = x0
        self.t: float = 1.0
        self.L: float = L0

        super().__init__(**kwargs)

    def step(self):
        """Take a single AcceleratedPAPGM step."""
        assert snp.sum(snp.array(self.rho_list)) == 1
        assert snp.all(snp.array([rho >= 0 for rho in self.rho_list]))

        x_old = self.x
        z = self.v - 1.0 / self.L * self.f.grad(self.v)

        self.fixed_point_residual = 0
        self.x = snp.zeros_like(z)
        for gi, rhoi in zip(self.g_list, self.rho_list):
            self.x += rhoi * gi.prox(z, 1.0 / self.L)
        self.fixed_point_residual += snp.linalg.norm(self.x - self.v)

        t_old = self.t
        self.t = 0.5 * (1 + snp.sqrt(1 + 4 * t_old**2))
        self.v = self.x + ((t_old - 1) / self.t) * (self.x - x_old)

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        return snp.all(snp.isfinite(self.x)) and snp.all(snp.isfinite(self.v))

    def minimizer(self):
        """Return current estimate of the functional mimimizer."""
        return self.x

    def objective(self, x: Optional[Union[Array, BlockArray]] = None) -> float:
        r"""Evaluate the objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(\mb{x}_i) \;.

        Args:
            x: Point at which to evaluate objective function. If ``None``,
                the objective is  evaluated at the current iterate
                :code:`self.x`.

        Returns:
            Value of the objective function.
        """
        if x is None:
            x = self.x
        out = 0.0
        if self.f:
            out += self.f(x)
        for gi, rhoi in zip(self.g_list, self.rho_list):
            out += rhoi * gi(x)
        return out

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        return (not self.f or self.f.has_eval) and all([_.has_eval for _ in self.g_list])

    def _itstat_extra_fields(self):
        """Define AcceleratedPAPGM iteration statistics fields."""
        itstat_fields = {"L": "%9.3e", "Residual": "%9.3e"}
        itstat_attrib = ["L", "norm_residual()"]
        return itstat_fields, itstat_attrib

    def norm_residual(self) -> float:
        r"""Return the fixed point residual.

        Return the fixed point residual (see Sec. 4.3 of
        :cite:`liu-2018-first`).
        """
        return self.fixed_point_residual
