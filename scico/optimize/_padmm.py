# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal ADMM solvers."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import scico.numpy as snp
from scico import cvjp, jvp
from scico.function import Function
from scico.functional import Functional
from scico.linop import Identity, LinearOperator, operator_norm
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm
from scico.typing import BlockShape, DType, PRNGKey, Shape

from ._common import Optimizer

# mypy: disable-error-code=override


class ProximalADMMBase(Optimizer):
    r"""Base class for proximal ADMM solvers.

    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
           :class:`.Loss`).
        g (:class:`.Functional`): Functional :math:`g`.
        rho (scalar): Penalty parameter.
        mu (scalar): First algorithm parameter.
        nu (scalar): Second algorithm parameter.
        x (array-like): Solution variable.
        z (array-like): Auxiliary variables :math:`\mb{z}` at current
          iteration.
        z_old (array-like): Auxiliary variables :math:`\mb{z}` at
          previous iteration.
        u (array-like): Scaled Lagrange multipliers :math:`\mb{u}` at
           current iteration.
        u_old (array-like): Scaled Lagrange multipliers :math:`\mb{u}` at
           previous iteration.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        rho: float,
        mu: float,
        nu: float,
        xshape: Union[Shape, BlockShape],
        zshape: Union[Shape, BlockShape],
        ushape: Union[Shape, BlockShape],
        xdtype: DType,
        zdtype: DType,
        udtype: DType,
        x0: Optional[Union[Array, BlockArray]] = None,
        z0: Optional[Union[Array, BlockArray]] = None,
        u0: Optional[Union[Array, BlockArray]] = None,
        fast_dual_residual: bool = True,
        **kwargs,
    ):
        r"""Initialize a :class:`ProximalADMMBase` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            rho: Penalty parameter.
            mu: First algorithm parameter.
            nu: Second algorithm parameter.
            xshape: Shape of variable :math:`\mb{x}`.
            zshape: Shape of variable :math:`\mb{z}`.
            ushape: Shape of variable :math:`\mb{u}`.
            xdtype: Dtype of variable :math:`\mb{x}`.
            zdtype: Dtype of variable :math:`\mb{z}`.
            udtype: Dtype of variable :math:`\mb{u}`.
            x0: Initial value for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            z0: Initial value for :math:`\mb{z}`. If ``None``, defaults
                to an array of zeros.
            u0: Initial value for :math:`\mb{u}`. If ``None``, defaults
                to an array of zeros.
            fast_dual_residual: Flag indicating whether to use fast
                approximation to the dual residual, or a slower but more
                accurate calculation.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.f: Functional = f
        self.g: Functional = g

        self.rho: float = rho
        self.mu: float = mu
        self.nu: float = nu
        self.fast_dual_residual: bool = fast_dual_residual

        if x0 is None:
            x0 = snp.zeros(xshape, dtype=xdtype)
        self.x = x0
        if z0 is None:
            z0 = snp.zeros(zshape, dtype=zdtype)
        self.z = z0
        self.z_old = self.z
        if u0 is None:
            u0 = snp.zeros(ushape, dtype=udtype)
        self.u = u0
        self.u_old = self.u

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
        return ["x", "z", "z_old", "u", "u_old"]

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
            scalar: Current value of the objective function.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of arguments 'x' and 'z' must be supplied")
        if x is None:
            x = self.x
            z = self.z
        return self.f(x) + self.g(z)


class ProximalADMM(ProximalADMMBase):
    r"""Proximal alternating direction method of multipliers.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + g(\mb{z}) \;
        \text{such that}\; A \mb{x} + B \mb{z} = \mb{c} \;,

    where :math:`f` and :math:`g` are instances of :class:`.Functional`,
    (in most cases :math:`f` will, more specifically be an instance
    of :class:`.Loss`), and :math:`A` and :math:`B` are instances of
    :class:`LinearOperator`.

    The optimization problem is solved via a variant of the proximal ADMM
    algorithm :cite:`deng-2015-global`, consisting of the iterations
    (see :meth:`step`)

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \mu^{-1} f} \left(
         \mb{x}^{(k)} - \mu^{-1} A^T \left(2 \mb{u}^{(k)} -
         \mb{u}^{(k-1)} \right) \right) \\
       \mb{z}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \nu^{-1} g} \left(
         \mb{z}^{(k)} - \nu^{-1} B^T \left(
         A \mb{x}^{(k+1)} + B \mb{z}^{(k)} - \mb{c} + \mb{u}^{(k)}
         \right) \right) \\
       \mb{u}^{(k+1)} &=  \mb{u}^{(k)} + A \mb{x}^{(k+1)} + B
         \mb{z}^{(k+1)} - \mb{c}  \;.
       \end{aligned}

    Parameters :math:`\mu` and :math:`\nu` are required to satisfy

    .. math::
       \mu > \norm{ A }_2^2 \quad \text{and} \quad \nu > \norm{ B }_2^2 \;.


    Attributes:
        A (:class:`.LinearOperator`): :math:`A` linear operator.
        B (:class:`.LinearOperator`): :math:`B` linear operator.
        c (array-like): constant :math:`\mb{c}`.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        A: LinearOperator,
        rho: float,
        mu: float,
        nu: float,
        B: Optional[LinearOperator] = None,
        c: Optional[Union[float, Array, BlockArray]] = None,
        x0: Optional[Union[Array, BlockArray]] = None,
        z0: Optional[Union[Array, BlockArray]] = None,
        u0: Optional[Union[Array, BlockArray]] = None,
        fast_dual_residual: bool = True,
        **kwargs,
    ):
        r"""Initialize a :class:`ProximalADMM` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            A: Linear operator :math:`A`.
            rho: Penalty parameter.
            mu: First algorithm parameter.
            nu: Second algorithm parameter.
            B: Linear operator :math:`B` (if ``None``, :math:`B = -I`
               where :math:`I` is the identity operator).
            c: Constant :math:`\mb{c}`. If ``None``, defaults to zero.
            x0: Starting value for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            z0: Starting value for :math:`\mb{z}`. If ``None``, defaults
                to an array of zeros.
            u0: Starting value for :math:`\mb{u}`. If ``None``, defaults
                to an array of zeros.
            fast_dual_residual: Flag indicating whether to use fast
                approximation to the dual residual, or a slower but more
                accurate calculation.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.A: LinearOperator = A
        if B is None:
            self.B = -Identity(self.A.output_shape, self.A.output_dtype)
        else:
            self.B = B
        if c is None:
            self.c = 0.0
        else:
            self.c = c

        super().__init__(
            f,
            g,
            rho,
            mu,
            nu,
            self.A.input_shape,
            self.B.input_shape,
            self.A.output_shape,
            self.A.input_dtype,
            self.B.input_dtype,
            self.A.output_dtype,
            x0=x0,
            z0=z0,
            u0=u0,
            fast_dual_residual=fast_dual_residual,
            **kwargs,
        )

    def norm_primal_residual(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
        z: Optional[Union[Array, BlockArray]] = None,
    ) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \norm{A \mb{x} + B \mb{z} - \mb{c}}_2 \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.x`.
            z: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.z`.

        Returns:
            Norm of primal residual.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of arguments 'x' and 'z' must be supplied")
        if x is None:
            x = self.x
            z = self.z

        return norm(self.A(x) + self.B(z) - self.c)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual. If the flag
        requesting a fast approximate calculation is set, it is computed
        as

        .. math::
            \norm{\mb{z}^{(k+1)} - \mb{z}^{(k)}}_2 \;,

        otherwise it is computed as

        .. math::
            \norm{A^T B ( \mb{z}^{(k+1)} - \mb{z}^{(k)} ) }_2 \;.

        Returns:
            Current norm of dual residual.
        """
        if self.fast_dual_residual:
            rsdl = self.z - self.z_old  # fast but poor approximation
        else:
            rsdl = self.A.H(self.B(self.z - self.z_old))
        return norm(rsdl)

    def step(self):
        r"""Perform a single algorithm iteration.

        Perform a single algorithm iteration.
        """
        proxarg = self.x - (1.0 / self.mu) * self.A.H(2.0 * self.u - self.u_old)
        self.x = self.f.prox(proxarg, (1.0 / (self.rho * self.mu)), v0=self.x)
        proxarg = self.z - (1.0 / self.nu) * self.B.H(
            self.A(self.x) + self.B(self.z) - self.c + self.u
        )
        self.z_old = self.z
        self.z = self.g.prox(proxarg, (1.0 / (self.rho * self.nu)), v0=self.z)
        self.u_old = self.u
        self.u = self.u + self.A(self.x) + self.B(self.z) - self.c

    @staticmethod
    def estimate_parameters(
        A: LinearOperator,
        B: Optional[LinearOperator] = None,
        factor: Optional[float] = 1.01,
        maxiter: int = 100,
        key: Optional[PRNGKey] = None,
    ) -> Tuple[float, float]:
        r"""Estimate `mu` and `nu` parameters of :class:`ProximalADMM`.

        Find values of the `mu` and `nu` parameters of :class:`ProximalADMM`
        that respect the constraints

        .. math::
           \mu > \norm{ A }_2^2 \quad \text{and} \quad \nu >
           \norm{ B }_2^2 \;.

        Args:
            A: Linear operator :math:`A`.
            B: Linear operator :math:`B` (if ``None``, :math:`B = -I`
               where :math:`I` is the identity operator).
            factor: Safety factor with which to multiply estimated
               operator norms to ensure strict inequality compliance. If
               ``None``, return the estimated squared operator norms.
            maxiter: Maximum number of power iterations to use in operator
               norm estimation (see :func:`.operator_norm`). Default: 100.
            key: Jax PRNG key to use in operator norm estimation (see
               :func:`.operator_norm`). Defaults to ``None``, in which
               case a new key is created.

        Returns:
            A tuple (`mu`, `nu`) representing the estimated parameter
            values or corresponding squared operator norm values,
            depending on the value of the `factor` parameter.
        """
        if B is None:
            B = -Identity(A.output_shape, A.output_dtype)  # type: ignore
        assert isinstance(B, LinearOperator)
        mu = operator_norm(A, maxiter=maxiter, key=key) ** 2
        nu = operator_norm(B, maxiter=maxiter, key=key) ** 2
        if factor is None:
            return (mu, nu)
        else:
            return (factor * mu, factor * nu)


class NonLinearPADMM(ProximalADMMBase):
    r"""Non-linear proximal alternating direction method of multipliers.

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + g(\mb{z}) \;
        \text{such that}\; H(\mb{x}, \mb{z}) = 0 \;,

    where :math:`f` and :math:`g` are instances of :class:`.Functional`,
    (in most cases :math:`f` will, more specifically be an instance
    of :class:`.Loss`), and :math:`H` is a function.

    The optimization problem is solved via a variant of the proximal ADMM
    algorithm for problems with a non-linear operator constraint
    :cite:`benning-2016-preconditioned`, consisting of the
    iterations (see :meth:`step`)

    .. math::
       \begin{aligned}
       A^{(k)} &= J_{\mb{x}} H(\mb{x}^{(k)}, \mb{z}^{(k)}) \\
       \mb{x}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \mu^{-1} f} \left(
         \mb{x}^{(k)} - \mu^{-1} (A^{(k)})^T \left(2 \mb{u}^{(k)} -
         \mb{u}^{(k-1)} \right) \right) \\
       B^{(k)} &= J_{\mb{z}} H(\mb{x}^{(k+1)}, \mb{z}^{(k)}) \\
       \mb{z}^{(k+1)} &= \mathrm{prox}_{\rho^{-1} \nu^{-1} g} \left(
         \mb{z}^{(k)} - \nu^{-1} (B^{(k)})^T \left(
         H(\mb{x}^{(k+1)}, \mb{z}^{(k)}) + \mb{u}^{(k)} \right) \right) \\
       \mb{u}^{(k+1)} &=  \mb{u}^{(k)} + H(\mb{x}^{(k+1)},
         \mb{z}^{(k+1)})  \;.
       \end{aligned}

    Parameters :math:`\mu` and :math:`\nu` are required to satisfy

    .. math::
       \mu > \norm{ A^{(k)} }_2^2 \quad \text{and} \quad \nu > \norm{ B^{(k)} }_2^2

    for all :math:`A^{(k)}` and :math:`B^{(k)}`.


    Attributes:
        H (:class:`.Function`): :math:`H` function.
    """

    def __init__(
        self,
        f: Functional,
        g: Functional,
        H: Function,
        rho: float,
        mu: float,
        nu: float,
        x0: Optional[Union[Array, BlockArray]] = None,
        z0: Optional[Union[Array, BlockArray]] = None,
        u0: Optional[Union[Array, BlockArray]] = None,
        fast_dual_residual: bool = True,
        **kwargs,
    ):
        r"""Initialize a :class:`NonLinearPADMM` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            H: Function :math:`H`.
            rho: Penalty parameter.
            mu: First algorithm parameter.
            nu: Second algorithm parameter.
            x0: Starting value for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            z0: Starting value for :math:`\mb{z}`. If ``None``, defaults
                to an array of zeros.
            u0: Starting value for :math:`\mb{u}`. If ``None``, defaults
                to an array of zeros.
            fast_dual_residual: Flag indicating whether to use fast
                approximation to the dual residual, or a slower but more
                accurate calculation.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        self.H: Function = H

        super().__init__(
            f,
            g,
            rho,
            mu,
            nu,
            H.input_shapes[0],
            H.input_shapes[1],
            H.output_shape,
            H.input_dtypes[0],
            H.input_dtypes[1],
            H.output_dtype,
            x0=x0,
            z0=z0,
            u0=u0,
            fast_dual_residual=fast_dual_residual,
            **kwargs,
        )

    def norm_primal_residual(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
        z: Optional[Union[Array, BlockArray]] = None,
    ) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \norm{H(\mb{x}, \mb{z})}_2 \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.x`.
            z: Point at which to evaluate primal residual. If ``None``,
               the primal residual is evaluated at the current iterate
               :code:`self.z`.

        Returns:
            Norm of primal residual.
        """
        if (x is None) != (z is None):
            raise ValueError("Both or neither of arguments 'x' and 'z' must be supplied")
        if x is None:
            x = self.x
            z = self.z

        return norm(self.H(x, z))

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual. If the flag
        requesting a fast approximate calculation is set, it is computed
        as

        .. math::
            \norm{\mb{z}^{(k+1)} - \mb{z}^{(k)}}_2 \;,

        otherwise it is computed as

        .. math::
            \norm{A^T B ( \mb{z}^{(k+1)} - \mb{z}^{(k)} ) }_2 \;,

        where

        .. math::
            A &= J_{\mb{x}} H(\mb{x}^{(k+1)}, \mb{z}^{(k+1)}) \\
            B &= J_{\mb{z}} H(\mb{x}^{(k+1)}, \mb{z}^{(k+1)}) \;.

        Returns:
            Current norm of dual residual.
        """
        if self.fast_dual_residual:
            rsdl = self.z - self.z_old  # fast but poor approximation
        else:
            Hz = lambda z: self.H(self.x, z)
            B = lambda u: jvp(Hz, (self.z,), (u,))[1]
            Hx = lambda x: self.H(x, self.z)
            AH = cvjp(Hx, self.x)[1]
            rsdl = AH(B(self.z - self.z_old))
        return norm(rsdl)

    def step(self):
        r"""Perform a single algorithm iteration.

        Perform a single algorithm iteration.
        """
        AH = self.H.vjp(0, self.x, self.z, conjugate=True)[1]
        proxarg = self.x - (1.0 / self.mu) * AH(2.0 * self.u - self.u_old)
        self.x = self.f.prox(proxarg, (1.0 / (self.rho * self.mu)), v0=self.x)
        BH = self.H.vjp(1, self.x, self.z, conjugate=True)[1]
        proxarg = self.z - (1.0 / self.nu) * BH(self.H(self.x, self.z) + self.u)
        self.z_old = self.z
        self.z = self.g.prox(proxarg, (1.0 / (self.rho * self.nu)), v0=self.z)
        self.u_old = self.u
        self.u = self.u + self.H(self.x, self.z)

    @staticmethod
    def estimate_parameters(
        H: Function,
        x: Optional[Union[Array, BlockArray]] = None,
        z: Optional[Union[Array, BlockArray]] = None,
        factor: Optional[float] = 1.01,
        maxiter: int = 100,
        key: Optional[PRNGKey] = None,
    ) -> Tuple[float, float]:
        r"""Estimate `mu` and `nu` parameters of :class:`NonLinearPADMM`.

        Find values of the `mu` and `nu` parameters of :class:`NonLinearPADMM`
        that respect the constraints

        .. math::
           \mu > \norm{ J_x H(\mb{x}, \mb{z}) }_2^2 \quad \text{and} \quad
           \nu > \norm{ J_z H(\mb{x}, \mb{z}) }_2^2 \;.

        Args:
            H: Constraint function :math:`H`.
            x: Value of :math:`\mb{x}` at which to evaluate the Jacobian.
               If ``None``, defaults to an array of zeros.
            z: Value of :math:`\mb{z}` at which to evaluate the Jacobian.
               If ``None``, defaults to an array of zeros.
            factor: Safety factor with which to multiply estimated
               operator norms to ensure strict inequality compliance. If
               ``None``, return the estimated squared operator norms.
            maxiter: Maximum number of power iterations to use in operator
               norm estimation (see :func:`.operator_norm`). Default: 100.
            key: Jax PRNG key to use in operator norm estimation (see
               :func:`.operator_norm`). Defaults to ``None``, in which
               case a new key is created.

        Returns:
            A tuple (`mu`, `nu`) representing the estimated parameter
            values or corresponding squared operator norm values,
            depending on the value of the `factor` parameter.
        """
        if x is None:
            x = snp.zeros(H.input_shapes[0], dtype=H.input_dtypes[0])
        if z is None:
            z = snp.zeros(H.input_shapes[1], dtype=H.input_dtypes[1])
        Jx = H.jacobian(0, x, z)
        Jz = H.jacobian(1, x, z)
        mu = operator_norm(Jx, maxiter=maxiter, key=key) ** 2
        nu = operator_norm(Jz, maxiter=maxiter, key=key) ** 2
        if factor is None:
            return (mu, nu)
        else:
            return (factor * mu, factor * nu)
