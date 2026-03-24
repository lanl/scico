# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM auxiliary classes (convolutional subproblem solvers)."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import reduce
from typing import Optional, Type, Union

try:
    from jaxdecomp.fft import pfft3d, pifft3d

    have_jaxdecomp = True
except ImportError:
    have_jaxdecomp = False

from jax import Device
from jax.sharding import Sharding

import scico.numpy as snp
import scico.optimize.admm as soa
from scico.functional import ZeroFunctional
from scico.linop import (
    CircularConvolve,
    CircularConvolve3D,
    ComposedLinearOperator,
    Identity,
)
from scico.loss import SquaredL2Loss
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_real_dtype
from scico.solver import ConvATADSolver

from ._admmaux import LinearSubproblemSolver, SubproblemSolver


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

    def __init__(
        self,
        ndims: Optional[int] = None,
        conv_class: Type[CircularConvolve] = CircularConvolve,
        device: Optional[Union[Device, Sharding]] = None,
    ):
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
            conv_class: Circular convolution linear operator class for
                convolution operations.
            device: Device or sharding for new arrays.
        """
        self.ndims = ndims
        self.conv_class = conv_class
        self.device = device

    def internal_init(self, admm: soa.ADMM):
        if admm.f is None:
            is_cc = [isinstance(C, self.conv_class) for C in admm.C_list]
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
            if not isinstance(admm.f.A, (self.conv_class, Identity)):
                raise TypeError(
                    "CircularConvoleSolver requires f.A to be a scico.linop.self.conv_class "
                    f"or scico.linop.Identity; got {type(admm.f.A)}."
                )
            auto_ndims = admm.f.A.ndims if isinstance(admm.f.A, self.conv_class) else None

        if self.ndims is None:
            self.ndims = auto_ndims

        SubproblemSolver.internal_init(self, admm)

        self.real_result = is_real_dtype(admm.C_list[0].input_dtype)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        lhs_op_list = [
            rho * self.conv_class.from_operator(C.gram_op, ndims=self.ndims, device=self.device)
            for rho, C in zip(admm.rho_list, admm.C_list)
        ]
        A_lhs = reduce(lambda a, b: a + b, lhs_op_list)
        if self.admm.f is not None:
            A_lhs += (
                2.0
                * admm.f.scale
                * self.conv_class.from_operator(
                    admm.f.A.gram_op, ndims=self.ndims, device=self.device
                )
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


class CircularConvolve3DSolver(CircularConvolveSolver):
    """Solver for 3D linear operators diagonalized in the DFT domain.

    This class specializes :class:`CircularConvolveSolver` for
    three-dimensional arrays, with the advantage of making use of
    sharding-efficient FFT operations (including via use of
    :class:`CircularConvolve3D` rather than :class:`CircularConvolve`.
    """

    def __init__(
        self,
        device: Optional[Union[Device, Sharding]] = None,
    ):
        """Initialize a :class:`CircularConvolve3DSolver` object.

        Args:
            device: Device or sharding for new arrays.
        """
        if not have_jaxdecomp:
            raise RuntimeError(
                "Package jaxdecomp is required for use of class CircularConvolve3DSolver."
            )

        super().__init__(ndims=3, conv_class=CircularConvolve3D, device=device)

    def solve(self, x0: Union[Array, BlockArray]) -> Union[Array, BlockArray]:
        """Solve the ADMM step.

        Args:
            x0: Initial value (unused, has no effect).

        Returns:
            Computed solution.
        """
        rhs = self.compute_rhs()
        rhs_dft = pfft3d(rhs)
        x_dft = rhs_dft / self.A_lhs.h_dft
        x = pifft3d(x_dft)
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

    def __init__(
        self,
        ndims: Optional[int] = None,
        check_solve: bool = False,
        device: Optional[Union[Device, Sharding]] = None,
    ):
        """Initialize a :class:`FBlockCircularConvolveSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
            device: Device or sharding for new arrays.
        """
        self.ndims = ndims
        self.check_solve = check_solve
        self.accuracy: Optional[float] = None
        self.device = device

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

        SubproblemSolver.internal_init(self, admm)

        assert isinstance(self.admm.f, SquaredL2Loss)
        assert isinstance(self.admm.f.A, ComposedLinearOperator)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        c_gram_list = [
            rho * CircularConvolve.from_operator(C.gram_op, ndims=self.ndims, device=self.device)
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

    def __init__(
        self,
        ndims: Optional[int] = None,
        check_solve: bool = False,
        device: Optional[Union[Device, Sharding]] = None,
    ):
        """Initialize a :class:`G0BlockCircularConvolveSolver` object.

        Args:
            check_solve: If ``True``, compute solver accuracy after each
                solve.
            device: Device or sharding for new arrays.
        """
        self.ndims = ndims
        self.check_solve = check_solve
        self.accuracy: Optional[float] = None
        self.device = device

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

        SubproblemSolver.internal_init(self, admm)

        assert isinstance(self.admm.g_list[0], SquaredL2Loss)
        assert isinstance(self.admm.C_list[0], ComposedLinearOperator)

        # All of the C operators are assumed to be linear and shift invariant
        # but this is not checked.
        c_gram_list = [
            rho * CircularConvolve.from_operator(C.gram_op, ndims=self.ndims, device=self.device)
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
        rhs = snp.zeros(C0.input_shape, C0.input_dtype, device=self.device)
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
