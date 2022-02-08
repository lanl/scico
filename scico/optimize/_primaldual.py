# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Primal-dual solvers."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import Callable, Optional, Union

import scico.numpy as snp
from scico.array import ensure_on_device
from scico.blockarray import BlockArray
from scico.diagnostics import IterationStats
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy.linalg import norm
from scico.typing import JaxArray
from scico.util import Timer


class PDHG:
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
    :class:`.LinearOperator`.

    The algorithm iterations are

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

    Attributes:
        f (:class:`.Functional`): Functional :math:`f` (usually a
          :class:`.Loss`).
        g (:class:`.Functional`): Functional :math:`g`.
        C (:class:`.LinearOperator`): :math:`C` operator.
        itnum (int): Iteration counter.
        maxiter (int): Number of ADMM outer-loop iterations.
        timer (:class:`.Timer`): Iteration timer.
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
        C: LinearOperator,
        tau: float,
        sigma: float,
        alpha: float = 1.0,
        x0: Optional[Union[JaxArray, BlockArray]] = None,
        z0: Optional[Union[JaxArray, BlockArray]] = None,
        maxiter: int = 100,
        itstat_options: Optional[dict] = None,
    ):
        r"""Initialize a :class:`PDHG` object.

        Args:
            f: Functional :math:`f` (usually a loss function).
            g: Functional :math:`g`.
            C: Operator :math:`C`.
            tau: First algorithm parameter.
            sigma: Second algorithm parameter.
            alpha: Relaxation parameter.
            x0: Starting point for :math:`\mb{x}`. If None, defaults to
               an array of zeros.
            z0: Starting point for :math:`\mb{z}`. If None, defaults to
               an array of zeros.
            maxiter: Number of ADMM outer-loop iterations. Default: 100.
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
        self.f: Functional = f
        self.g: Functional = g
        self.C: LinearOperator = C
        self.tau: float = tau
        self.sigma: float = sigma
        self.alpha: float = alpha
        self.itnum: int = 0
        self.maxiter: int = maxiter
        self.timer: Timer = Timer()

        # iteration number and time fields
        itstat_fields = {
            "Iter": "%d",
            "Time": "%8.2e",
        }
        itstat_attrib = ["itnum", "timer.elapsed()"]
        # objective function can be evaluated if 'g' function can be evaluated
        if g.has_eval:
            itstat_fields.update({"Objective": "%9.3e"})
            itstat_attrib.append("objective()")
        # primal and dual residual fields
        itstat_fields.update({"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"})
        itstat_attrib.extend(["norm_primal_residual()", "norm_dual_residual()"])

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
            input_shape = C.input_shape
            dtype = C.input_dtype
            x0 = snp.zeros(input_shape, dtype=dtype)
        self.x = ensure_on_device(x0)
        self.x_old = self.x
        if z0 is None:
            input_shape = C.output_shape
            dtype = C.output_dtype
            z0 = snp.zeros(input_shape, dtype=dtype)
        self.z = ensure_on_device(z0)
        self.z_old = self.z

    def objective(
        self,
        x: Optional[Union[JaxArray, BlockArray]] = None,
    ) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + g(C \mb{x}) \;.

        Args:
            x: Point at which to evaluate objective function. If `None`,
                the objective is evaluated at the current iterate
                :code:`self.x`

        Returns:
            scalar: Current value of the objective function.
        """
        if x is None:
            x = self.x
        out = self.f(x) + self.g(self.C(x))
        return out

    def norm_primal_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \tau^{-1} \norm{\mb{x}^{(k)} - \mb{x}^{(k-1)}}_2 \;.

        Returns:
            Current value of primal residual.
        """

        return norm(self.x - self.x_old) / self.tau

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \sigma^{-1} \norm{\mb{z}^{(k)} - \mb{z}^{(k-1)}}_2 \;.

        Returns:
            Current value of dual residual.

        """
        return norm(self.z - self.z_old) / self.sigma

    def step(self):
        """Perform a single iteration."""
        self.x_old = self.x
        self.z_old = self.z
        proxarg = self.x - self.tau * self.C.conj().T(self.z)
        self.x = self.f.prox(proxarg, self.tau, v0=self.x)
        proxarg = self.z + self.sigma * self.C(
            (1.0 + self.alpha) * self.x - self.alpha * self.x_old
        )
        self.z = self.g.conj_prox(proxarg, self.sigma, v0=self.z)

    def solve(
        self,
        callback: Optional[Callable[[PDHG], None]] = None,
    ) -> Union[JaxArray, BlockArray]:
        r"""Initialize and run the PDHG algorithm.

        Initialize and run the PDHG algorithm for a total of
        ``self.maxiter`` iterations.

        Args:
            callback: An optional callback function, taking an a single
               argument of type :class:`PDHG`, that is called at the end
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
