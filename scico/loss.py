# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.


"""Loss function classes."""


from copy import copy
from functools import wraps
from typing import Callable, Optional, Union

import scico.numpy as snp
from scico import functional, linop, operator
from scico.array import ensure_on_device
from scico.blockarray import BlockArray
from scico.scipy.special import gammaln
from scico.solver import cg
from scico.typing import JaxArray


def _loss_mul_div_wrapper(func):
    @wraps(func)
    def wrapper(self, other):
        if snp.isscalar(other) or isinstance(other, jax.core.Tracer):
            return func(self, other)

        raise NotImplementedError(
            f"Operation {func} not defined between {type(self)} and {type(other)}"
        )

    return wrapper


class Loss(functional.Functional):
    r"""Generic loss function.

    Generic loss function

    .. math::
        \alpha l(\mb{y}, A(\mb{x})) \;,

    where :math:`\alpha` is the scaling parameter and :math:`l(\cdot)` is
    the loss functional.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""
        Args:
            y: Measurement.
            A: Forward operator. Defaults to ``None``, in which case
               ``self.A`` is a :class:`.Identity`.
            scale: Scaling parameter. Default: 0.5.

        """
        self.y = ensure_on_device(y)
        if A is None:
            # y and x must have same shape
            A = linop.Identity(self.y.shape)
        self.A = A
        self.scale = scale

        # Set functional-specific flags
        self.has_prox = False
        self.has_eval = True

        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this loss at point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate loss.
        """
        raise NotImplementedError

    @_loss_mul_div_wrapper
    def __mul__(self, other):
        new_loss = copy(self)
        new_loss.set_scale(self.scale * other)
        return new_loss

    def __rmul__(self, other):
        return self.__mul__(other)

    @_loss_mul_div_wrapper
    def __truediv__(self, other):
        new_loss = copy(self)
        new_loss.set_scale(self.scale / other)
        return new_loss

    def set_scale(self, new_scale: float):
        r"""Update the scale attribute."""
        self.scale = new_scale


class WeightedSquaredL2Loss(Loss):
    r"""Weighted squared :math:`\ell_2` loss.

    Weighted squared :math:`\ell_2` loss

    .. math::
        \alpha \norm{\mb{y} - A(\mb{x})}_W^2 =
        \alpha \left(\mb{y} - A(\mb{x})\right)^T W \left(\mb{y} -
        A(\mb{x})\right) \;,

    where :math:`\alpha` is the scaling parameter and :math:`W` is an
    instance of :class:`scico.linop.Diagonal`. If :math:`W` is None,
    reverts to the behavior of :class:`.SquaredL2Loss`.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        W: Optional[linop.Diagonal] = None,
        prox_kwargs: dict = {"maxiter": 1000, "tol": 1e-12},
    ):

        r"""
        Args:
            y: Measurement.
            A: Forward operator. If ``None``, defaults to :class:`.Identity`.
            scale: Scaling parameter.
            W: Weighting diagonal operator. Must be non-negative.
                If ``None``, defaults to :class:`.Identity`.
        """
        y = ensure_on_device(y)

        self.W: linop.Diagonal

        if W is None:
            self.W = linop.Identity(y.shape)
        elif isinstance(W, linop.Diagonal):
            if snp.all(W.diagonal >= 0):
                self.W = W
            else:
                raise ValueError(f"The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}")

        super().__init__(y=y, A=A, scale=scale)

        if prox_kwargs is None:
            prox_kwargs = dict
        self.prox_kwargs = prox_kwargs

        if isinstance(self.A, linop.LinearOperator):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * (self.W.diagonal * snp.abs(self.y - self.A(x)) ** 2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        if not isinstance(self.A, linop.LinearOperator):
            raise NotImplementedError(
                f"prox is not implemented for {type(self)} when `A` is {type(self.A)}; "
                "must be LinearOperator"
            )

        if isinstance(self.A, linop.Diagonal):
            c = 2.0 * self.scale * lam
            A = self.A.diagonal
            W = self.W.diagonal
            lhs = c * A.conj() * W * self.y + v
            ATWA = c * A.conj() * W * A
            return lhs / (ATWA + 1.0)

        #   prox_{f}(v) = arg min  1/2 || v - x ||^2 + λ α || A x - y ||^2_W
        #                    x
        # solution at:
        #
        #   (I + λ 2α A^T W A) x = v + λ 2α A^T W y
        #
        W = self.W
        A = self.A
        α = self.scale
        y = self.y
        if "x0" in kwargs and kwargs["x0"] is not None:
            x0 = kwargs["x0"]
        else:
            x0 = snp.zeros_like(v)
        hessian = self.hessian  # = (2α A^T W A)
        lhs = linop.Identity(v.shape) + lam * hessian
        rhs = v + 2 * lam * α * A.adj(W(y))
        x, _ = cg(lhs, rhs, x0, **self.prox_kwargs)
        return x

    @property
    def hessian(self) -> linop.LinearOperator:
        r"""Compute the hessian of a linear operator.

        If ``self.A`` is a :class:`scico.linop.LinearOperator`, returns a
        :class:`scico.linop.LinearOperator` corresponding to  the Hessian
        :math:`2 \alpha \mathrm{A^H W A}`. Otherwise not implemented.
        """
        A = self.A
        W = self.W
        if isinstance(A, linop.LinearOperator):
            return linop.LinearOperator(
                input_shape=A.input_shape,
                output_shape=A.input_shape,
                eval_fn=lambda x: 2 * self.scale * A.adj(W(A(x))),
                adj_fn=lambda x: 2 * self.scale * A.adj(W(A(x))),
            )

        raise NotImplementedError(
            f"Hessian is not implemented for {type(self)} when `A` is {type(A)}; "
            "must be LinearOperator"
        )


class SquaredL2Loss(WeightedSquaredL2Loss):
    r"""Squared :math:`\ell_2` loss.

    Squared :math:`\ell_2` loss

    .. math::
        \alpha \norm{\mb{y} - A(\mb{x})}_2^2 \;,

    where :math:`\alpha` is the scaling parameter.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        prox_kwargs: dict = {"maxiter": 1000, "tol": 1e-12},
    ):
        r"""
        Args:
            y: Measurement.
            A: Forward operator. If ``None``, defaults to :class:`.Identity`.
            scale: Scaling parameter.
        """
        super().__init__(y=y, A=A, scale=scale, W=None, prox_kwargs=prox_kwargs)


class PoissonLoss(Loss):
    r"""Poisson negative log likelihood loss.

    Poisson negative log likelihood loss

    .. math::
        \alpha \left( \sum_i [A(x)]_i - y_i \log\left( [A(x)]_i \right) +
        \log(y_i!) \right) \;,

    where :math:`\alpha` is the scaling parameter.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""
        Args:
            y: Measurement.
            A: Forward operator. Defaults to ``None``, in which case
                ``self.A`` is a :class:`.Identity`.
            scale: Scaling parameter. Default: 0.5.
        """
        y = ensure_on_device(y)
        super().__init__(y=y, A=A, scale=scale)

        #: Constant term in Poisson log likehood; equal to ln(y!)
        self.const = gammaln(self.y + 1.0)  # ln(y!)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        Ax = self.A(x)
        return self.scale * snp.sum(Ax - self.y * snp.log(Ax) + self.const)
