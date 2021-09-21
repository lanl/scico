# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
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
from scico.blockarray import BlockArray
from scico.scipy.special import gammaln
from scico.typing import JaxArray
from scico.util import ensure_on_device

__author__ = """\n""".join(
    ["Luke Pfister <pfister@lanl.gov>", "Thilo Balke <thilo.balke@gmail.com>"]
)


def _loss_mul_div_wrapper(func):
    @wraps(func)
    def wrapper(self, other):
        if snp.isscalar(other) or isinstance(other, jax.core.Tracer):
            return func(self, other)
        else:
            raise NotImplementedError(
                f"Operation {func} not defined between {type(self)} and {type(other)}"
            )

    return wrapper


class Loss(functional.Functional):
    r"""Generic Loss function.

    .. math::
        \mathrm{scale} \cdot l(\mb{y}, A(\mb{x})) \;

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""Initialize a :class:`Loss` object.

        Args:
            y : Measurements
            A : Forward operator.  Defaults to None.  If None, ``self.A`` is a :class:`.Identity`.
            scale : Scaling parameter.  Default: 0.5.

        """
        self.y = ensure_on_device(y)
        if A is None:
            # y & x must have same shape
            A = linop.Identity(self.y.shape)
        self.A = A
        self.scale = scale

        # Set functional-specific flags
        self.has_prox = False  # TODO: implement a generic prox solver?
        self.has_eval = True

        #: True if :math:`l(\mb{y}, A(\mb{x})` is quadratic in :math:`\mb{x}`.
        self.is_quadratic: bool = False

        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this loss at point :math:`\mb{x}`.

        Args:
            x : Point at which to evaluate loss.
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


class SquaredL2Loss(Loss):
    r"""
    Squared :math:`\ell_2` loss.

    .. math::
        \mathrm{scale} \cdot \norm{\mb{y} - A(\mb{x})}_2^2 \;

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        y = ensure_on_device(y)
        self.functional = scale * functional.SquaredL2Norm()
        super().__init__(y=y, A=A, scale=scale)

        if isinstance(A, operator.Operator):
            self.is_smooth = A.is_smooth
        else:
            self.is_smooth = None

        if isinstance(self.A, linop.LinearOperator):
            self.is_quadratic = True

        if isinstance(self.A, linop.Diagonal):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this loss at point :math:`\mb{x}`.

        Args:
            x : Point at which to evaluate loss.
        """
        return self.functional(self.y - self.A(x))

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        if isinstance(self.A, linop.Diagonal):
            c = 2.0 * self.scale
            A = self.A
            lhs = c * lam * A.adj(self.y) + x
            return lhs / (c * lam * snp.abs(A.diagonal) ** 2 + 1.0)
        else:
            raise NotImplementedError

    @property
    def hessian(self) -> linop.LinearOperator:
        r"""If ``self.A`` is a :class:`.LinearOperator`, returns a new :class:`.LinearOperator` corresponding
        to Hessian :math:`\mathrm{A^*A}`.

        Otherwise not implemented.
        """
        if isinstance(self.A, linop.LinearOperator):
            return 2 * self.scale * self.A.gram_op
        else:
            raise NotImplementedError(
                f"Hessian is not implemented for {type(self)} when `A` is {type(self.A)}; must be LinearOperator"
            )


class WeightedSquaredL2Loss(Loss):
    r"""
    Weighted squared :math:`\ell_2` loss.

    .. math::
        \mathrm{scale} \cdot \norm{\mb{y} - A(\mb{x})}_{\mathrm{W}}^2 =
        \mathrm{scale} \cdot \norm{\mathrm{W}^{1/2} \left( \mb{y} - A(\mb{x})\right)}_2^2\;

    Where :math:`\mathrm{W}` is an instance of :class:`scico.linop.LinearOperator`.  If
    :math:`\mathrm{W}` is None, reverts to the behavior of :class:`.SquaredL2Loss`.

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        weight_op: Optional[operator.Operator] = None,
    ):

        r"""Initialize a :class:`WeightedSquaredL2Loss` object.

        Args:
            y : Measurements
            A : Forward operator.  If None, defaults to :class:`.Identity`.
            scale : Scaling parameter
            weight_op:  Weighting linear operator. Corresponds to :math:`W^{1/2}`
                in the standard definition of the weighted squared :math:`\ell_2` loss.
                If None, defaults to :class:`.Identity`.
        """
        y = ensure_on_device(y)

        self.weight_op: operator.Operator

        self.functional = scale * functional.SquaredL2Norm()
        if weight_op is None:
            self.weight_op = linop.Identity(y.shape)
        elif isinstance(weight_op, linop.LinearOperator):
            self.weight_op = weight_op
        else:
            raise TypeError(f"weight_op must be None or a LinearOperator, got {type(weight_op)}")
        super().__init__(y=y, A=A, scale=scale)

        if isinstance(A, operator.Operator):
            self.is_smooth = A.is_smooth
        else:
            self.is_smooth = None

        if isinstance(self.A, linop.LinearOperator):
            self.is_quadratic = True

        if isinstance(self.A, linop.Diagonal) and isinstance(self.weight_op, linop.Diagonal):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.functional(self.weight_op(self.y - self.A(x)))

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        if isinstance(self.A, linop.Diagonal):
            c = 2.0 * self.scale
            A = self.A.diagonal
            W = self.weight_op.diagonal
            lhs = c * lam * A.conj() * W * self.y + x
            ATWA = W * snp.abs(A) ** 2
            return lhs / (c * lam * ATWA + 1.0)
        else:
            raise NotImplementedError

    @property
    def hessian(self) -> linop.LinearOperator:
        r"""If ``self.A`` is a :class:`scico.linop.LinearOperator`, returns a
        :class:`scico.linop.LinearOperator` corresponding to Hessian :math:`\mathrm{A^* W A}`.

        Otherwise not implemented.
        """
        if isinstance(self.A, linop.LinearOperator):
            return linop.LinearOperator(
                input_shape=self.A.input_shape,
                output_shape=self.A.input_shape,
                eval_fn=lambda x: 2 * self.scale * self.A.adj(self.weight_op(self.A(x))),
                adj_fn=lambda x: 2 * self.scale * self.A.adj(self.weight_op(self.A(x))),
            )
        else:
            raise NotImplementedError(
                f"Hessian is not implemented for {type(self)} when `A` is {type(self.A)}; must be LinearOperator"
            )


class PoissonLoss(Loss):
    r"""
    Poisson negative log likelihood loss

    .. math::
        \mathrm{scale} \cdot \sum_i [A(x)]_i - y_i \log\left( [A(x)]_i \right) + \log(y_i!)

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""Initialize a :class:`Loss` object.

        Args:
            y : Measurements
            A : Forward operator.  Defaults to None.  If None, ``self.A`` is a :class:`.Identity`.
            scale : Scaling parameter.  Default: 0.5.

        """
        y = ensure_on_device(y)
        super().__init__(y=y, A=A, scale=scale)

        #: Constant term in Poisson log likehood; equal to ln(y!)
        self.const: float = gammaln(self.y + 1)  # ln(y!)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        ε = 1e-9  # So that loss < infinity
        Ax = self.A(x)
        return self.scale * snp.sum(Ax - self.y * snp.log(Ax + ε) + self.const)
