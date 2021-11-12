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
    ["Luke Pfister <luke.pfister@gmail.com>", "Thilo Balke <thilo.balke@gmail.com>"]
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
        \alpha l(\mb{y}, A(\mb{x})) \;

    where :math:`\alpha` is the scaling parameter and :math:`l(\cdot)` is the loss functional.

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""Initialize a :class:`Loss` object.

        Args:
            y : Measurement.
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
        \alpha \norm{\mb{y} - A(\mb{x})}_2^2 \;

    where :math:`\alpha` is the scaling parameter.

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""Initialize a :class:`SquaredL2Loss` object.

        Args:
            y : Measurement.
            A : Forward operator.  If None, defaults to :class:`.Identity`.
            scale : Scaling parameter.
        """
        y = ensure_on_device(y)
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
        return self.scale * (snp.abs(self.y - self.A(x)) ** 2).sum()

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
        to Hessian :math:`2 \alpha \mathrm{A^H A}`.

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
        \alpha \norm{\mb{y} - A(\mb{x})}_W^2 =
        \alpha \left(\mb{y} - A(\mb{x})\right)^T W \left(\mb{y} - A(\mb{x})\right)\;

    where :math:`\alpha` is the scaling parameter and :math:`W` is an
    instance of :class:`scico.linop.Diagonal`.  If :math:`W` is None,
    reverts to the behavior of :class:`.SquaredL2Loss`.

    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        W: Optional[linop.Diagonal] = None,
    ):

        r"""Initialize a :class:`WeightedSquaredL2Loss` object.

        Args:
            y : Measurement.
            A : Forward operator.  If None, defaults to :class:`.Identity`.
            scale : Scaling parameter.
            W:  Weighting diagonal operator. Must be non-negative.
                If None, defaults to :class:`.Identity`.
        """
        y = ensure_on_device(y)

        self.W: linop.Diagonal

        if W is None:
            self.W = linop.Identity(y.shape)
        elif isinstance(W, linop.Diagonal):
            if snp.all(W.diagonal >= 0):
                self.W = W
            else:
                raise Exception(f"The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}")

        super().__init__(y=y, A=A, scale=scale)

        if isinstance(A, operator.Operator):
            self.is_smooth = A.is_smooth
        else:
            self.is_smooth = None

        if isinstance(self.A, linop.LinearOperator):
            self.is_quadratic = True

        if isinstance(self.A, linop.Diagonal) and isinstance(self.W, linop.Diagonal):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * (self.W.diagonal * snp.abs(self.y - self.A(x)) ** 2).sum()

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        if isinstance(self.A, linop.Diagonal):
            c = 2.0 * self.scale * lam
            A = self.A.diagonal
            W = self.W.diagonal
            lhs = c * A.conj() * W * self.y + x
            ATWA = c * A.conj() * W * A
            return lhs / (ATWA + 1.0)
        else:
            raise NotImplementedError

    @property
    def hessian(self) -> linop.LinearOperator:
        r"""If ``self.A`` is a :class:`scico.linop.LinearOperator`, returns a
        :class:`scico.linop.LinearOperator` corresponding to  the Hessian
        :math:`2 \alpha \mathrm{A^H W A}`.

        Otherwise not implemented.
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
        else:
            raise NotImplementedError(
                f"Hessian is not implemented for {type(self)} when `A` is {type(A)}; must be LinearOperator"
            )


class PoissonLoss(Loss):
    r"""
    Poisson negative log likelihood loss

    .. math::
        \alpha \left( \sum_i [A(x)]_i - y_i \log\left( [A(x)]_i \right) + \log(y_i!) \right)

    where :math:`\alpha` is the scaling parameter.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
    ):
        r"""Initialize a :class:`Loss` object.

        Args:
            y : Measurement.
            A : Forward operator. Defaults to None.  If None, ``self.A`` is a :class:`.Identity`.
            scale : Scaling parameter. Default: 0.5.
        """
        y = ensure_on_device(y)
        super().__init__(y=y, A=A, scale=scale)

        #: Constant term in Poisson log likehood; equal to ln(y!)
        self.const: float = gammaln(self.y + 1)  # ln(y!)

        # The Poisson Loss is only smooth in the positive quadrant.
        self.is_smooth = None

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        Ax = self.A(x)
        return self.scale * snp.sum(Ax - self.y * snp.log(Ax) + self.const)
