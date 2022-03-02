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
from scico.array import ensure_on_device, no_nan_divide
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
        \alpha f(\mb{y}, A(\mb{x})) \;,

    where :math:`\alpha` is the scaling parameter and :math:`f(\cdot)` is
    the loss functional.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        f: Optional[functional.Functional] = None,
        scale: float = 1.0,
    ):
        r"""
        Args:
            y: Measurement.
            A: Forward operator. Defaults to ``None``, in which case
               ``self.A`` is a :class:`.Identity`.
            f: Functional :math:`f`. If defined, the loss function is
               :math:`\alpha f(\mb{y} - A(\mb{x}))`. If ``None``, then
               :meth:`__call__` and :meth:`prox` (where appropriate) must
               be defined in a derived class.
            scale: Scaling parameter. Default: 1.0.

        """
        self.y = ensure_on_device(y)
        if A is None:
            # y and x must have same shape
            A = linop.Identity(self.y.shape)
        self.A = A
        self.f = f
        self.scale = scale

        # Set functional-specific flags
        self.has_eval = True
        if self.f is not None and isinstance(self.A, linop.Identity):
            self.has_prox = True
        else:
            self.has_prox = False
        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this loss at point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate loss.
        """
        if self.f is None:
            raise NotImplementedError(
                "Functional l is not defined and __call__ has" " not been overridden"
            )
        return self.scale * self.f(self.A(x) - self.y)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Scaled proximal operator of loss function.

        Evaluate scaled proximal operator of this loss function, with
        scaling :math:`\lambda` = `lam` and evaluated at point
        :math:`\mb{v}` = `v`. If :meth:`prox` is not defined in a derived
        class, and if operator :math:`A` is the identity operator, then
        the proximal operator is computed using the proximal operator of
        functional :math:`l`, via Theorem 6.11 in :cite:`beck-2017-first`.

        Args:
            v: Point at which to evaluate prox function.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes. These include ``x0``, an initial guess for the
                minimizer in the defintion of :math:`\mathrm{prox}`.
        """
        if not self.has_prox:
            raise NotImplementedError(
                f"prox is not implemented for {type(self)} when A is {type(self.A)}; "
                "must be Identity"
            )
        return self.f.prox(v - self.y, self.scale * lam, **kwargs) + self.y

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

        #   prox_{f}(v) = arg min  1/2 || v - x ||_2^2 + λ 𝛼 || A x - y ||^2_W
        #                    x
        # solution at:
        #
        #   (I + λ 2𝛼 A^T W A) x = v + λ 2𝛼 A^T W y
        #
        W = self.W
        A = self.A
        𝛼 = self.scale
        y = self.y
        if "x0" in kwargs and kwargs["x0"] is not None:
            x0 = kwargs["x0"]
        else:
            x0 = snp.zeros_like(v)
        hessian = self.hessian  # = (2𝛼 A^T W A)
        lhs = linop.Identity(v.shape) + lam * hessian
        rhs = v + 2 * lam * 𝛼 * A.adj(W(y))
        x, _ = cg(lhs, rhs, x0, **self.prox_kwargs)
        return x

    @property
    def hessian(self) -> linop.LinearOperator:
        r"""Compute the Hessian of linear operator `A`.

        If `self.A` is a :class:`scico.linop.LinearOperator`, returns a
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
            f"Hessian is not implemented for {type(self)} when A is {type(A)}; "
            "must be LinearOperator."
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
        prox_kwargs: dict = {"maxiter": 100, "tol": 1e-5},
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
                `self.A` is a :class:`.Identity`.
            scale: Scaling parameter. Default: 0.5.
        """
        y = ensure_on_device(y)
        super().__init__(y=y, A=A, scale=scale)

        #: Constant term in Poisson log likehood; equal to ln(y!)
        self.const = gammaln(self.y + 1.0)  # ln(y!)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        Ax = self.A(x)
        return self.scale * snp.sum(Ax - self.y * snp.log(Ax) + self.const)


class WeightedSquaredL2AbsLoss(Loss):
    r"""Weighted squared :math:`\ell_2` with absolute value loss.

    Weighted squared :math:`\ell_2` with absolute value loss

    .. math::
        \alpha \norm{\mb{y} - | A(\mb{x}) |\,}_W^2 =
        \alpha \left(\mb{y} - | A(\mb{x}) |\right)^T W \left(\mb{y} -
        | A(\mb{x}) |\right) \;,

    where :math:`\alpha` is the scaling parameter and :math:`W` is an
    instance of :class:`scico.linop.Diagonal`.

    Proximal operator :meth:`prox` is implemented when :math:`A` is an
    instance of :class:`scico.linop.Identity`. This is not proximal
    operator according to the strict definition since the loss function
    is non-convex (Sec. 3) :cite:`soulez-2016-proximity`.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        W: Optional[linop.Diagonal] = None,
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

        if W is None:
            self.W: Union[linop.Diagonal, linop.Identity] = linop.Identity(y.shape)
        elif isinstance(W, linop.Diagonal):
            if snp.all(W.diagonal >= 0):
                self.W = W
            else:
                raise ValueError(f"The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}.")

        super().__init__(y=y, A=A, scale=scale)

        if isinstance(self.A, linop.Identity) and snp.all(y >= 0):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * (self.W.diagonal * snp.abs(self.y - snp.abs(self.A(x))) ** 2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        if not self.has_prox:
            raise NotImplementedError(f"prox is not implemented.")

        𝛼 = lam * 2.0 * self.scale * self.W.diagonal
        y = self.y
        r = snp.abs(v)
        𝛽 = (𝛼 * y + r) / (𝛼 + 1.0)
        x = snp.where(r > 0, (𝛽 / r) * v, 𝛽)
        return x


def _cbrt(x):
    """Compute the cube root of the argument.

    The two standard options for computing the cube root of an array are
    :func:`numpy.cbrt`, or raising to the power of (1/3), i.e. `x ** (1/3)`.
    The former cannot be used for complex values, and the latter returns
    a complex root of a negative real value. This functions can be used
    for both real and complex values, and returns the real root of
    negative real values.

    Args:
        x: Input array

    Returns:
        Array of cube roots of input `x`.
    """
    s = snp.where(snp.abs(snp.angle(x)) <= 3 * snp.pi / 4, 1, -1)
    return s * (s * x) ** (1 / 3)


def _dep_cubic_root(p, q):
    r"""Compute the positive real root of a depressed cubic equation.

    A depressed cubic equation is one that can be written in the form

    .. math::
       x^3 + px + q \;.

    This function finds the positive real root of such an equation via
    `Cardano's method <https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula>`__,
    for `p` and `q` such that there is a single positive real root
    (see Sec. 3.C of :cite:`soulez-2016-proximity`).
    """
    q2 = q / 2
    Δ = q2 ** 2 + (p / 3) ** 3
    Δrt = snp.sqrt(Δ + 0j)
    u3, v3 = -q2 + Δrt, -q2 - Δrt
    u, v = _cbrt(u3), _cbrt(v3)
    r = (u + v).real
    assert snp.allclose(snp.abs(r ** 3 + p * r + q), 0, atol=1e-4)
    return r


class WeightedSquaredL2AbsSquaredLoss(Loss):
    r"""Weighted squared :math:`\ell_2` with squared absolute value loss.

    Weighted squared :math:`\ell_2` with squared absolute value loss

    .. math::
        \alpha \norm{\mb{y} - | A(\mb{x}) |^2 \,}_W^2 =
        \alpha \left(\mb{y} - | A(\mb{x}) |^2 \right)^T W \left(\mb{y} -
        | A(\mb{x}) |^2 \right) \;,

    where :math:`\alpha` is the scaling parameter and :math:`W` is an
    instance of :class:`scico.linop.Diagonal`.

    Proximal operator :meth:`prox` is implemented when :math:`A` is an
    instance of :class:`scico.linop.Identity`. This is not proximal
    operator according to the strict definition since the loss function
    is non-convex (Sec. 3) :cite:`soulez-2016-proximity`.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        W: Optional[linop.Diagonal] = None,
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

        if W is None:
            self.W: Union[linop.Diagonal, linop.Identity] = linop.Identity(y.shape)
        elif isinstance(W, linop.Diagonal):
            if snp.all(W.diagonal >= 0):
                self.W = W
            else:
                raise ValueError(f"The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}.")

        super().__init__(y=y, A=A, scale=scale)

        if isinstance(self.A, linop.Identity) and snp.all(y >= 0):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * (self.W.diagonal * snp.abs(self.y - snp.abs(self.A(x)) ** 2) ** 2).sum()

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        if not self.has_prox:
            raise NotImplementedError(f"prox is not implemented.")

        𝛼 = lam * 4.0 * self.scale * self.W.diagonal
        𝛽 = snp.abs(v)
        p = no_nan_divide(1.0 - 𝛼 * self.y, 𝛼)
        q = no_nan_divide(-𝛽, 𝛼)
        r = _dep_cubic_root(p, q)
        φ = snp.where(𝛽 > 0, v / snp.abs(v), 1.0)
        x = snp.where(𝛼 > 0, r * φ, v)
        return x
