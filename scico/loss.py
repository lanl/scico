# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.


"""Loss function classes."""

import warnings
from copy import copy
from functools import wraps
from typing import Callable, Optional, Union

import jax

import scico.numpy as snp
from scico import functional, linop, operator
from scico.numpy import BlockArray
from scico.numpy.util import ensure_on_device, no_nan_divide
from scico.scipy.special import gammaln  # type: ignore
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
               `self.A` is a :class:`.Identity` with input shape
               and dtype determined by the shape and dtype of `y`.
            f: Functional :math:`f`. If defined, the loss function is
               :math:`\alpha f(\mb{y} - A(\mb{x}))`. If ``None``, then
               :meth:`__call__` and :meth:`prox` (where appropriate) must
               be defined in a derived class.
            scale: Scaling parameter. Default: 1.0.
        """
        self.y = ensure_on_device(y)
        if A is None:
            # y and x must have same shape
            A = linop.Identity(input_shape=self.y.shape, input_dtype=self.y.dtype)  # type: ignore
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
                "Functional f is not defined and __call__ has not been overridden"
            )
        return self.scale * self.f(self.A(x) - self.y)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1, **kwargs
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
               classes. These include `x0`, an initial guess for the
               minimizer in the defintion of :math:`\mathrm{prox}`.
        """
        if not self.has_prox:
            raise NotImplementedError(
                f"prox is not implemented for {type(self)} when A is {type(self.A)}; "
                "must be Identity"
            )
        assert self.f is not None
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


class SquaredL2Loss(Loss):
    r"""Weighted squared :math:`\ell_2` loss.

    Weighted squared :math:`\ell_2` loss

    .. math::
        \alpha \norm{\mb{y} - A(\mb{x})}_W^2 =
        \alpha \left(\mb{y} - A(\mb{x})\right)^T W \left(\mb{y} -
        A(\mb{x})\right) \;,

    where :math:`\alpha` is the scaling parameter and :math:`W` is an
    instance of :class:`scico.linop.Diagonal`. If :math:`W` is ``None``,
    the weighting is an identity operator, giving an unweighted squared
    :math:`\ell_2` loss.
    """

    def __init__(
        self,
        y: Union[JaxArray, BlockArray],
        A: Optional[Union[Callable, operator.Operator]] = None,
        scale: float = 0.5,
        W: Optional[linop.Diagonal] = None,
        prox_kwargs: Optional[dict] = None,
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
            self.W = linop.Identity(y.shape)  # type: ignore
        elif isinstance(W, linop.Diagonal):
            if snp.all(W.diagonal >= 0):  # type: ignore
                self.W = W
            else:
                raise ValueError(f"The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}")

        super().__init__(y=y, A=A, scale=scale)

        default_prox_kwargs = {"maxiter": 100, "tol": 1e-5}
        if prox_kwargs:
            default_prox_kwargs.update(prox_kwargs)
        self.prox_kwargs = default_prox_kwargs

        if isinstance(self.A, linop.LinearOperator):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * snp.sum(self.W.diagonal * snp.abs(self.y - self.A(x)) ** 2)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
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
            lhs = c * A.conj() * W * self.y + v  # type: ignore
            ATWA = c * A.conj() * W * A  # type: ignore
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
        x, _ = cg(lhs, rhs, x0, **self.prox_kwargs)  # type: ignore
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
                eval_fn=lambda x: 2 * self.scale * A.adj(W(A(x))),  # type: ignore
                adj_fn=lambda x: 2 * self.scale * A.adj(W(A(x))),  # type: ignore
                input_dtype=A.input_dtype,
            )

        raise NotImplementedError(
            f"Hessian is not implemented for {type(self)} when A is {type(A)}; "
            "must be LinearOperator."
        )


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

        #: Constant term, :math:`\ln(y!)`, in Poisson log likehood.
        self.const = gammaln(self.y + 1.0)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        Ax = self.A(x)
        return self.scale * snp.sum(Ax - self.y * snp.log(Ax) + self.const)


class SquaredL2AbsLoss(Loss):
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
                raise ValueError("The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}.")

        super().__init__(y=y, A=A, scale=scale)

        if isinstance(self.A, linop.Identity) and snp.all(y >= 0):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * snp.sum(self.W.diagonal * snp.abs(self.y - snp.abs(self.A(x))) ** 2)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        if not self.has_prox:
            raise NotImplementedError(f"prox is not implemented.")

        𝛼 = lam * 2.0 * self.scale * self.W.diagonal
        y = self.y
        r = snp.abs(v)
        𝛽 = (𝛼 * y + r) / (𝛼 + 1.0)
        x = snp.where(r > 0, (𝛽 / r) * v, 𝛽)
        return x


def _cbrt(x: Union[JaxArray, BlockArray]) -> Union[JaxArray, BlockArray]:
    """Compute the cube root of the argument.

    The two standard options for computing the cube root of an array are
    :func:`numpy.cbrt`, or raising to the power of (1/3), i.e. `x ** (1/3)`.
    The former cannot be used for complex values, and the latter returns
    a complex root of a negative real value. This functions can be used
    for both real and complex values, and returns the real root of
    negative real values.

    Args:
        x: Input array.

    Returns:
        Array of cube roots of input `x`.
    """
    s = snp.where(snp.abs(snp.angle(x)) <= 2 * snp.pi / 3, 1, -1)
    return s * (s * x) ** (1 / 3)


def _check_root(
    x: Union[JaxArray, BlockArray],
    p: Union[JaxArray, BlockArray],
    q: Union[JaxArray, BlockArray],
    tol: float = 1e-4,
):
    """Check the precision of a cubic equation solution.

    Check the precision of an array of depressed cubic equation solutions,
    issuing a warning if any of the errors exceed a specified tolerance.

    Args:
        x: Array of roots of a depressed cubic equation.
        p: Array of linear parameters of a depressed cubic equation.
        q: Array of constant parameters of a depressed cubic equation.
        tol: Expected tolerance for solution precision.
    """
    err = snp.abs(x**3 + p * x + q)
    if not snp.allclose(err, 0, atol=tol):
        idx = snp.argmax(err)
        msg = (
            "Low precision in root calculation. Worst error is "
            f"{err.ravel()[idx]:.3e} for p={p.ravel()[idx]} and q={q.ravel()[idx]}"
        )
        warnings.warn(msg)


def _dep_cubic_root(
    p: Union[JaxArray, BlockArray], q: Union[JaxArray, BlockArray]
) -> Union[JaxArray, BlockArray]:
    r"""Compute a real root of a depressed cubic equation.

    A depressed cubic equation is one that can be written in the form

    .. math::
       x^3 + px + q = 0 \;.

    The determinant is

    .. math::
       \Delta = (q/2)^2 + (p/3)^3 \;.

    When :math:`\Delta > 0` this equation has one real root and two
    complex (conjugate) roots, when :math:`\Delta = 0`, all three roots
    are real, with at least two being equal, and when :math:`\Delta < 0`,
    all roots are real and unequal.

    According to Vieta's formulas, the roots :math:`x_0, x_1`, and
    :math:`x_2` of this equation satisfy

    .. math::
       x_0 + x_1 + x_2 &= 0 \\
       x_0 x_1 + x_0 x_2 + x_2 x_3 &= p \\
       x_0 x_1 x_2 &= -q \;.

    Therefore, when :math:`q` is negative, the equation has a single real
    positive root since at least one root must be negative for their sum
    to be zero, and their product could not be positive if only one root
    were zero. This function always returns a real root; when :math:`q`
    is negative, it returns the single positive root.

    The solution is computed using
    `Vieta's substitution <https://mathworld.wolfram.com/CubicFormula.html>`__,

    .. math::
       w = x - \frac{p}{3w} \;,

    which reduces the depressed cubic equation to

    .. math::
       w^3 - \frac{p^3}{27w^3} + q = 0\;,

    which can be expressed as a quadratic equation in :math:`w^3` by
    multiplication by :math:`w^3`, leading to

    .. math::
       w^3 = -\frac{q}{2} \pm \sqrt{\frac{q^2}{4} + \frac{p^3}{27}} \;.

    Note that the multiplication by :math:`w^3` introduces a spurious
    solution at zero in the case :math:`p = 0`, which must be handled
    separately as

    .. math::
       w^3 = -q \;.

    Despite taking this into account, very poor numerical precision can
    be obtained when :math:`p` is small but non-zero since, in this case

    .. math::
       \sqrt{\Delta} = \sqrt{(q/2)^2 + (p/3)^3} \approx q/2 \;,

    so that an incorrect solutions :math:`w^3 = 0` or :math:`w^3 = -q`
    are obtained, depending on the choice of sign in the equation for
    :math:`w^3`.

    An alternative derivation leads to the equation

    .. math::
       x = \sqrt[3]{-q/2 + \sqrt{\Delta}} + \sqrt[3]{-q/2 - \sqrt{\Delta}}

    for the real root, but this is also prone to severe numerical errors
    in single precision arithmetic.

    Args:
       p: Array of :math:`p` values.
       q: Array of :math:`q` values.

    Returns:
       Array of real roots of the cubic equation.
    """
    Δ = (q**2) / 4.0 + (p**3) / 27.0
    w3 = snp.where(snp.abs(p) <= 1e-7, -q, -q / 2.0 + snp.sqrt(Δ + 0j))
    w = _cbrt(w3)
    r = (w - no_nan_divide(p, 3 * w)).real
    _check_root(r, p, q)
    return r


class SquaredL2SquaredAbsLoss(Loss):
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
                raise ValueError("The weights, W.diagonal, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}.")

        super().__init__(y=y, A=A, scale=scale)

        if isinstance(self.A, linop.Identity) and snp.all(y >= 0):
            self.has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * snp.sum(
            self.W.diagonal * snp.abs(self.y - snp.abs(self.A(x)) ** 2) ** 2
        )

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
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
