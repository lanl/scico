# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functional base class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Union

import scico
from scico import numpy as snp
from scico.numpy import Array, BlockArray


class Functional:
    r"""Base class for functionals.

    A functional maps an :code:`array-like` to a scalar; abstractly, a
    functional is a mapping from :math:`\mathbb{R}^n` or
    :math:`\mathbb{C}^n` to :math:`\mathbb{R}`.
    """

    #: True if this functional can be evaluated, False otherwise.
    #: This attribute must be overridden and set to True or False in any derived classes.
    has_eval: Optional[bool] = None

    #: True if this functional has the prox method, False otherwise.
    #: This attribute must be overridden and set to True or False in any derived classes.
    has_prox: Optional[bool] = None

    def __init__(self):
        self._grad = scico.grad(self.__call__)

    def __repr__(self):
        return f"""{type(self)} (has_eval = {self.has_eval}, has_prox = {self.has_prox})"""

    def __mul__(self, other: Union[float, int]) -> ScaledFunctional:
        if snp.util.is_scalar_equiv(other):
            return ScaledFunctional(self, other)
        return NotImplemented

    def __rmul__(self, other: Union[float, int]) -> ScaledFunctional:
        return self.__mul__(other)

    def __add__(self, other: Functional) -> FunctionalSum:
        if isinstance(other, Functional):
            return FunctionalSum(self, other)
        return NotImplemented

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        r"""Evaluate this functional at point :math:`\mb{x}`.

        Args:
           x: Point at which to evaluate this functional.

        """
        # Functionals that can be evaluated should override this method.
        raise NotImplementedError(f"Functional {type(self)} cannot be evaluated.")

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""Scaled proximal operator of functional.

        Evaluate scaled proximal operator of this functional, with
        scaling :math:`\lambda` = `lam` and evaluated at point
        :math:`\mb{v}` = `v`. The scaled proximal operator is defined as

        .. math::
           \prox_{\lambda f}(\mb{v}) = \argmin_{\mb{x}}
           \lambda f(\mb{x}) +
           \frac{1}{2} \norm{\mb{v} - \mb{x}}_2^2\;,

        where :math:`\lambda f(\mb{x})` represents this functional evaluated at
        :math:`\mb{x}` multiplied by :math:`\lambda`.

        Args:
            v: Point at which to evaluate prox function.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes. These include `x0`, an initial guess for the
                minimizer in the definition of :math:`\prox`.
        """
        # Functionals that have a prox should override this method.
        raise NotImplementedError(f"Functional {type(self)} does not have a prox.")

    def conj_prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""Scaled proximal operator of convex conjugate of functional.

        Evaluate scaled proximal operator of convex conjugate (Fenchel
        conjugate) of this functional, with scaling
        :math:`\lambda` = `lam`, and evaluated at point
        :math:`\mb{v}` = `v`. Denoting this functional by :math:`f` and
        its convex conjugate by :math:`f^*`, the proximal operator of
        :math:`f^*` is computed as follows by exploiting the extended
        Moreau decomposition (see Sec. 6.6 of :cite:`beck-2017-first`)

        .. math::
           \prox_{\lambda f^*}(\mb{v}) = \mb{v} - \lambda \,
           \prox_{\lambda^{-1} f}(\mb{v / \lambda}) \;.

        Args:
            v: Point at which to evaluate prox function.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional keyword args, passed directly to
               `self.prox`.
        """
        return v - lam * self.prox(v / lam, 1.0 / lam, **kwargs)

    def grad(self, x: Union[Array, BlockArray]):
        r"""Evaluates the gradient of this functional at :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate gradient.
        """
        return self._grad(x)


class ScaledFunctional(Functional):
    r"""A functional multiplied by a scalar."""

    def __init__(self, functional: Functional, scale: float):
        self.functional = functional
        self.scale = scale
        self.has_eval = functional.has_eval
        self.has_prox = functional.has_prox
        super().__init__()

    def __repr__(self):
        return (
            "Scaled functional of type " + str(type(self.functional)) + f" (scale = {self.scale})"
        )

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        return self.scale * self.functional(x)

    def __mul__(self, other: Union[float, int]) -> ScaledFunctional:
        if snp.util.is_scalar_equiv(other):
            return ScaledFunctional(self.functional, other * self.scale)
        return NotImplemented

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""Evaluate the scaled proximal operator of the scaled functional.

        Note that, by definition, the scaled proximal operator of a
        functional is the proximal operator of the scaled functional. The
        scaled proximal operator of a scaled functional is the scaled
        proximal operator of the unscaled functional with the proximal
        operator scaling consisting of the product of the two scaling
        factors, i.e., for functional :math:`f` and scaling factors
        :math:`\alpha` and :math:`\beta`, the proximal operator with
        scaling parameter :math:`\alpha` of scaled functional
        :math:`\beta f` is the proximal operator with scaling parameter
        :math:`\alpha \beta` of functional :math:`f`,

        .. math::
           \prox_{\alpha (\beta f)}(\mb{v}) =
           \prox_{(\alpha \beta) f}(\mb{v}) \;.

        """
        return self.functional.prox(v, lam * self.scale, **kwargs)


class SeparableFunctional(Functional):
    r"""A functional that is separable in its arguments.

    A separable functional :math:`f : \mathbb{C}^N \to \mathbb{R}` can
    be written as the sum of functionals :math:`f_i : \mathbb{C}^{N_i}
    \to \mathbb{R}` with :math:`\sum_i N_i = N`. In particular,

    .. math::
       f(\mb{x}) = f(\mb{x}_1, \dots, \mb{x}_N) = f_1(\mb{x}_1) + \dots
       + f_N(\mb{x}_N) \;.

    A :class:`SeparableFunctional` accepts a :class:`.BlockArray` and is
    separable in the block components.
    """

    def __init__(self, functional_list: List[Functional]):
        r"""
        Args:
            functional_list: List of component functionals f_i. This
               functional takes as an input a :class:`.BlockArray` with
               `num_blocks == len(functional_list)`.
        """
        self.functional_list: List[Functional] = functional_list

        self.has_eval: bool = all(fi.has_eval for fi in functional_list)
        self.has_prox: bool = all(fi.has_prox for fi in functional_list)

        super().__init__()

    def __call__(self, x: BlockArray) -> float:
        if len(x.shape) == len(self.functional_list):
            return snp.sum(snp.array([fi(xi) for fi, xi in zip(self.functional_list, x)]))
        raise ValueError(
            f"Number of blocks in x, {len(x.shape)}, and length of functional_list, "
            f"{len(self.functional_list)}, do not match."
        )

    def prox(self, v: BlockArray, lam: float = 1.0, **kwargs) -> BlockArray:
        r"""Evaluate proximal operator of the separable functional.

        Evaluate proximal operator of the separable functional (see
        Theorem 6.6 of :cite:`beck-2017-first`).

          .. math::
             \prox_{\lambda f}(\mb{v})
             =
             \begin{bmatrix}
               \prox_{\lambda f_1}(\mb{v}_1) \\ \vdots \\
               \prox_{\lambda f_N}(\mb{v}_N) \\
             \end{bmatrix} \;.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.

        """
        if len(v.shape) == len(self.functional_list):
            return snp.blockarray(
                [fi.prox(vi, lam, **kwargs) for fi, vi in zip(self.functional_list, v)]
            )
        raise ValueError(
            f"Number of blocks in v, {len(v.shape)}, and length of functional_list, "
            f"{len(self.functional_list)}, do not match."
        )


class FunctionalSum(Functional):
    r"""A sum of two functionals."""

    def __init__(self, functional1: Functional, functional2: Functional):
        self.functional1 = functional1
        self.functional2 = functional2
        self.has_eval = functional1.has_eval and functional2.has_eval
        self.has_prox = False
        super().__init__()

    def __repr__(self):
        return (
            "Sum of functionals of types "
            + str(type(self.functional1))
            + " and "
            + str(type(self.functional2))
        )

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        return self.functional1(x) + self.functional2(x)


class ZeroFunctional(Functional):
    r"""Zero functional, :math:`f(\mb{x}) = 0 \in \mbb{R}` for any input."""

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        return 0.0

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        return v
