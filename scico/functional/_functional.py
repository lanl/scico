# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functional base class."""

from typing import List, Optional, Union

import jax

import scico
from scico import numpy as snp
from scico.blockarray import BlockArray
from scico.typing import JaxArray


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
        return f"""{type(self)}
has_eval = {self.has_eval}
has_prox = {self.has_prox}
        """

    def __mul__(self, other):
        if snp.isscalar(other) or isinstance(other, jax.core.Tracer):
            return ScaledFunctional(self, other)
        raise NotImplementedError(
            f"Operation __mul__ not defined between {type(self)} and {type(other)}"
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this functional at point :math:`\mb{x}`.

        Args:
           x: Point at which to evaluate this functional.

        """
        if not self.has_eval:
            raise NotImplementedError(
                f"Functional {type(self)} cannot be evaluated; has_eval={self.has_eval}"
            )

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Scaled proximal operator of functional.

        Evaluate scaled proximal operator of this functional, with
        scaling :math:`\lambda` = `lam` and evaluated at point
        :math:`\mb{v}` = `v`. The scaled proximal operator is defined as

        .. math::
           \mathrm{prox}_{\lambda f}(\mb{v}) = \argmin_{\mb{x}}
           \lambda f(\mb{x}) +
           \frac{1}{2} \norm{\mb{v} - \mb{x}}_2^2\;,

        where :math:`\lambda f(\mb{x})` represents this functional evaluated at
        :math:`\mb{x}` multiplied by :math:`\lambda`.

        Args:
            v: Point at which to evaluate prox function.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes. These include ``x0``, an initial guess for the
                minimizer in the defintion of :math:`\mathrm{prox}`.

        """
        if not self.has_prox:
            raise NotImplementedError(
                f"Functional {type(self)} does not have a prox; has_prox={self.has_prox}"
            )

    def conj_prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Scaled proximal operator of convex conjugate of functional.

        Evaluate scaled proximal operator of convex conjugate (Fenchel
        conjugate) of this functional, with scaling
        :math:`\lambda` = `lam`, and evaluated at point
        :math:`\mb{v}` = `v`. Denoting this functional by :math:`f` and
        its convex conjugate by :math:`f^*`, the proximal operator of
        :math:`f^*` is computed as follows by exploiting the extended
        Moreau decomposition (see Sec. 6.6 of :cite:`beck-2017-first`)

        .. math::
           \mathrm{prox}_{\lambda f^*}(\mb{v}) = \mb{v} - \lambda
           \mathrm{prox}_{\lambda^{-1} f}(\mb{v / \lambda}) \;.

        Args:
            v: Point at which to evaluate prox function.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional keyword args, passed directly to
               ``self.prox``.
        """
        return v - lam * self.prox(v / lam, 1.0 / lam, **kwargs)

    def grad(self, x: Union[JaxArray, BlockArray]):
        r"""Evaluates the gradient of this functional at :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate gradient.
        """
        return self._grad(x)


class ScaledFunctional(Functional):
    r"""A functional multiplied by a scalar."""

    def __repr__(self):
        return "Scaled functional of type " + str(type(self.functional))

    def __init__(self, functional: Functional, scale: float):
        self.functional = functional
        self.scale = scale
        self.has_eval = functional.has_eval
        self.has_prox = functional.has_prox
        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * self.functional(x)

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        r"""Evaulate the scaled proximal operator of the scaled functional.

        Note that, by definition, the scaled proximal operator of a
        functional is the proximal operator of the scaled functional. The
        scaled proximal operator of a scaled functional is the scaled
        proximal operator of the unscaled functional with the proximal
        operator scaling consisting of the product of the two scaling
        factors, i.e., for functional :math:`f` and scaling factors
        :math:`\alpha` and :math:`\beta`, the proximal operator with scaling
        parameter :math:`\alpha` of scaled functional :math:`\beta f` is
        the proximal operator with scaling parameter :math:`\alpha \beta` of
        functional :math:`f`,

        .. math::
           \mathrm{prox}_{\alpha (\beta f)}(\mb{v}) =
           \mathrm{prox}_{(\alpha \beta) f}(\mb{v}) \;.

        """
        return self.functional.prox(v, lam * self.scale)


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
            functional_list:  List of component functionals f_i. This functional
                takes as an input a :class:`.BlockArray` with
                ``num_blocks == len(functional_list)``.
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
            f"{len(self.functional_list)}, do not match"
        )

    def prox(self, v: BlockArray, lam: float = 1.0, **kwargs) -> BlockArray:
        r"""Evaluate proximal operator of the separable functional.

        Evaluate proximal operator of the separable functional (see
        Theorem 6.6 of :cite:`beck-2017-first`).

          .. math::
             \mathrm{prox}_{\lambda f}(\mb{v})
             =
             \begin{bmatrix}
               \mathrm{prox}_{\lambda f_1}(\mb{v}_1) \\
               \vdots \\
               \mathrm{prox}_{\lambda f_N}(\mb{v}_N) \\
             \end{bmatrix} \;.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda`.
            kwargs: Additional arguments that may be used by derived
                classes.

        """
        if len(v.shape) == len(self.functional_list):
            return BlockArray.array([fi.prox(vi, lam) for fi, vi in zip(self.functional_list, v)])
        raise ValueError(
            f"Number of blocks in v, {len(v.shape)}, and length of functional_list, "
            f"{len(self.functional_list)}, do not match"
        )


class ZeroFunctional(Functional):
    r"""Zero functional, :math:`f(\mb{x}) = 0 \in \mbb{R}` for any input."""

    has_eval = True
    has_prox = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return 0.0

    def prox(
        self, v: Union[JaxArray, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[JaxArray, BlockArray]:
        return v
