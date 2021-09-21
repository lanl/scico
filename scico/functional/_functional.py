# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"Functional base class." ""

import warnings
from typing import List, Optional, Union

import jax

import scico
from scico import numpy as snp
from scico.blockarray import BlockArray
from scico.typing import JaxArray

__author__ = """\n""".join(
    ["Luke Pfister <pfister@lanl.gov>", "Thilo Balke <thilo.balke@gmail.com>"]
)


class Functional:
    r"""Base class for functionals.

    A functional maps an :code:`array-like` to a scalar; abstractly, a functional is
    a mapping from :math:`\mathbb{R}^n` or :math:`\mathbb{C}^n` to :math:`\mathbb{R}`.
    """

    #: True if this functional can be evaluated, False otherwise.
    #: This attribute must be overridden and set to True or False in any derived classes.
    has_eval: Optional[bool] = None

    #: True if this functional has the prox method, False otherwise.
    #: This attribute must be overridden and set to True or False in any derived classes.
    has_prox: Optional[bool] = None

    #: True if this functional is differentiable, False otherwise.
    #: Note that ``is_smooth = False`` does not preclude the use of the :func:`.grad` method.
    #: This attribute must be overridden and set to True or False in any derived classes.
    is_smooth: Optional[bool] = None

    def __init__(self):
        self._grad = scico.grad(self.__call__)

    def __repr__(self):
        return f"""{type(self)}
has_eval = {self.has_eval}
has_prox = {self.has_prox}
is_smooth = {self.is_smooth}
        """

    def __mul__(self, other):
        if snp.isscalar(other) or isinstance(other, jax.core.Tracer):
            return ScaledFunctional(self, other)
        else:
            raise NotImplementedError(
                f"Operation __mul__ not defined between {type(self)} and {type(other)}"
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        r"""Evaluate this functional at point :math:`\mb{x}`.

        Args:
           x : Point at which to evaluate this functional.

        """
        if not self.has_eval:
            raise NotImplementedError(
                f"Functional {type(self)} cannot be evaluated; has_eval={self.has_eval}"
            )

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        r"""Evaluate proximal mapping at point :math:`\mb{x}` with scaling lam

        .. math::
           \mathrm{prox}(\mb{x}, \lambda) = \argmin_{\mb{v}} \frac{1}{2}
           \norm{\mb{x} - \mb{v}}_2^2 + \lambda \ \mathrm{f}(\mb{v})

        where :math:`f(\mb{v})` represents this functional evaluated at :math:`\mb{v}`.

        Args:
            x : Point at which to evaluate prox function.
            lam : Proximal parameter :math:`\lambda`
        """
        if not self.has_prox:
            raise NotImplementedError(
                f"Functional {type(self)} does not have a prox; has_prox={self.has_prox}"
            )

    def grad(self, x: Union[JaxArray, BlockArray]):
        r"""Evaluates the gradient of this functional at point :math:`\mb{x}`.

        Args:
            x : Point at which to evaluate gradient.
        """
        if not self.is_smooth:  # could be True, False, or None
            warnings.warn("This functional isn't smooth!", stacklevel=2)

        return self._grad(x)


class ScaledFunctional(Functional):
    """A functional times a scalar.

    For a scalar :math:`c` and :class:`Functional` :math:`f`, the scaled functional
    :math:`g = cf` has proximal operator

    .. math::

       \mathrm{prox}_g(v, \lambda) = \mathrm{prox}_f(v, c \lambda)


    """

    def __repr__(self):
        return "Scaled functional of type " + str(type(self.functional))

    def __init__(self, functional: Functional, scale: float):
        self.functional = functional
        self.scale = scale
        self.is_smooth = functional.is_smooth
        self.has_eval = functional.has_eval
        self.has_prox = functional.has_prox
        super().__init__()

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return self.scale * self.functional(x)

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        return self.functional.prox(x, lam * self.scale)


class SeparableFunctional(Functional):
    r"""A functional that is separable in its arguments.

    A separable functional :math:`f : \mathbb{C}^N \to \mathbb{R}` can be written as the sum
    of functionals :math:`f_i : \mathbb{C}^{N_i} \to \mathbb{R}` with :math:`\sum_i N_i = N`.
    In particular,

    .. math::
       f(\mb{x}) = f(\mb{x}_1, \dots, \mb{x}_N) = f_1(\mb{x}_1) + \dots + f_N(\mb{x}_N)

    A :class:`SeparableFunctional` accepts a :class:`.BlockArray` and is separable
    in the block components.

    """

    def __init__(self, functional_list: List[Functional]):
        r"""
        Args:
            functional_list:  List of component functionals f_i.  This functional
                takes as an input a :class:`.BlockArray` with
                ``num_blocks == len(functional_list)``.
        """
        self.functional_list: List[Functional] = functional_list

        self.has_eval: bool = all(fi.has_eval for fi in functional_list)
        self.has_prox: bool = all(fi.has_prox for fi in functional_list)
        self.is_smooth: bool = all(fi.is_smooth for fi in functional_list)

        super().__init__()

    def __call__(self, x: BlockArray) -> float:
        if len(x.shape) == len(self.functional_list):
            return snp.sum(snp.array([fi(xi) for fi, xi in zip(self.functional_list, x)]))
        else:
            raise ValueError(
                f"Number of blocks in x, {len(x.shape)}, and length of functional_list, {len(self.functional_list)}, do not match"
            )

    def prox(self, x: BlockArray, lam: float) -> BlockArray:
        r"""Evaluate proximal operator of the separable functional (see Theorem 6.6 of :cite:`beck-2017-first`).

          .. math::
             \mathrm{prox}_f(\mb{x}, \lambda)
             =
             \begin{bmatrix}
               \mathrm{prox}_{f_1}(\mb{x}_1, \lambda) \\
               \vdots \\
               \mathrm{prox}_{f_N}(\mb{x}_N, \lambda) \\
             \end{bmatrix}

        Args:
            x :  Input array :math:`\mb{x}`
            lam : Proximal parameter :math:`\lambda`
        """
        if len(x.shape) == len(self.functional_list):
            return BlockArray.array([fi.prox(xi, lam) for fi, xi in zip(self.functional_list, x)])
        else:
            raise ValueError(
                f"Number of blocks in x, {len(x.shape)}, and length of functional_list, {len(self.functional_list)}, do not match"
            )


class ZeroFunctional(Functional):
    r"""Zero functional, :math:`f(\mb{x}) = 0 \in \mbb{R}` for any input."""

    has_eval = True
    has_prox = True
    is_smooth = True

    def __call__(self, x: Union[JaxArray, BlockArray]) -> float:
        return 0.0

    def prox(self, x: Union[JaxArray, BlockArray], lam: float) -> Union[JaxArray, BlockArray]:
        return x
