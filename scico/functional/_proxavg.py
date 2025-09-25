# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Implementation of the proximal average method."""

from typing import List, Optional, Union

from scico.numpy import Array, BlockArray, isinf

from ._functional import Functional


class ProximalAverage(Functional):
    """Weighted average of functionals.

    A functional that is composed of a weighted average of functionals.
    All of the component functionals are required to have proximal
    operators. The proximal operator of the composite functional is
    approximated via the proximal average method :cite:`yu-2013-better`,
    which holds for small scaling parameters. This does not imply that it
    can only be applied to problems requiring a small regularization
    parameter since most proximal algorithms include an additional
    algorithm parameter that also plays a role in the parameter of the
    proximal operator. For example, in :class:`.PGM` and
    :class:`.AcceleratedPGM`, the scaled proximal operator parameter
    is the regularization parameter divided by the `L0` algorithm
    parameter, and for :class:`.ADMM`, the scaled proximal operator
    parameters are the regularization parameters divided by the entries
    in the `rho_list` algorithm parameter.
    """

    def __init__(
        self,
        func_list: List[Functional],
        alpha_list: Optional[List[float]] = None,
        no_inf_eval=True,
    ):
        """
        Args:
            func_list: List of component :class:`.Functional` objects,
                all of which must have a proximal operator.
            alpha_list: List of scalar weights for each
                :class:`.Functional`. If not specified, defaults to equal
                weights. If specified, the list of weights must have the
                same length as the :class:`.Functional` list. If the
                weights do not sum to unity, they are scaled to ensure
                that they do.
            no_inf_eval: If ``True``, exclude infinite values (typically
                associated with a functional that is an indicator
                function) from the evaluation of the sum of component
                functionals.
        """
        self.has_prox = all([f.has_prox for f in func_list])
        if not self.has_prox:
            raise ValueError("All functionals in 'func_list' must have has_prox == True.")
        self.has_eval = all([f.has_eval for f in func_list])
        self.no_inf_eval = no_inf_eval
        self.func_list = func_list
        N = len(func_list)
        if alpha_list is None:
            self.alpha_list = [1.0 / N] * N
        else:
            if len(alpha_list) != N:
                raise ValueError(
                    "If specified, argument 'alpha_list' must have the same length as func_list"
                )
            alpha_sum = sum(alpha_list)
            if alpha_sum != 1.0:
                alpha_list = [alpha / alpha_sum for alpha in alpha_list]
            self.alpha_list = alpha_list

    def __repr__(self):
        return (
            Functional.__repr__(self)
            + "\n  Weights: "
            + ", ".join([str(alpha) for alpha in self.alpha_list])
            + "\n  Components:\n"
            + "\n".join(["    " + repr(f) for f in self.func_list])
        )

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        """Evaluate the weighted average of component functionals."""
        if self.has_eval:
            weight_func_vals = [alpha * f(x) for (alpha, f) in zip(self.alpha_list, self.func_list)]
            if self.no_inf_eval:
                weight_func_vals = list(filter(lambda x: not isinf(x), weight_func_vals))
            return sum(weight_func_vals)
        else:
            raise ValueError(
                "At least one functional in argument 'func_list' has has_eval == False."
            )

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        r"""Approximate proximal operator of the average of functionals.

        Approximation of the proximal operator of a weighted average of
        functionals computed via the proximal average method
        :cite:`yu-2013-better`.

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lam`.
            **kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Result of evaluating the scaled proximal operator at `v`.
        """
        return sum(
            [
                alpha * f.prox(v, lam, **kwargs)
                for (alpha, f) in zip(self.alpha_list, self.func_list)
            ]
        )
