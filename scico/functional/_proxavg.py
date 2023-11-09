# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal average."""

from typing import List, Optional, Union

from scico.numpy import Array, BlockArray

from ._functional import Functional


class ProximalAverage(Functional):
    """
    See :cite:`yu-2013-better`.

    """

    def __init__(self, func_list: List[Functional], alpha_list: Optional[List[float]] = None):
        self.has_prox = all([f.has_prox for f in func_list])
        if not self.has_prox:
            raise ValueError("All functionals in func_list must have has_prox == True.")
        self.has_eval = all([f.has_eval for f in func_list])
        self.func_list = func_list
        N = len(func_list)
        if alpha_list is None:
            self.alpha_list = [1.0 / N] * N
        else:
            if len(alpha_list) != N:
                raise ValueError("If specified, alpha_list must have the same length as func_list")
            alpha_sum = sum(alpha_list)
            if alpha_sum != 1.0:
                alpha_list = [alpha / alpha_sum for alpha in alpha_list]
            self.alpha_list = alpha_list

    def __repr__(self):
        return (
            Functional.__repr__(self)
            + "Components:\n"
            + "\n".join(["  " + repr(f) for f in self.func_list])
        )

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        if self.has_eval:
            return sum([alpha * f(x) for (alpha, f) in zip(self.alpha_list, self.func_list)])
        else:
            raise ValueError("At least one functional in func_list has has_eval == False.")

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        """ """
        return sum(
            [
                alpha * f.prox(v, lam, **kwargs)
                for (alpha, f) in zip(self.alpha_list, self.func_list)
            ]
        )
