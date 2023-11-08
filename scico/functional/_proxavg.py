# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Proximal average."""

from typing import List, Union

from scico.numpy import Array, BlockArray

from ._functional import Functional


class ProximalAverage(Functional):
    """
    See :cite:`yu-2013-better`.

    """

    def __init__(self, func_list: List[Functional]):
        self.has_prox = all([f.has_prox for f in func_list])
        if not self.has_prox:
            raise ValueError("All functionals in func_list must have has_prox == True.")
        self.has_eval = all([f.has_eval for f in func_list])
        self.func_list = func_list

    def __repr__(self):
        return (
            Functional.__repr__(self)
            + "Components:\n"
            + "\n".join(["  " + repr(f) for f in self.func_list])
        )

    def __call__(self, x: Union[Array, BlockArray]) -> float:
        if self.has_eval:
            return sum([f(x) for f in self.func_list])
        else:
            raise ValueError("At least one functionals in func_list has has_eval == False.")

    def prox(
        self, v: Union[Array, BlockArray], lam: float = 1.0, **kwargs
    ) -> Union[Array, BlockArray]:
        """ """
        return sum([f.prox(v, lam, **kwargs) for f in self.func_list]) / len(self.func_list)
