# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Evaluate NN models implemented in objax."""

from typing import Any, Callable

from flax import linen as nn
import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import JaxArray

from ._functional import Functional

PyTree = Any

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""


class FlaxMap(Functional):
    r"""Functional whose prox applies a trained Flax model."""

    has_eval = False
    has_prox = True
    is_smooth = False

    def __init__(self, model: Callable[..., nn.Module], variables: PyTree):
        r"""Initialize a :class:`FlaxMap` object.

        Args:
            model : Flax model to apply.
            variables : parameters and batch stats of trained model.
        """
        self.model = model
        self.variables = variables
        super().__init__()

    def prox(self, x: JaxArray, lam: float = 1) -> JaxArray:
        r"""Apply trained flax model.

        Args:
            x : input.
            lam : noise estimate (not used).
        """
        if isinstance(x, BlockArray):
            raise NotImplementedError
        else:
            # add input singleton
            # scico works on (NxN) or (NxNxC) arrays
            # flax works on (KxNxNxC) arrays
            # (generally KxHxWxC arrays)
            # K: input dim
            if x.ndim == 2:
                x = x.reshape((1,) + x.shape + (1,))
            elif x.ndim == 3:
                x = x.reshape((1,) + x.shape)
            y = self.model.apply(self.variables, x, train=False, mutable=False)
            return snp.squeeze(y)
