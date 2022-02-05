# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Evaluate NN models implemented in flax."""

from typing import Any

from flax import linen as nn
from scico.blockarray import BlockArray
from scico.typing import JaxArray

from ._functional import Functional

PyTree = Any


class FlaxMap(Functional):
    r"""Functional whose prox applies a trained flax model."""

    has_eval = False
    has_prox = True

    def __init__(self, model: nn.Module, variables: PyTree):
        r"""Initialize a :class:`FlaxMap` object.

        Args:
            model: Flax model to apply.
            variables: Parameters and batch stats of trained model.
        """
        self.model = model
        self.variables = variables
        super().__init__()

    def prox(self, x: JaxArray, lam: float = 1.0, **kwargs) -> JaxArray:  # type: ignore
        r"""Apply trained flax model.

        *Warning*: The ``lam`` parameter is ignored, and has no effect on
        the output.

        Args:
            x: input.
            lam: noise estimate (ignored).

        Returns:
            Output of flax model.
        """
        if isinstance(x, BlockArray):
            raise NotImplementedError

        # Add singleton to input as necessary:
        #   scico typically works with (HxW) or (HxWxC) arrays
        #   flax expects (KxHxWxC) arrays
        #   H: spatial height  W: spatial width
        #   K: batch size  C: channel size
        x_shape = x.shape
        if x.ndim == 2:
            x = x.reshape((1,) + x.shape + (1,))
        elif x.ndim == 3:
            x = x.reshape((1,) + x.shape)
        y = self.model.apply(self.variables, x, train=False, mutable=False)
        return y.reshape(x_shape)
