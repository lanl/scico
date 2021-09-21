# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Evaluate NN models implemented in objax."""

from typing import Callable

import objax
import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.typing import JaxArray

from ._functional import Functional

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""


class ObjaxMap(Functional):
    r"""Functional whose prox applies a trained Objax model."""

    has_eval = False
    has_prox = True
    is_smooth = False

    def __init__(self, model: Callable[..., objax.Module]):
        r"""Initialize a :class:`ObjaxMap` object.

        Args:
            model : Objax model to apply.
        """
        self.model = model
        super().__init__()

    def prox(self, x: JaxArray, lam: float) -> JaxArray:
        r"""Apply trained objax model.

        Args:
            x : input.
            lam : noise estimate (not used).
        """
        if isinstance(x, BlockArray):
            raise NotImplementedError
        else:
            # add input singleton
            # scico works on (NxN) or (NxNxC) arrays
            # objax works on (KxCxNxN) arrays
            # (generally KxCxHxW arrays)
            # K: input dim
            if x.ndim == 2:
                x = x.reshape((1, 1) + x.shape)
            elif x.ndim == 3:
                # channel first
                x = snp.transpose(x, (2, 0, 1))
                x = x.reshape((1,) + x.shape)
            y = self.model(x, training=False)
            return snp.squeeze(y)
