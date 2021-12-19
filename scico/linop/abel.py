# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Abel transform LinearOperator wrapping the pyabel package.

Abel transform LinearOperator wrapping the
forked `pyabel <https://github.com/smajee/PyAbel>`_ package.
"""


import numpy as np

import jax
import jax.experimental.host_callback

import abel

from scico.linop import LinearOperator
from scico.typing import JaxArray


class AbelProjector(LinearOperator):
    def __init__(self, img_shape):

        self._eval = jax.custom_vjp(self._proj_hcb)
        self._eval.defvjp(lambda x: (self._proj_hcb(x), None), lambda _, y: (self._bproj_hcb(y),))

        self._adj = jax.custom_vjp(self._bproj_hcb)
        self._adj.defvjp(lambda y: (self._bproj_hcb(y), None), lambda _, x: (self._proj_hcb(x),))

        super().__init__(
            input_shape=img_shape,
            output_shape=img_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    @staticmethod
    def _proj(x: JaxArray) -> JaxArray:
        return abel.Transform(np.array(x), direction="forward", method="daun").transform

    def _proj_hcb(self, x):
        # host callback wrapper for _proj
        y = jax.experimental.host_callback.call(
            lambda x: self._proj(x),
            x,
            result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype),
        )
        return y

    @staticmethod
    def _bproj(y: JaxArray) -> JaxArray:
        return abel.Transform(
            np.array(y), direction="forward", method="daun", transform_options=dict(transposed=True)
        ).transform

    def _bproj_hcb(self, y):
        # host callback wrapper for _bproj
        x = jax.experimental.host_callback.call(
            lambda y: self._bproj(y),
            y,
            result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype),
        )
        return x
