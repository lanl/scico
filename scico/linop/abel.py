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

import math
from typing import Optional

import numpy as np

import jax
import jax.experimental.host_callback

import abel

from scico.linop import LinearOperator
from scico.typing import Array, JaxArray
from scipy.linalg import solve_triangular


class AbelProjector(LinearOperator):
    def __init__(self, img_shape, center=None):

        self._eval = jax.custom_vjp(self._proj_hcb)
        self._eval.defvjp(lambda x: (self._proj_hcb(x), None), lambda _, y: (self._bproj_hcb(y),))

        self._adj = jax.custom_vjp(self._bproj_hcb)
        self._adj.defvjp(lambda y: (self._bproj_hcb(y), None), lambda _, x: (self._proj_hcb(x),))

        if center is None:
            self.center = center
        else:
            raise ValueError("Not implemented yet")

        self.proj_mat_quad = pyabel_daun_get_proj_matrix(img_shape)

        super().__init__(
            input_shape=img_shape,
            output_shape=img_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    @staticmethod
    def _proj(x: JaxArray, proj_mat_quad: Array, center: Optional[int] = None) -> JaxArray:
        return pyabel_transform(np.array(x), direction="forward", proj_mat_quad=proj_mat_quad)

    def _proj_hcb(self, x):
        # host callback wrapper for _proj
        y = jax.experimental.host_callback.call(
            lambda x: self._proj(x, self.proj_mat_quad),
            x,
            result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype),
        )
        return y

    @staticmethod
    def _bproj(y: JaxArray, proj_mat_quad: Array, center: Optional[int] = None) -> JaxArray:
        return pyabel_transform(np.array(y), direction="transpose", proj_mat_quad=proj_mat_quad)

    def _bproj_hcb(self, y):
        # host callback wrapper for _bproj
        x = jax.experimental.host_callback.call(
            lambda y: self._bproj(y, self.proj_mat_quad),
            y,
            result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype),
        )
        return x

    def inverse(self, y):
        return pyabel_transform(np.array(y), direction="inverse", proj_mat_quad=self.proj_mat_quad)


def pyabel_transform(x, direction, proj_mat_quad, symmetry_axis=[None]):

    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(x, symmetry_axis=symmetry_axis)

    def transform_quad(data):
        if direction == "forward":
            return data.dot(proj_mat_quad)
        elif direction == "transpose":
            return data.dot(proj_mat_quad.T)
        elif direction == "inverse":
            return solve_triangular(proj_mat_quad.T, data.T).T
        else:
            ValueError("Unsupported direction")

    AQ0 = AQ1 = AQ2 = AQ3 = None
    AQ1 = transform_quad(Q1)

    if 1 not in symmetry_axis:
        AQ2 = transform_quad(Q2)

    if 0 not in symmetry_axis:
        AQ0 = transform_quad(Q0)

    if None in symmetry_axis:
        AQ3 = transform_quad(Q3)

    return abel.tools.symmetry.put_image_quadrants(
        (AQ0, AQ1, AQ2, AQ3), original_image_shape=x.shape, symmetry_axis=symmetry_axis
    )


def pyabel_daun_get_proj_matrix(img_shape):

    return abel.daun.get_bs_cached(
        math.ceil(img_shape[1] / 2),
        degree=0,
        reg_type=None,
        strength=0,
        direction="forward",
        verbose=False,
    )
