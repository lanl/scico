# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Abel transform LinearOperator wrapping the pyabel package.

Abel transform LinearOperator wrapping the
`pyabel <https://github.com/PyAbel/PyAbel>`_ package.
"""

import math
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.numpy.fft as jnfft

import abel

from scico.linop import LinearOperator
from scico.typing import JaxArray, Shape
from scipy.linalg import solve_triangular


class AbelProjector(LinearOperator):
    r"""Abel transform projector based on `PyAbel <https://github.com/PyAbel/PyAbel>`_.

    Perform Abel transform (parallel beam tomographic projection of
    cylindrically symmetric objects) for a 2D image. The input 2D image
    is assumed to be centered and left-right symmetric.
    """

    def __init__(self, img_shape: Shape):
        """
        Args:
            img_shape: Shape of the input image.
        """
        self.proj_mat_quad = _pyabel_daun_get_proj_matrix(img_shape)

        super().__init__(
            input_shape=img_shape,
            output_shape=img_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=True,
        )

    def _eval(self, x: JaxArray) -> JaxArray:
        return _pyabel_transform(x, direction="forward", proj_mat_quad=self.proj_mat_quad).astype(
            self.output_dtype
        )

    def _adj(self, x: JaxArray) -> JaxArray:
        return _pyabel_transform(x, direction="transpose", proj_mat_quad=self.proj_mat_quad).astype(
            self.input_dtype
        )

    def inverse(self, y: JaxArray) -> JaxArray:
        """Perform inverse Abel transform.

        Args:
            y: Input image (assumed to be a result of an Abel transform)

        Returns:
            Output of inverse Abel transform
        """
        return _pyabel_transform(y, direction="inverse", proj_mat_quad=self.proj_mat_quad).astype(
            self.input_dtype
        )


def _pyabel_transform(
    x: JaxArray, direction: str, proj_mat_quad: JaxArray, symmetry_axis: Optional[list] = None
) -> JaxArray:
    """Perform Abel transformations (forward, inverse and transposed).

    This function contains code copied from `PyAbel <https://github.com/PyAbel/PyAbel>`_.
    """

    if symmetry_axis is None:
        symmetry_axis = [None]

    Q0, Q1, Q2, Q3 = get_image_quadrants(
        x, symmetry_axis=symmetry_axis, use_quadrants=(True, True, True, True)
    )

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

    return put_image_quadrants(
        (AQ0, AQ1, AQ2, AQ3), original_image_shape=x.shape, symmetry_axis=symmetry_axis
    )


def _pyabel_daun_get_proj_matrix(img_shape: Shape) -> JaxArray:
    """Get single-quadrant projection matrix."""
    proj_matrix = abel.daun.get_bs_cached(
        math.ceil(img_shape[1] / 2),
        degree=0,
        reg_type=None,
        strength=0,
        direction="forward",
        verbose=False,
    )
    return jax.device_put(proj_matrix)


# Read abel.tools.symmetry module into a string.
mod_file = abel.tools.symmetry.__file__
with open(mod_file, "r") as f:
    mod_str = f.read()

# Replace numpy functions that touch the main arrays with corresponding jax.numpy functions
mod_str = mod_str.replace("fftpack.", "jnfft.")
mod_str = mod_str.replace("np.atleast_2d", "jnp.atleast_2d")
mod_str = mod_str.replace("np.flip", "jnp.flip")
mod_str = mod_str.replace("np.concat", "jnp.concat")

# Exec the module extract defined functions from the exec scope
scope = {"jnp": jnp, "jnfft": jnfft}
exec(mod_str, scope)
get_image_quadrants = scope["get_image_quadrants"]
put_image_quadrants = scope["put_image_quadrants"]
