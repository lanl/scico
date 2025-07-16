# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Cone beam Abel transform LinearOperator.

Cone beam Abel transform :class:`.LinearOperator` based on code
modified from the `axitom <https://github.com/PolymerGuy/AXITOM>`_
package.
"""

from functools import partial
from typing import Optional, Tuple

import numpy as np

from jax import Array, jit
from jax.scipy.ndimage import map_coordinates

from scico.typing import Shape

from .._linop import LinearOperator
from ._axitom import config, project


@partial(jit, static_argnames=["axis", "center"])
def _volume_by_axial_symmetry(x: Array, axis: int = 0, center: Optional[int] = None) -> Array:
    """Create a volume by axial rotation of a plane.

    Args:
        x: 2D array that is rotated about an axis to generate a volume.
        axis: Index of axis of symmetry (must be 0 or 1).
        center: If ``None``, defaults to the center of the image on the
          specified axis. Otherwise identifies the center coordinate on
          that axis.

    Returns:
        Volume as a 3D array.
    """
    N0, N1 = x.shape
    N2 = x.shape[axis]
    N0h, N1h, N2h = (N0 + 1) / 2 - 1, (N1 + 1) / 2 - 1, (N2 + 1) / 2 - 1
    half_shape = (N0h, N1h, N2h)
    if axis == 0:
        g1d = [np.arange(-N0h, N0h + 1), np.arange(0, N1), np.arange(-N2h, N2h + 1)]
    else:
        g1d = [np.arange(0, N0), np.arange(-N1h, N1h + 1), np.arange(-N2h, N2h + 1)]

    if center is not None:
        offset = center - half_shape[axis]
        g1d[axis] += offset

    g0, g1, g2 = np.meshgrid(*g1d, indexing="ij")
    grids = (g0, g1, g2)
    r = np.hypot(grids[axis], g2)
    z = np.where(grids[1 - axis] >= 0, half_shape[axis] + r, half_shape[axis] - r)
    v = map_coordinates(x, [z, grids[1 - axis]], cval=0.0, order=1)

    return v


class AxiallySymmetricVolume(LinearOperator):
    """Create a volume by axial rotation of a plane."""

    def __init__(self, input_shape, input_dtype=np.float32, axis=0, center=None):
        """
        Args:
            input_shape: Shape of the input image.
            input_dtype: Dtype of the input image.
            axis: Index of axis of symmetry (must be 0 or 1).
            center: If ``None``, defaults to the center of the image on
              the specified axis. Otherwise identifies the center
              coordinate on that axis.
        """
        self.axis = axis
        self.center = center
        output_shape = input_shape + (input_shape[axis],)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
            eval_fn=lambda x: _volume_by_axial_symmetry(x, axis=self.axis, center=self.center),
            jit=True,
        )


class AbelTransformCone(LinearOperator):
    """Cone beam Abel transform.

    Compute cone beam Abel transform of a cylindricaly symmetric volume
    represented by a 2D central slice, which is rotated about axis 1 to
    generate a 3D volume for projection. The implementation is based on
    code modified from the `axitom <https://github.com/PolymerGuy/AXITOM>`_
    package.
    """

    def __init__(
        self,
        output_shape: Shape,
        det_size: Tuple[float, float],
        obj_dist: float,
        det_dist: float,
        sym_center: float = 0,
    ):
        """
        Args:
            output_shape: Shape of the output array (projection).
            det_size: Tuple of detector size values in mm.
            obj_dist: Source-object distance in mm.
            det_dist: Source-detector distance in mm.
            sym_center: Position of the rotation axis in pixels, with 0
              corresponding to the center of the image.
        """
        self.config = config.Config(*output_shape, *det_size, det_dist, obj_dist, sym_center)
        input_shape = (self.config.detector_us.size, self.config.detector_vs.size)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=lambda x: project._forward_project(x, self.config),
            jit=True,
        )
