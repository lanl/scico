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

import jax.numpy as jnp
from jax import Array, jit, vjp
from jax.scipy.ndimage import map_coordinates

from scico.typing import DType, Shape

from .._linop import LinearOperator
from ._axitom import config, project


@partial(jit, static_argnames=["axis", "center"])
def _volume_by_axial_symmetry(
    x: Array, axis: int = 0, center: Optional[int] = None, zrange: Optional[Array] = None
) -> Array:
    """Create a volume by axial rotation of a plane.

    Args:
        x: 2D array that is rotated about an axis to generate a volume.
        axis: Index of axis of symmetry (must be 0 or 1).
        center: Location of the axis of symmetry on the other axis. If
          ``None``, defaults to center of that axis. Otherwise identifies
          the center coordinate on that axis.
        zrange: 1D array of points at which the extended axis is
          constructed. Defaults to the same as for axis :code:`1 - axis`.

    Returns:
        Volume as a 3D array.
    """
    N0, N1 = x.shape
    N0h, N1h = (N0 + 1) / 2 - 1, (N1 + 1) / 2 - 1
    half_shape = (N0h, N1h)
    if zrange is None:
        N2 = x.shape[1 - axis]
        N2h = (N2 + 1) / 2 - 1
        zrange = jnp.arange(-N2h, N2h + 1)
    if axis == 0:
        g1d = [np.arange(0, N0), jnp.arange(-N1h, N1h + 1), zrange]
    else:
        g1d = [np.arange(-N0h, N0h + 1), jnp.arange(0, N1), zrange]

    if center is None:
        offset = 0
    else:
        offset = center - half_shape[1 - axis]

    g0, g1, g2 = jnp.meshgrid(*g1d, indexing="ij")
    grids = (g0, g1, g2)
    r = jnp.hypot(grids[1 - axis], g2)
    sym_ax_crd = jnp.where(
        grids[1 - axis] >= 0, half_shape[1 - axis] + offset + r, half_shape[1 - axis] + offset - r
    )
    if axis == 0:
        coords = [grids[axis], sym_ax_crd]
    else:
        coords = [sym_ax_crd, grids[axis]]
    v = map_coordinates(x, coords, cval=0.0, order=1)

    return v


class AxiallySymmetricVolume(LinearOperator):
    """Create a volume by axial rotation of a plane."""

    def __init__(
        self,
        input_shape: Shape,
        input_dtype: DType = np.float32,
        axis: int = 0,
        center: Optional[int] = None,
    ):
        """
        Args:
            input_shape: Input image shape.
            input_dtype: Input image dtype.
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

    Compute cone beam Abel transform of a cylindrically symmetric volume
    represented by a 2D central slice, which is rotated about axis 1 to
    generate a 3D volume for projection. The implementation is based on
    code modified from the `axitom <https://github.com/PolymerGuy/AXITOM>`_
    package.
    """

    def __init__(
        self,
        input_shape: Shape,
        det_size: Tuple[float, float],
        obj_dist: float,
        det_dist: float,
        num_blocks: int = 1,
    ):
        """
        Args:
            input_shape: Shape of the input array. If 2D, it is extended
              to 3D (onto axis 1) by cylindrical symmetry.
            det_size: Tuple of detector size values in mm.
            obj_dist: Source-object distance in mm.
            det_dist: Source-detector distance in mm.
            num_blocks: Number of blocks into which the volume should be
              divided (for serial processing, to limit memory usage) in
              the imaging direction.
        """
        if len(input_shape) == 2:
            self.input_2d = True
            output_shape = input_shape
        else:
            self.input_2d = False
            output_shape = (input_shape[0], input_shape[2])
        self.config = config.Config(*output_shape, *det_size, det_dist, obj_dist)
        self.num_blocks = num_blocks
        eval_fn = lambda x: project.forward_project(
            x, self.config, num_blocks=self.num_blocks, input_2d=self.input_2d
        )
        # use vjp rather than linear_transpose due to jax-ml/jax#30552
        adj_fn = vjp(eval_fn, jnp.zeros(output_shape))[1]
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=eval_fn,
            adj_fn=lambda x: adj_fn(x)[0],
            jit=True,
        )
