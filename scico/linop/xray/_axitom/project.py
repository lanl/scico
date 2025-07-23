"""
Forward projection routines.

This module contains the functions used to forward project a volume onto
a sensor plane.
"""

from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.ndimage import map_coordinates

from .config import Config


@partial(jit, static_argnames=["config", "input_2d"])
def _forward_project(volume: Array, config: Config, input_2d: bool = False) -> Array:
    """Projection of a volume onto a sensor plane.

    Projection of a cylindrically symmetric volume onto a sensor plane
    using conical beam geometry.

    Args:
      volume: The volume that will be projected onto the sensor.
      config: The settings object.
      input_2d: If ``True``, the input is a 2D image from which a 3D
        volume is constructed by rotation about the center of axis 1
        of the image.

    Returns:
        The projection.

    """
    # Integrate along the rays
    uu, vv = jnp.meshgrid(config.detector_us, config.detector_vs)

    ratios = (config.object_ys + config.source_to_object_dist) / config.source_to_detector_dist

    pvs = (
        vv[:, jnp.newaxis, :] * ratios[jnp.newaxis, :, jnp.newaxis] - config.object_zs[0]
    ) / config.voxel_size_z
    pys = jnp.arange(pvs.shape[1])[jnp.newaxis, :, jnp.newaxis] * jnp.ones_like(pvs)
    pus = (
        uu[:, jnp.newaxis, :] * ratios[jnp.newaxis, :, jnp.newaxis] - config.object_xs[0]
    ) / config.voxel_size_x

    if input_2d:
        ax0c, ax1c, ax2c = ((np.array(pvs.shape) + 1) / 2 - 1).tolist()
        r = jnp.hypot(pus - ax2c, pys - ax1c)
        ax1 = jnp.where(pys >= ax1c, ax1c + r, ax1c - r)
        proj2d = jnp.sum(map_coordinates(volume, [pvs, ax1], cval=0.0, order=1), axis=1)
    else:
        proj2d = jnp.sum(map_coordinates(volume, [pvs, pys, pus], cval=0.0, order=1), axis=1)

    dist = (
        jnp.sqrt(config.source_to_detector_dist**2.0 + uu**2.0 + vv**2.0)
        / (config.source_to_detector_dist)
        * config.voxel_size_y
    )

    return proj2d * dist
