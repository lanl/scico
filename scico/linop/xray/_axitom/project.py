"""
Forward projection routines.

This module contains the functions used to forward project a volume onto
a sensor plane.
"""

from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.ndimage import map_coordinates

from .config import Config


@partial(jit, static_argnames=["config"])
def _forward_project(volume: Array, config: Config) -> Array:
    """Projection of a volume onto a sensor plane.

    Projection of a cylindricaly symmetric volume onto a sensor plane
    using conical beam geometry. The volume is represented by a 2D
    central slice, which is rotated about axis 1 to generate a 3D
    volume for projection.

    Args:
      volume: The volume that will be projected onto the sensor.
      config: The settings object.

    Returns:
        The projection.

    """
    # Integrate along the rays
    uu, vv = jnp.meshgrid(config.detector_us, config.detector_vs)

    ratios = (config.object_ys + config.source_to_object_dist) / config.source_to_detector_dist

    pus = (
        uu[:, jnp.newaxis, :] * ratios[jnp.newaxis, :, jnp.newaxis] - config.object_xs[0]
    ) / config.voxel_size_x
    pvs = (
        vv[:, jnp.newaxis, :] * ratios[jnp.newaxis, :, jnp.newaxis] - config.object_zs[0]
    ) / config.voxel_size_z
    pys = jnp.arange(pus.shape[1])[jnp.newaxis, :, jnp.newaxis] * jnp.ones_like(pvs)

    ax0c, ax1c = (pvs.shape[0] - 1) / 2, (pvs.shape[1] - 1) / 2
    r = jnp.hypot(pvs - ax0c, pys - ax1c)
    ax0 = jnp.where(pvs >= ax0c, ax0c + r, ax0c - r)

    proj2d = jnp.sum(
        map_coordinates(volume, [ax0, pus], cval=0.0, order=1),
        axis=1,
    )

    dist = (
        jnp.sqrt(config.source_to_detector_dist**2.0 + uu**2.0 + vv**2.0)
        / (config.source_to_detector_dist)
        * config.voxel_size_y
    )

    return proj2d * dist
