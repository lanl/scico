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

    Projection of a cylindrically symmetric volume onto a sensor plane
    using conical beam geometry.

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

    proj2d = jnp.sum(map_coordinates(volume, [pvs, pys, pus], cval=0.0, order=1), axis=1)

    dist = (
        jnp.sqrt(config.source_to_detector_dist**2.0 + uu**2.0 + vv**2.0)
        / (config.source_to_detector_dist)
        * config.voxel_size_y
    )

    return proj2d * dist
