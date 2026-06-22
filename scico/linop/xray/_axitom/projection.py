"""
This file is a modified version of "projection.py" from the
[AXITOM](https://github.com/PolymerGuy/AXITOM) package.

Forward projection routines.

This module contains the functions used to forward project a volume onto
a sensor plane.
"""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.ndimage import map_coordinates

from .config import Config


@partial(jit, static_argnames=["config", "input_2d"])
def _partial_forward_project(
    volume: Array,
    uu: Array,
    vv: Array,
    irslab,
    config: Config,
    input_2d: bool = False,
) -> Array:
    """Partial projection of a volume onto a sensor plane.

    Partial projection of a cylindrically symmetric volume onto a sensor
    plane using conical beam geometry: this functional only sums along
    the section of the imaging direction specified by :code:`ratios`.

    Args:
      volume: The volume that will be projected onto the sensor.
      uu: Detector grid in axis 1 direction.
      vv: Detector grid in axis 0 direction.
      irslab: Array of indices and ratios.
      config: The settings object.
      input_2d: If ``True``, the input is a 2D image from which a 3D
        volume is constructed by rotation about the center of axis 1
        of the image.

    Returns:
        The projection.
    """
    islab = irslab[0]
    rslab = irslab[1]
    N = config.object_ys.size

    pvs = (
        vv[:, jnp.newaxis, :] * rslab[jnp.newaxis, :, jnp.newaxis] - config.object_zs[0]
    ) / config.voxel_size_z
    pys = islab[jnp.newaxis, :, jnp.newaxis] * jnp.ones_like(pvs)
    pus = (
        uu[:, jnp.newaxis, :] * rslab[jnp.newaxis, :, jnp.newaxis] - config.object_xs[0]
    ) / config.voxel_size_x

    if input_2d:
        ax0c, ax1c, ax2c = ((np.array(pvs.shape) + 1) / 2 - 1).tolist()
        ax1c = (N + 1) / 2 - 1
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


@partial(jit, static_argnames=["config", "num_slabs", "input_2d"])
def forward_project(
    volume: Array, config: Config, num_slabs: int = 8, input_2d: bool = False
) -> Array:
    """Projection of a volume onto a sensor plane.

    Projection of a cylindrically symmetric volume onto a sensor plane
    using conical beam geometry.

    Args:
      volume: The volume that will be projected onto the sensor.
      config: The settings object.
      num_slabs: Number of slabs into which the volume should be
        divided (for serial processing, to limit memory usage) in
        the imaging direction.
      input_2d: If ``True``, the input is a 2D image from which a 3D
        volume is constructed by rotation about the center of axis 1
        of the image.

    Returns:
        The projection.
    """
    uu, vv = jnp.meshgrid(config.detector_us, config.detector_vs)
    ratios = (config.object_ys + config.source_to_object_dist) / config.source_to_detector_dist
    N = ratios.size
    slab_size = N // num_slabs
    remainder = N % num_slabs
    islabs = jnp.stack(jnp.split(jnp.arange(0, slab_size * num_slabs), num_slabs))
    rslabs = jnp.stack(jnp.split(ratios[0 : slab_size * num_slabs], num_slabs))
    irslabs = jnp.stack((islabs, rslabs), axis=1)

    func = lambda irslab: _partial_forward_project(
        volume, uu, vv, irslab, config, input_2d=input_2d
    )
    # jax.checkpoint used to avoid excessive memory requirements
    proj = jnp.sum(jax.lax.map(jax.checkpoint(func), irslabs), axis=0)

    if remainder:
        irslab = jnp.stack((jnp.arange(slab_size * num_slabs, N), ratios[-remainder:]))
        proj += jax.checkpoint(func)(irslab)

    return proj
