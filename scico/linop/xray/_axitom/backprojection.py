"""
This file is a modified version of "backprojection.py" from the
[AXITOM](https://github.com/PolymerGuy/AXITOM) package.

Filtered back projection functions

This module contains the Feldkamp David Kress filtered back projection
routines.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

from .config import Config
from .filtering import ramp_filter_and_weight
from .utilities import rotate_coordinates


def map_object_to_detector_coords(object_xs, object_ys, object_zs, config):
    """Map the object coordinates to detector pixel coordinates
    accounting for cone beam divergence.

    Parameters
    ----------
    object_xs : np.ndarray
        The x-coordinate array of the object to be reconstructed
    object_ys : np.ndarray
        The y-coordinate array of the object to be reconstructed
    object_zs : np.ndarray
        The z-coordinate array of the object to be reconstructed
    config : obj
        The config object containing all necessary settings for the
        reconstruction

    Returns
    -------
    detector_cords_a
        The detector coordinates along the a-axis corresponding to the
        given points
    detector_cords_b
        The detector coordinates along the b-axis corresponding to the
        given points
    """

    detector_cords_a = (
        ((object_ys * config.source_to_detector_dist) / (object_xs + config.source_to_object_dist))
        - config.detector_us[0]
    ) / config.pixel_size_u

    if object_xs.ndim == 2:
        detector_cords_b = (
            (
                (object_zs[jnp.newaxis, jnp.newaxis, :] * config.source_to_detector_dist)
                / (object_xs[:, :, jnp.newaxis] + config.source_to_object_dist)
            )
            - config.detector_vs[0]
        ) / config.pixel_size_v

    elif object_xs.ndim == 1:
        detector_cords_b = (
            (
                (object_zs[jnp.newaxis, :] * config.source_to_detector_dist)
                / (object_xs[:, jnp.newaxis] + config.source_to_object_dist)
            )
            - config.detector_vs[0]
        ) / config.pixel_size_v
    else:
        raise ValueError("Invalid dimensions on the object coordinates")

    return detector_cords_a, detector_cords_b


def _fdk_axisym(projection_filtered, config, angles):
    """Filtered back projection algorithm as proposed by Feldkamp David
    Kress, adapted for axisymmetry.

    This implementation has been adapted for axis-symmetry by using a
    single projection only and by only reconstructing a single R-Z slice.

    This algorithm is based on:
       https://doi.org/10.1364/JOSAA.1.000612
    but follows the notation used by:
    Henrik Turbell, Cone-Beam Reconstruction Using Filtered
    Backprojection, PhD Thesis, Linkoping Studies in Science and
    Technology
    https://people.csail.mit.edu/bkph/courses/papers/Exact_Conebeam/Turbell_Thesis_FBP_2001.pdf

    Parameters
    ----------
    projection_filtered : jnp.ndarray
        The ramp filtered and weighted projection used in the reconstruction
    config : obj
        The config object containing all necessary settings for the reconstruction


    Returns
    -------
    ndarray
        The reconstructed slice is a R-Z plane of a axis-symmetric
        tomogram where Z is the symmetry axis.
    """
    proj_width, proj_height = projection_filtered.shape
    proj_center = int(proj_width / 2)

    # Allocate an empty array
    recon_slice = jnp.zeros((proj_width, proj_height), dtype=jnp.float32)

    for frame_nr, angle in enumerate(angles):
        x_rotated, y_rotated = rotate_coordinates(
            jnp.zeros_like(config.object_xs),
            config.object_ys,
            jnp.radians(angle),
        )
        detector_cords_a, detector_cords_b = map_object_to_detector_coords(
            x_rotated, y_rotated, config.object_zs, config
        )
        # a is independent of Z but has to match the shape of b
        detector_cords_a = detector_cords_a[:, jnp.newaxis] * jnp.ones_like(detector_cords_b)
        # This term is caused by the divergent cone geometry
        ratio = (config.source_to_object_dist**2.0) / (
            config.source_to_object_dist + x_rotated
        ) ** 2.0
        recon_slice = recon_slice + ratio[:, jnp.newaxis] * map_coordinates(
            projection_filtered, [detector_cords_a, detector_cords_b], cval=0.0, order=1
        )

    return recon_slice / angles.size


def fdk(projection: Array, config: Config, angles: Optional[Array] = None) -> Array:
    """Filtered back projection algorithm as proposed by Feldkamp David
    Kress, adapted for axisymmetry.

    This implementation has been adapted for axis-symmetry by using a
    single projection only and by only reconstructing a single R-Z slice.

    This algorithm is based on:
       https://doi.org/10.1364/JOSAA.1.000612
    but follows the notation used by:
    Henrik Turbell, Cone-Beam Reconstruction Using Filtered
    Backprojection, PhD Thesis, Linkoping Studies in Science and
    Technology
    https://people.csail.mit.edu/bkph/courses/papers/Exact_Conebeam/Turbell_Thesis_FBP_2001.pdf

    Args:
      projection: The projection used in the reconstruction
      config: The config object containing all necessary settings for the
        reconstruction.
      angles: Array of angles at which reconstruction should be computed.
        Defaults to 0 to 359 degrees with a 1 degree step.

    Returns:
        The reconstructed slice is a R-Z plane of a axis-symmetric
        tomogram where Z is the symmetry axis.
    """
    if angles is None:
        angles = jnp.arange(0, 360)

    if not isinstance(config, Config):
        raise ValueError("Only instances of Config are valid settings")

    if projection.ndim == 2:
        projection_filtered = ramp_filter_and_weight(projection, config)
    else:
        raise ValueError("The projection has to be a 2D array")

    tomo = _fdk_axisym(projection_filtered, config, angles)
    return tomo
