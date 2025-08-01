"""
This file is a modified version of "filtering.py" from the
[AXITOM](https://github.com/PolymerGuy/AXITOM) package.

Filter tools

This module contains the ramp filter and the weighting function.
"""

import numpy as np

import jax.numpy as jnp
import jax.scipy.signal as sig


def _ramp_kernel_real(cutoff, length):
    """Ramp filter kernel in real space defined by the cut-off frequency
    and the spatial dimension.

    Parameters
    ----------
    cutoff : float
        The cut-off frequency
    length : int
        The kernel filter length

    Returns
    -------
    ndarray
        The filter kernel
    """
    pos = jnp.arange(-length, length, 1)
    return cutoff**2.0 * (2.0 * jnp.sinc(2 * pos * cutoff) - jnp.sinc(pos * cutoff) ** 2.0)


def _add_weights(projection, config):
    """Add weights to the projection according to the ray length traveled
    through a voxel.

    Parameters
    ----------
    projection : jnp.ndarray
        The projection used in the reconstruction
    config : obj
        The config object containing all necessary settings for the
        reconstruction

    Returns
    -------
    ndarray
        The projections weighted by the ray length
    """
    uu, vv = jnp.meshgrid(config.detector_vs, config.detector_us)

    weights = config.source_to_detector_dist / jnp.sqrt(
        config.source_to_detector_dist**2.0 + uu**2.0 + vv**2.0
    )

    return projection * weights


def ramp_filter_and_weight(projection, config):
    """Add weights to the projection and apply a ramp-high-pass filter
    set to 0.5*Nyquist_frequency

    Parameters
    ----------
    projection : jnp.ndarray
        The projection used in the reconstruction
    config : obj
        The config object containing all necessary settings for the
        reconstruction

    Returns
    -------
    ndarray
        The projections weighted by the ray length and filtered by ramp
        filter
    """
    projections_weighted = _add_weights(projection, config)

    n_pixels_u, _ = np.shape(projections_weighted)
    ramp_kernel = _ramp_kernel_real(0.5, n_pixels_u)

    projections_filtered = np.zeros_like(projections_weighted)

    _, n_lines = projections_weighted.shape

    for j in range(n_lines):
        projections_filtered[:, j] = sig.fftconvolve(
            projections_weighted[:, j], ramp_kernel, mode="same"
        )

    scale_factor = (
        1.0
        / config.pixel_size_u
        * np.pi
        * (config.source_to_detector_dist / config.source_to_object_dist)
    )

    return projections_filtered * scale_factor
