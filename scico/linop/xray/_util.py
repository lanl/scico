# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for CT data."""

from typing import Optional, Tuple

import numpy as np

import jax.numpy as jnp
from jax import Array
from jax.image import ResizeMethod, scale_and_translate
from jax.scipy.ndimage import map_coordinates
from jax.scipy.spatial.transform import Rotation
from jax.typing import ArrayLike

import scico.linop.xray.astra


def image_centroid(v: ArrayLike, center_offset: bool = False) -> Tuple[float, ...]:
    """Compute the centroid of an image.

    Compute the centroid of an image or higher-dimentional array.

    Args:
        v: Array for which centroid is to be computed.
        center_offset: If ``True``, compute centroid coordinates
           relative to the spatial center of the image.

    Returns:
        Tuple of centroid coordinates.
    """
    if center_offset:
        offset = (jnp.array(v.shape, dtype=jnp.float32) - 1.0) / 2.0
    else:
        offset = jnp.zeros((v.ndim,), dtype=jnp.float32)
    g1d = [jnp.arange(size, dtype=jnp.float32) - offset[idx] for idx, size in enumerate(v.shape)]
    g = jnp.meshgrid(*g1d, sparse=True, indexing="ij")
    m00 = v.astype(jnp.float32).sum()
    if m00 == 0.0:
        c = (0.0,) * v.ndim
    else:
        c = tuple([(jnp.sum(v * g[idx]) / m00).item() for idx in range(v.ndim)])

    return c


def center_image(
    v: ArrayLike,
    axes: Optional[Tuple[int, ...]] = None,
    method: ResizeMethod = ResizeMethod.LANCZOS3,
) -> Array:
    """Translate an image to center the centroid.

    Translate an image (or higher-dimentional array) so that the centroid
    is at the spatial center of the image grid.

    Args:
        vol: Array to be centered.
        axes: Array axes on which centering is to be applied. Defaults to
          all axes.
        method: Interpolation method for image translation.

    Returns:
        Centered array.
    """
    if axes is None:
        axes = tuple(range(v.ndim))
    c = jnp.array(image_centroid(v, center_offset=True), dtype=jnp.float32)
    scale = jnp.ones((v.ndim,), dtype=jnp.float32)[jnp.array(axes)]
    trans = -c[jnp.array(axes)]
    cv = scale_and_translate(v, v.shape, axes, scale, trans, method=method)
    return cv


def rotate_volume(
    vol: ArrayLike,
    rot: Rotation,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
) -> Array:
    """Rotate a 3D array.

    Rotate a 3D array as specified by an instance of
    :class:`~jax.scipy.spatial.transform.Rotation`. Any axis coordinates
    that are not specified default to a range corresponding to the size
    of the array on that axis, starting at zero.

    Args:
        vol: Array to be rotated.
        rot: Rotation specification.
        x: Coordinates for :code:`x` axis (axis 0).
        y: Coordinates for :code:`y` axis (axis 1).
        z: Coordinates for :code:`z` axis (axis 2).
        center: A 3-vector specifying the center of rotation.
           Defaults to the center of the array.

    Returns:
        Rotated array.
    """
    shape = vol.shape
    if x is None:
        x = jnp.arange(shape[0])
    if y is None:
        y = jnp.arange(shape[1])
    if z is None:
        z = jnp.arange(shape[2])
    if center is None:
        center = (jnp.array(shape, dtype=jnp.float32) - 1.0) / 2.0
    gx, gy, gz = jnp.meshgrid(x - center[0], y - center[1], z - center[2], indexing="ij")
    crd = jnp.stack((gx.ravel(), gy.ravel(), gz.ravel()))
    rot_crd = rot.as_matrix() @ crd + center[:, jnp.newaxis]  # faster than rot.apply(crd.T)
    rot_vol = map_coordinates(vol, rot_crd.reshape((3,) + shape), order=1)
    return rot_vol


def image_alignment_rotation(
    img: ArrayLike, max_angle: float = 2.5, angle_step: float = 0.025, center_factor: float = 5e-3
) -> float:
    r"""Compute an image alignment rotation.

    Compute the rotation that best aligns vertical straight lines in
    the image with the vertical axis.

    The approach is roughly based on that used in the
    `find_img_rotation_2D` function in the `cSAXS base package` released
    by the CXS group at the Paul Scherrer Institute, which selects the
    rotation angle that results in the sparsest row sum according to the
    sparsity measure proposed in Sec 3.1 of
    :cite:`hoyer-2004-nonnegative`. (Note that an :math:`\ell_1` norm
    sparsity measure is not suitable for this purpose since it is, in
    typical cases, appropximately invariant to the rotation angle.) The
    implementation here uses the plain ratio of :math:`\ell_1` and
    :math:`\ell_2`norms as a sparsity measure, more efficiently computes
    the column sums at different angles by exploiting the 2D X-ray
    transform, and includes a small bias for smaller angle rotations that
    improves performance when a range of rotation angles have the same
    sparsity measure.

    Args:
        img: Array of pixel values.
        max_angle: Maximum  angle (negative and positive) to test, in
          degrees.
        angle_step: Increment in angle values for range of angles to
          test, in degrees.
        center_factor: The angle multiplied by this scalar is added to
          the sparsity measure to slightly prefer smaller-angle
          solutions.

    Returns:
        Rotation angle (in degrees) providing best alignment with the
        horizontal axis.
    """
    angles = np.arange(-max_angle, max_angle, angle_step)
    max_angle_rad = max_angle * np.pi / 180
    # choose det_count so that projected image is within the detector bounds
    det_count = int(
        1.05 * (img.shape[0] * np.sin(max_angle_rad) + img.shape[1] * np.cos(max_angle_rad))
    )
    A = scico.linop.xray.astra.XRayTransform2D(
        img.shape,
        det_count=det_count,
        det_spacing=1.0,
        angles=angles * np.pi / 180.0,
    )
    y = A @ jnp.abs(img)
    # compute the ℓ1/ℓ2 norm of the projection for each view angle
    cost = jnp.sum(jnp.abs(y), axis=1) / jnp.sqrt(jnp.sum(y**2, axis=1))
    ext_cost = cost + center_factor * (cost.max() - cost.min()) * jnp.abs(angles)
    idx = jnp.argmin(ext_cost)
    return angles[idx]
