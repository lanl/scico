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
import jax.scipy.spatial.transform as jsst
from jax import Array
from jax.image import ResizeMethod, scale_and_translate
from jax.scipy.ndimage import map_coordinates
from jax.typing import ArrayLike

import scico.linop.xray.astra
import scipy.spatial.transform as sst


def image_centroid(v: ArrayLike, center_offset: bool = False) -> Tuple[float, ...]:
    """Compute the centroid of an image.

    Compute the centroid of an image or higher-dimensional array.

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

    Translate an image (or higher-dimensional array) so that the centroid
    is at the spatial center of the image grid.

    Args:
        v: Array to be centered.
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
    rot: jsst.Rotation,
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
    r"""Estimate an image alignment rotation.

    Estimate the rotation that best aligns vertical straight lines in
    the image with the vertical axis.

    The approach is roughly based on that used in the
    :code:`find_img_rotation_2D` function in the `cSAXS base package`
    released by the CXS group at the Paul Scherrer Institute, which
    finds the rotation angle that results in the sparsest column sum
    according to the sparsity measure proposed in Sec 3.1 of
    :cite:`hoyer-2004-nonnegative`. (Note that an :math:`\ell_1` norm
    sparsity measure is not suitable for this purpose since it is, in
    typical cases, appropximately invariant to the rotation angle.) The
    implementation here uses the plain ratio of :math:`\ell_1` and
    :math:`\ell_2` norms as a sparsity measure, more efficiently computes
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
        vertical (0) axis.

    Notes:
        The number number of detector pixels for the 2D X-ray transform
        is chosen based on the shape :math:`(N_0, N_1)` of :code:`img`
        and the value :math:`\theta` of parameter :code:`max_angle`, as
        indicated in Fig. 1.

        .. figure:: /figures/img_align.svg
           :align: center
           :width: 40%

           Fig 1. Calculation of the number of detector pixels for the 2D
           X-ray transform.


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


def volume_alignment_rotation(
    vol: ArrayLike,
    xslice: Optional[int] = None,
    yslice: Optional[int] = None,
    max_angle: float = 2.5,
    angle_step: float = 0.025,
    center_factor: float = 5e-3,
) -> jsst.Rotation:
    r"""Estimate a volume alignment rotation.

    Estimate the 3D rotation that best aligns planar structures in a
    volume with the x-y (0-1) plane. The algorithm is based on
    independent rotation angle estimates, obtained using
    :func:`image_alignment_rotation`, within 2D slices in the x-z (0-2)
    and y-z (1-2) planes. These estimates are integrated into a
    combined 3D rotation specification as explained in the technical note
    below.

    Args:
        vol: Array of voxel values.
        xslice: Index of slice on axis 0.
        yslice: Index of slice on axis 1.
        max_angle: Maximum  angle (negative and positive) to test, in
          degrees.
        angle_step: Increment in angle values for range of angles to
          test, in degrees.
        center_factor: The angle multiplied by this scalar is added to
          the sparsity measure to slightly prefer smaller-angle
          solutions.

    Returns:
        Rotation object.

    Notes:
        The estimation of the 3D rotation required to align planar
        structure in the volume with the x-y (0-1) plane is approached
        by estimating the 3D normal vector to this structure, illustrated
        in Fig. 1. The independent rotation angle estimates with the x-z
        (0-2) and y-z (1-2) planes are exploited as estimates (after a
        90° rotation of each) as estimates of the projections of this
        normal vector into the x-z (0-2) and y-z (1-2) planes,
        illustrated in Figs. 2 and 3 respectively.

        .. figure:: /figures/vol_align_xyz.svg
           :align: center
           :width: 60%

           Fig 1. 3D orientation of the normal to the plane that is
           desired to be aligned with the x-y plane.


        .. list-table::
           :width: 100

           * - .. figure:: /figures/vol_align_xz.svg
                  :align: center
                  :width: 100%

                  Fig 2. Projection of the normal onto the x-z plane.

             - .. figure:: /figures/vol_align_yz.svg
                  :align: center
                  :width: 100%

                  Fig 3. Projection of the normal onto the y-z plane.

        It can be observed from these figures that

        .. math::

           x &= r_x \cos (\theta_x) \\
           y &= r_y \cos (\theta_y) \\
           z &= r_x \sin (\theta_x) = r_y \sin (\theta_y) \;,

        where :math:`(x, y, z)` are the coordinates of the normal
        vector. We can write

        .. math::

           r_x = \frac{z}{\sin(\theta_x)} \quad \text{and} \quad
           r_y = \frac{z}{\sin(\theta_y)} \;,

        and therefore

        .. math::
           x = z \cot (\theta_x) \quad \text{and} \quad
           y = z \cot (\theta_y) \;.

        Since :math:`(x, y, z) = z (\cot (\theta_x), \cot (\theta_y), 1)`
        it is clear that the choice of :math:`z` only affects the norm of
        the vector, and can therefore be set to unity. The rotation of
        this vector is then determined by computing the rotation required
        to align it (after normalization) with the :math:`z` axis
        :math:`(0, 0, 1)`.
    """
    # x, y, z volume axes correspond to axes 0, 1, 2
    if xslice is None:
        xslice = vol.shape[0] // 2  # default to central slice
    if yslice is None:
        yslice = vol.shape[1] // 2  # default to central slice
    # projected angles of normal to plane angles identified in yz and xz slices
    angle_y = (
        (90 - image_alignment_rotation(vol[xslice], max_angle=max_angle, angle_step=angle_step))
        * np.pi
        / 180
    )
    angle_x = (
        (90 - image_alignment_rotation(vol[:, yslice], max_angle=max_angle, angle_step=angle_step))
        * np.pi
        / 180
    )
    # unit vector normal to plane
    vec = np.array([1.0 / np.tan(angle_x), 1.0 / np.tan(angle_y), 1.0])
    vec /= np.linalg.norm(vec)
    # rotation required to align unit vector with z axis
    r = sst.Rotation.align_vectors(vec, np.array([0, 0, 1]))[0]
    # jax.scipy.spatial.transform.Rotation does not have align_vectors method
    return jsst.Rotation.from_quat(r.as_quat())
