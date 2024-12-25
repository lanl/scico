# -*- coding: utf-8 -*-
# Copyright (C) 2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for CT data."""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax.scipy.spatial.transform import Rotation
from jax.typing import ArrayLike


def rotate_volume(
    vol: ArrayLike,
    rot: Rotation,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
):
    """Rotate a 3D array.

    Rotate a 3D array as specified by an instance of
    :class:`~jax.scipy.spatial.transform.Rotation`. Any axis coordinates
    that are not specified default to a range corresponding to the size
    of the array on that axis, shifted so that 0 is at the center of the
    array.

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
