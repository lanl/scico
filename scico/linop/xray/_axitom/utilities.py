"""
This file is a modified version of "utilities.py" from the
[AXITOM](https://github.com/PolymerGuy/AXITOM) package.

Utilites

This module contains various utility functions that does not have any
other obvious home.
"""

import numpy as np

import jax.numpy as jnp


def _find_center_of_gravity_in_projection(projection, background_internsity=0.9):
    """Find axis of rotation in the projection.
    This is done by binarization of the image into object and background
    and determining the center of gravity of the object.

    Parameters
    ----------
    projection : ndarray
        The projection, normalized between 0 and 1
    background_internsity : float
        The background intensity threshold


    Returns
    -------
    float64
        The center of gravity in the u-direction
    float64
        The center of gravity in the v-direction

    """
    m, n = np.shape(projection)

    binary_proj = np.zeros_like(projection, dtype=np.float)
    binary_proj[projection < background_internsity] = 1.0

    area_x = np.sum(binary_proj, axis=1)
    area_y = np.sum(binary_proj, axis=0)

    non_zero_rows = np.arange(n)[area_y != 0.0]
    non_zero_columns = np.arange(m)[area_x != 0.0]

    # Now removing all columns that does not intersect the object
    object_pixels = binary_proj[non_zero_columns, :][:, non_zero_rows]
    area_x = area_x[non_zero_columns]
    area_y = area_y[non_zero_rows]
    xs, ys = np.meshgrid(non_zero_rows, non_zero_columns)

    # Determine center of gravity
    center_of_grav_x = np.average(np.sum(xs * object_pixels, axis=1) / area_x) - n / 2.0
    center_of_grav_y = np.average(np.sum(ys * object_pixels, axis=0) / area_y) - m / 2.0
    return center_of_grav_x, center_of_grav_y


def find_center_of_rotation(projection, background_internsity=0.9, method="center_of_gravity"):
    """Find the axis of rotation of the object in the projection

    Parameters
    ----------
    projection : ndarray
        The projection, normalized between 0 and 1
    background_internsity : float
        The background intensity threshold
    method : string
        The background intensity threshold


    Returns
    -------
    float64
        The center of gravity in the v-direction
    float64
        The center of gravity in the u-direction

    """
    if projection.ndim != 2:
        raise ValueError("Invalid projection shape. It has to be a 2d numpy array")

    if method == "center_of_gravity":
        center_v, center_u = _find_center_of_gravity_in_projection(
            projection, background_internsity
        )
    else:
        raise ValueError("Invalid method")

    return center_v, center_u


def rotate_coordinates(xs_array, ys_array, angle_rad):
    """Rotate coordinate arrays by a given angle

    Parameters
    ----------
    xs_array : ndarray
        Two dimensional coordinate array with x-coordinates
    ys_array : ndarray
        Two dimensional coordinate array with y-coordinates
    angle_rad : float
        Rotation angle in radians

    Returns
    -------
    ndarray
        The rotated x-coordinates
    ndarray
        The rotated x-coordinates

    """
    rx = xs_array * jnp.cos(angle_rad) + ys_array * jnp.sin(angle_rad)
    ry = -xs_array * jnp.sin(angle_rad) + ys_array * jnp.cos(angle_rad)
    return rx, ry
