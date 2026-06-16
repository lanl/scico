# -*- coding: utf-8 -*-
# Copyright (C) 2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Computed laminography functions."""

from typing import Union

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve

from ._xray import XRayTransform3D as scicoXRayTransform3D

try:  # scico.astra cannot be imported if astra is not installed
    from .astra import XRayTransform3D as astraXRayTransform3D
except ModuleNotFoundError:
    pass


def cl_angles_to_vecs(theta: np.ndarray, alpha: float = 60.0 * (np.pi / 180.0)) -> np.ndarray:
    r"""Construct astra geometry vectors from laminography view angles.

    Construct parallel beam astra geometry vectors from laminography
    view angles. The vectors are computed as in :cite:`aarle-2016-fast`,
    with modifications for parallel beam geometry.

    Args:
        theta: View angles in radians around laminography rotation axis.
        alpha: Laminography tilt angle (see angle :math:`\alpha` in Fig.
            3(a) in :cite:`aarle-2016-fast`) in radians.
    Returns:
        An array of astra "parallel3d_vec" geometry specification vectors,
        as described in the documentation of :class:`.astra.XRayTransform3D`.
    """
    ones = np.ones_like(theta)
    𝛥s, 𝛥t, 𝛥d = 1.0, 1.0, 1.0
    ray = (
        -𝛥s * np.sin(alpha) * np.sin(theta),
        𝛥s * np.sin(alpha) * np.cos(theta),
        𝛥s * np.cos(alpha) * ones,
    )
    d = (
        -𝛥d * np.sin(alpha) * np.sin(theta),
        𝛥d * np.sin(alpha) * np.cos(theta),
        𝛥d * np.cos(alpha) * ones,
    )
    u = (𝛥t * np.cos(theta), 𝛥t * np.sin(theta), 0.0 * ones)
    v = (
        𝛥t * np.cos(alpha) * np.sin(theta),
        -𝛥t * np.cos(alpha) * np.cos(theta),
        𝛥t * np.sin(alpha) * ones,
    )
    vectors = np.stack((ray + d + u + v)).T
    return vectors.astype(np.float32)


@jax.jit
def _filter_projection(y: jax.Array, alpha: float) -> jax.Array:
    r"""Apply filter appropriate for CL FBP.

    Apply filter appropriate for CL FBP (see (13) in
    :cite:`myagotin-2013-efficient`).

    Args:
        y: Projection array of shape (Nrow, Nview, Ncol) where Nview is
           the number of views, and the sensor consists of Nrow
           :math:`\times` Ncol pixiels.
        alpha: Laminography tilt angle (see angle :math:`\alpha` in Fig.
           3(a) in :cite:`aarle-2016-fast`) in radians.

    Returns:
        Filtered projection array.
    """
    # Adapted from an implementation by Yancey Sechrest
    Nu = y.shape[2]
    nu = jnp.arange(Nu + 1) - Nu // 2
    H = jnp.where(
        nu == 0,
        jnp.sin(jnp.abs(alpha)) / 8.0,
        jnp.where(nu % 2, -jnp.sin(jnp.abs(alpha)) / (2.0 * np.pi**2 * nu**2 + (nu == 0)), 0),
    )
    Hy = convolve(jnp.pad(y, ((0, 0), (0, 0), (0, 1))), H.reshape((1, 1, -1)), mode="same")
    return Hy[..., :-1]


@jax.jit(static_argnums=(2,))
def cl_fbp(
    y: jax.Array, alpha: float, X: Union[scicoXRayTransform3D, astraXRayTransform3D]
) -> jax.Array:
    r"""Compute FBP reconstruction for CL geometry.

    Compute FBP reconstruction for CL geometry, based on the approach
    proposed in :cite:`myagotin-2013-efficient`.

    Args:
        y: Projection array of shape (Nrow, Nview, Ncol) where Nview is
           the number of views, and the sensor consists of Nrow
           :math:`\times` Ncol pixels.
        alpha: Laminography tilt angle (see angle :math:`\alpha` in Fig.
           3(a) in :cite:`aarle-2016-fast`) in radians.
        X: :class:`.xray.XRayTransform3D` or :class:`.astra.XRayTransform3D`
           linear operator corresponding to the imaging geometry.

    Returns:
        FBP reconstruction.
    """
    yf = _filter_projection(y, alpha)
    n_proj = y.shape[1]
    x = (2 * np.pi / n_proj) * X.T @ yf
    return x
