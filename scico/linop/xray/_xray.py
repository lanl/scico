# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform classes."""


from functools import partial
from typing import Optional
from warnings import warn

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from scico.numpy.util import is_scalar_equiv
from scico.typing import Shape

from .._linop import LinearOperator


class XRayTransform(LinearOperator):
    """X-ray transform operator.

    Wrap an X-ray projector object in a SCICO :class:`LinearOperator`.
    """

    def __init__(self, projector):
        r"""
        Args:
            projector: instance of an X-ray projector object to wrap,
                currently the only option is
                :class:`Parallel2dProjector`
        """
        self.projector = projector
        self._eval = projector.project

        super().__init__(
            input_shape=projector.im_shape,
            output_shape=(len(projector.angles), projector.det_count),
        )


class Parallel2dProjector:
    """Parallel ray, single axis, 2D X-ray projector.

    This implementation approximates the projection of each rectangular
    pixel as a boxcar function (whereas the exact projection is a
    trapezoid). Detector pixels are modeled as bins (rather than points)
    and this approximation allows fast calculation of the contribution
    of each pixel to each bin because the integral of the boxcar is
    simple.

    By requiring the width of a projected pixel to be less than or equal
    to the bin width (which is defined to be 1.0), we ensure that
    each pixel contributes to at most two bins, which accelerates the
    accumulation of pixel values into bins (equivalently, makes the
    linear operator sparse).

    `x0`, `dx`, and `y0` should be expressed in units such that the
    detector spacing `dy` is 1.0.
    """

    def __init__(
        self,
        im_shape: Shape,
        angles: ArrayLike,
        x0: Optional[ArrayLike] = None,
        dx: Optional[ArrayLike] = None,
        y0: Optional[float] = None,
        det_count: Optional[int] = None,
    ):
        r"""
        Args:
            im_shape: Shape of input array.
            angles: (num_angles,) array of angles in radians. Viewing an
                (M, N) array as a matrix with M rows and N columns, an
                angle of 0 corresponds to summing rows, an angle of pi/2
                corresponds to summing columns, and an angle of pi/4
                corresponds to summing along antidiagonals.
            x0: (x, y) position of the corner of the pixel `im[0,0]`. By
                default, `(-im_shape / 2, -im_shape / 2)`.
            dx: Image pixel side length in x- and y-direction. Should be
                <= 1.0 in each dimension. By default, [1.0, 1.0].
            y0: Location of the edge of the first detector bin. By
                default, `-det_count / 2`
            det_count: Number of elements in detector. If ``None``,
                defaults to the size of the diagonal of `im_shape`.
        """
        self.im_shape = im_shape
        self.angles = angles

        self.nx = np.array(im_shape)
        if dx is None:
            dx = np.full((2,), np.sqrt(2) / 2)
        if is_scalar_equiv(dx):
            dx = dx * np.ones(2)
        self.dx = dx

        # check projected pixel width assumption
        Pdx = np.stack((dx[0] * jnp.cos(angles), dx[1] * jnp.sin(angles)))
        Pdiag1 = np.abs(Pdx[0] + Pdx[1])
        Pdiag2 = np.abs(Pdx[0] - Pdx[1])
        max_width = np.max(np.maximum(Pdiag1, Pdiag2))

        if max_width > 1:
            warn(
                f"A projected pixel has width {max_width} > 1.0, "
                "which will reduce projector accuracy."
            )

        if x0 is None:
            x0 = -(self.nx * self.dx) / 2
        self.x0 = x0

        if det_count is None:
            det_count = int(np.ceil(np.linalg.norm(im_shape)))
        self.det_count = det_count
        self.ny = det_count

        if y0 is None:
            y0 = -self.ny / 2
        self.y0 = y0
        self.dy = 1.0

    def project(self, im):
        """Compute X-ray projection."""
        return _project(im, self.x0, self.dx, self.y0, self.ny, self.angles)

    def back_project(self, y):
        """Compute X-ray back projection"""
        return _back_project(y, self.x0, self.dx, tuple(self.nx), self.y0, self.angles)


@partial(jax.jit, static_argnames=["ny"])
def _project(im, x0, dx, y0, ny, angles):
    r"""
    Args:
        im: Input array, (M, N).
        x0: (x, y) position of the corner of the pixel im[0,0].
        dx: Pixel side length in x- and y-direction. Units are such
            that the detector bins have length 1.0.
        y0: Location of the edge of the first detector bin.
        ny: Number of detector bins.
        angles: (num_angles,) array of angles in radians. Pixels are
            projected onto units vectors pointing in these directions.
    """
    nx = im.shape
    inds, weights = _calc_weights(x0, dx, nx, angles, y0)
    # Handle out of bounds indices. In the .at call, inds >= y0 are
    # ignored, while inds < 0 wrap around. So we set inds < 0 to ny.
    inds = jnp.where(inds >= 0, inds, ny)

    y = (
        jnp.zeros((len(angles), ny))
        .at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds]
        .add(im * weights)
    )

    y = y.at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds + 1].add(im * (1 - weights))

    return y


@partial(jax.jit, static_argnames=["nx"])
def _back_project(y, x0, dx, nx, y0, angles):
    r"""
    Args:
        y: Input projection, (num_angles, N).
        x0: (x, y) position of the corner of the pixel im[0,0].
        dx: Pixel side length in x- and y-direction. Units are such
            that the detector bins have length 1.0.
        nx: Shape of back projection.
        y0: Location of the edge of the first detector bin.
        angles: (num_angles,) array of angles in radians. Pixels are
            projected onto units vectors pointing in these directions.
    """
    ny = y.shape[1]
    inds, weights = _calc_weights(x0, dx, nx, angles, y0)
    # Handle out of bounds indices. In the .at call, inds >= y0 are
    # ignored, while inds < 0 wrap around. So we set inds < 0 to ny.
    inds = jnp.where(inds >= 0, inds, ny)

    # the idea: [y[0, inds[0]], y[1, inds[1]], ...]
    HTy = jnp.sum(y[jnp.arange(len(angles)).reshape(-1, 1, 1), inds] * weights, axis=0)
    HTy = HTy + jnp.sum(
        y[jnp.arange(len(angles)).reshape(-1, 1, 1), inds + 1] * (1 - weights), axis=0
    )

    return HTy


@partial(jax.jit, static_argnames=["nx", "y0"])
@partial(jax.vmap, in_axes=(None, None, None, 0, None))
def _calc_weights(x0, dx, nx, angle, y0):
    """

    Args:
        x0: Location of the corner of the pixel im[0,0].
        dx: Pixel side length in x- and y-direction. Units are such
            that the detector bins have length 1.0.
        nx: Input image shape.
        angle: (num_angles,) array of angles in radians. Pixels are
            projected onto units vectors pointing in these directions.
            (This argument is `vmap`ed.)
        y0: Location of the edge of the first detector bin.
    """
    u = [jnp.cos(angle), jnp.sin(angle)]
    Px0 = x0[0] * u[0] + x0[1] * u[1] - y0
    Pdx = [dx[0] * u[0], dx[1] * u[1]]
    Pxmin = jnp.min(jnp.array([Px0, Px0 + Pdx[0], Px0 + Pdx[1], Px0 + Pdx[0] + Pdx[1]]))

    Px = (
        Pxmin
        + Pdx[0] * jnp.arange(nx[0]).reshape(-1, 1)
        + Pdx[1] * jnp.arange(nx[1]).reshape(1, -1)
    )

    # detector bin inds
    inds = jnp.floor(Px).astype(int)

    # weights
    Pdx = jnp.array(u) * jnp.array(dx)
    diag1 = jnp.abs(Pdx[0] + Pdx[1])
    diag2 = jnp.abs(Pdx[0] - Pdx[1])
    w = jnp.max(jnp.array([diag1, diag2]))
    f = jnp.min(jnp.array([diag1, diag2]))

    width = (w + f) / 2
    distance_to_next = 1 - (Px - inds)  # always in (0, 1]
    weights = jnp.minimum(distance_to_next, width) / width

    return inds, weights
