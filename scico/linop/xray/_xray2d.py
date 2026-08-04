# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""2D X-ray transform class."""

from functools import partial
from typing import Optional, Tuple
from warnings import warn

import numpy as np

import jax
import jax.numpy as jnp
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax.typing import ArrayLike

import scico.numpy as snp
from scico.numpy.util import is_scalar_equiv
from scico.typing import Shape

from .._linop import LinearOperator


class XRayTransform2D(LinearOperator):
    r"""Parallel ray, single axis, 2D X-ray projector.

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

    Warning: The default pixel spacing is :math:`\sqrt{2}/2` (rather
    than 1) in order to satisfy the aforementioned spacing requirement.

    `x0`, `dx`, and `y0` should be expressed in units such that the
    detector spacing `dy` is 1.0.
    """

    def __init__(
        self,
        input_shape: Shape,
        angles: ArrayLike,
        x0: Optional[ArrayLike] = None,
        dx: Optional[ArrayLike] = None,
        y0: Optional[float] = None,
        det_count: Optional[int] = None,
        input_device: xc.Device | Sharding | None = None,
        output_device: xc.Device | Sharding | None = None,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            angles: (num_angles,) array of angles in radians. Viewing an
                (M, N) array as a matrix with M rows and N columns, an
                angle of 0 corresponds to summing rows, an angle of pi/2
                corresponds to summing columns, and an angle of pi/4
                corresponds to summing along antidiagonals.
            x0: (x, y) position of the corner of the pixel `im[0,0]`. By
                default, `(-input_shape * dx[0] / 2, -input_shape * dx[1] / 2)`.
            dx: Image pixel side length in x- and y-direction (axis 0 and
                1 respectively). Must be set so that the width of a
                projected pixel is never larger than 1.0. By default,
                [:math:`\sqrt{2}/2`, :math:`\sqrt{2}/2`].
            y0: Location of the edge of the first detector bin. By
                default, `-det_count / 2`
            det_count: Number of elements in detector. If ``None``,
                defaults to the size of the diagonal of `input_shape`.
            input_device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                for input arrays.
            output_device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                for output arrays.
        """
        self.input_shape = input_shape
        self.angles = angles

        self.nx = tuple(input_shape)
        if dx is None:
            dx = 2 * (np.sqrt(2) / 2,)
        if is_scalar_equiv(dx):
            dx = 2 * (dx,)
        self.dx = dx

        # check projected pixel width assumption
        Pdx = np.stack((dx[0] * jnp.cos(angles), dx[1] * jnp.sin(angles)))
        Pdiag1 = np.abs(Pdx[0] + Pdx[1])
        Pdiag2 = np.abs(Pdx[0] - Pdx[1])
        max_width: float = np.max(np.maximum(Pdiag1, Pdiag2))

        if max_width > 1:
            warn(
                f"A projected pixel has width {max_width} > 1.0, "
                "which will reduce projector accuracy."
            )

        if x0 is None:
            x0 = -(np.array(self.nx) * self.dx) / 2
        self.x0 = x0

        if det_count is None:
            det_count = int(np.ceil(np.linalg.norm(input_shape)))
        self.det_count = det_count
        self.ny = det_count
        self.output_shape = (len(angles), det_count)

        if y0 is None:
            y0 = -self.ny / 2
        self.y0 = y0
        self.dy = 1.0

        self.fbp_filter: Optional[snp.Array] = None
        self.fbp_mask: Optional[snp.Array] = None

        self.input_device = input_device
        self.output_device = output_device

        super().__init__(
            input_shape=self.input_shape,
            input_dtype=np.float32,
            output_shape=self.output_shape,
            output_dtype=np.float32,
            eval_fn=self.project,
            adj_fn=self.back_project,
        )

    def project(self, im: ArrayLike) -> snp.Array:
        """Compute X-ray projection, equivalent to `H @ im`.

        Args:
            im: Input array representing the image to project.
        """
        return XRayTransform2D._project(
            im, self.x0, self.dx, self.y0, self.ny, self.angles, device=self.output_device
        )

    def back_project(self, y: ArrayLike) -> snp.Array:
        """Compute X-ray back projection, equivalent to `H.T @ y`.

        Args:
            y: Input array representing the sinogram to back project.
        """
        return XRayTransform2D._back_project(
            y, self.x0, self.dx, self.nx, self.y0, self.angles, device=self.input_device
        )

    def fbp(self, y: ArrayLike) -> snp.Array:
        r"""Compute filtered back projection (FBP) inverse of projection.

        Compute the filtered back projection inverse by filtering each
        row of the sinogram with the filter defined in (61) in
        :cite:`kak-1988-principles` and then back projecting. The
        projection angles are assumed to be evenly spaced in
        :math:`[0, \pi)`; reconstruction quality may be poor if
        this assumption is violated. Poor quality reconstructions should
        also be expected when `dx[0]` and `dx[1]` are not equal.

        Args:
            y: Input projection, (num_angles, N).

        Returns:
            FBP inverse of projection.
        """
        N = y.shape[1]

        if self.fbp_filter is None:
            nvec = jnp.arange(N) - (N - 1) // 2
            self.fbp_filter = XRayTransform2D._ramp_filter(nvec, 1.0).reshape(1, -1)

        if self.fbp_mask is None:
            unit_sino = jnp.ones_like(y)
            # Threshold is multiplied by 0.99... fudge factor to account for numerical errors
            # in back projection.
            self.fbp_mask = self.back_project(unit_sino) >= (self.output_shape[0] * (1.0 - 1e-5))  # type: ignore

        # Apply ramp filter in the frequency domain, padding to avoid
        # boundary effects
        h = self.fbp_filter
        hf = jnp.fft.fft(h, n=2 * N - 1, axis=1)
        yf = jnp.fft.fft(y, n=2 * N - 1, axis=1)
        hy = jnp.fft.ifft(hf * yf, n=2 * N - 1, axis=1)[
            :, (N - 1) // 2 : -(N - 1) // 2
        ].real.astype(jnp.float32)

        x = (jnp.pi * self.dx[0] * self.dx[1] / y.shape[0]) * self.fbp_mask * self.back_project(hy)  # type: ignore
        return x

    @staticmethod
    def _ramp_filter(x: ArrayLike, tau: float) -> snp.Array:
        """Compute coefficients of ramp filter used in FBP.

        Compute coefficients of ramp filter used in FBP, as defined in
        (61) in :cite:`kak-1988-principles`.

        Args:
            x: Sampling locations at which to compute filter coefficients.
            tau: Sampling rate.

        Returns:
            Spatial-domain coefficients of ramp filter.
        """
        # The (x == 0) term in x**2 * np.pi**2 * tau**2 + (x == 0)
        # is included to avoid division by zero warnings when x == 1
        # since np.where evaluates all values for both True and False
        # branches.
        return jnp.where(
            x == 0,
            1.0 / (4.0 * tau**2),
            jnp.where(x % 2, -1.0 / (x**2 * np.pi**2 * tau**2 + (x == 0)), 0),
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["ny"])
    def _project(
        im: ArrayLike,
        x0: ArrayLike,
        dx: ArrayLike,
        y0: float,
        ny: int,
        angles: ArrayLike,
        device: xc.Device | Sharding | None = None,
    ) -> snp.Array:
        r"""Compute X-ray projection.

        Args:
            im: Input array, (M, N).
            x0: (x, y) position of the corner of the pixel im[0,0].
            dx: Pixel side length in x- and y-direction. Units are such
                that the detector bins have length 1.0.
            y0: Location of the edge of the first detector bin.
            ny: Number of detector bins.
            angles: (num_angles,) array of angles in radians. Pixels are
                projected onto unit vectors pointing in these directions.
            device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                to which the output will be committed.
        """
        nx = im.shape
        inds, weights = XRayTransform2D._calc_weights(x0, dx, nx, angles, y0)

        # avoid incompatible types in the .add (scatter operation)
        weights = weights.astype(im.dtype)

        # Handle out of bounds indices by setting weight to zero
        weights_valid = jnp.where((inds >= 0) * (inds < ny), weights, 0.0)
        y = (
            jnp.zeros((len(angles), ny), dtype=im.dtype, device=device)
            .at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds]
            .add(im * weights_valid)
        )

        weights_valid = jnp.where((inds + 1 >= 0) * (inds + 1 < ny), 1 - weights, 0.0)
        y = y.at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds + 1].add(im * weights_valid)

        return y

    @staticmethod
    @partial(jax.jit, static_argnames=["nx"])
    def _back_project(
        y: ArrayLike,
        x0: ArrayLike,
        dx: ArrayLike,
        nx: Shape,
        y0: float,
        angles: ArrayLike,
        device: xc.Device | Sharding | None = None,
    ) -> snp.Array:
        r"""Compute X-ray back projection.

        Args:
            y: Input projection, (num_angles, N).
            x0: (x, y) position of the corner of the pixel im[0,0].
            dx: Pixel side length in x- and y-direction. Units are such
                that the detector bins have length 1.0.
            nx: Shape of back projection.
            y0: Location of the edge of the first detector bin.
            angles: (num_angles,) array of angles in radians. Pixels are
                projected onto units vectors pointing in these directions.
            device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                to which the output will be committed.
        """
        ny = y.shape[1]
        inds, weights = XRayTransform2D._calc_weights(x0, dx, nx, angles, y0)
        # Handle out of bounds indices by setting weight to zero
        weights_valid = jnp.where((inds >= 0) * (inds < ny), weights, 0.0)

        # the idea: [y[0, inds[0]], y[1, inds[1]], ...]
        HTy = jnp.zeros(nx, dtype=y.dtype, device=device)
        HTy += jnp.sum(y[jnp.arange(len(angles)).reshape(-1, 1, 1), inds] * weights_valid, axis=0)

        weights_valid = jnp.where((inds + 1 >= 0) * (inds + 1 < ny), 1 - weights, 0.0)
        HTy = HTy + jnp.sum(
            y[jnp.arange(len(angles)).reshape(-1, 1, 1), inds + 1] * weights_valid, axis=0
        )

        return HTy.astype(jnp.float32)

    @staticmethod
    @partial(jax.jit, static_argnames=["nx"])
    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _calc_weights(
        x0: ArrayLike, dx: ArrayLike, nx: Shape, angles: ArrayLike, y0: float
    ) -> Tuple[snp.Array, snp.Array]:
        """

        Args:
            x0: Location of the corner of the pixel im[0,0].
            dx: Pixel side length in x- and y-direction. Units are such
                that the detector bins have length 1.0.
            nx: Input image shape.
            angles: (num_angles,) array of angles in radians. Pixels are
                projected onto units vectors pointing in these directions.
                (This argument is `vmap`ed.)
            y0: Location of the edge of the first detector bin.
        """
        u = [jnp.cos(angles), jnp.sin(angles)]
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
