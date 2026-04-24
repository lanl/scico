# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform classes."""

from functools import partial
from typing import Optional, Tuple
from warnings import warn

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import scico.numpy as snp
from scico.numpy.util import is_scalar_equiv
from scico.typing import Shape
from scipy.spatial.transform import Rotation

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
        return XRayTransform2D._project(im, self.x0, self.dx, self.y0, self.ny, self.angles)

    def back_project(self, y: ArrayLike) -> snp.Array:
        """Compute X-ray back projection, equivalent to `H.T @ y`.

        Args:
            y: Input array representing the sinogram to back project.
        """
        return XRayTransform2D._back_project(y, self.x0, self.dx, self.nx, self.y0, self.angles)

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
            unit_sino = jnp.ones(self.output_shape, dtype=np.float32)
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
        im: ArrayLike, x0: ArrayLike, dx: ArrayLike, y0: float, ny: int, angles: ArrayLike
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
        """
        nx = im.shape
        inds, weights = XRayTransform2D._calc_weights(x0, dx, nx, angles, y0)

        # avoid incompatible types in the .add (scatter operation)
        weights = weights.astype(im.dtype)

        # Handle out of bounds indices by setting weight to zero
        weights_valid = jnp.where((inds >= 0) * (inds < ny), weights, 0.0)
        y = (
            jnp.zeros((len(angles), ny), dtype=im.dtype)
            .at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds]
            .add(im * weights_valid)
        )

        weights_valid = jnp.where((inds + 1 >= 0) * (inds + 1 < ny), 1 - weights, 0.0)
        y = y.at[jnp.arange(len(angles)).reshape(-1, 1, 1), inds + 1].add(im * weights_valid)

        return y

    @staticmethod
    @partial(jax.jit, static_argnames=["nx"])
    def _back_project(
        y: ArrayLike, x0: ArrayLike, dx: ArrayLike, nx: Shape, y0: float, angles: ArrayLike
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
        """
        ny = y.shape[1]
        inds, weights = XRayTransform2D._calc_weights(x0, dx, nx, angles, y0)
        # Handle out of bounds indices by setting weight to zero
        weights_valid = jnp.where((inds >= 0) * (inds < ny), weights, 0.0)

        # the idea: [y[0, inds[0]], y[1, inds[1]], ...]
        HTy = jnp.sum(y[jnp.arange(len(angles)).reshape(-1, 1, 1), inds] * weights_valid, axis=0)

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


class XRayTransform3D(LinearOperator):
    r"""General-purpose, 3D, parallel ray X-ray projector.

    This projector approximates cubic voxels projecting onto
    rectangular pixels and provides a back projector that is the exact
    adjoint of the forward projector. It is written purely in JAX,
    allowing it to run on either CPU or GPU and minimizing host copies.

    Warning: This class is experimental and may be up to ten times slower
    than :class:`scico.linop.xray.astra.XRayTransform3D`.

    For each view, the projection geometry is specified by an array
    with shape (2, 4) that specifies a :math:`2 \times 3` projection
    matrix and a :math:`2 \times 1` offset vector. Denoting the matrix
    by :math:`\mathbf{M}` and the offset by :math:`\mathbf{t}`, a voxel at array
    index `(i, j, k)` has its center projected to the detector coordinates

    .. math::
        \mathbf{M} \begin{bmatrix}
        i + \frac{1}{2} \\ j + \frac{1}{2} \\ k + \frac{1}{2}
        \end{bmatrix} + \mathbf{t} \,.

    The detector pixel at index `(i, j)` covers detector coordinates
    :math:`[i+1) \times [j+1)`.

    :meth:`XRayTransform3D.matrices_from_euler_angles` can help to
    make these geometry arrays.
    """

    def __init__(
        self,
        input_shape: Shape,
        matrices: ArrayLike,
        det_shape: Shape,
    ):
        r"""
        Args:
            input_shape: Shape of input image.
            matrices: (num_views, 2, 4) array of homogeneous projection matrices.
            det_shape: Shape of detector.
        """

        self.input_shape: Shape = input_shape
        self.matrices = jnp.asarray(matrices, dtype=np.float32)
        self.det_shape = det_shape
        self.output_shape = (len(matrices), *det_shape)
        super().__init__(
            input_shape=input_shape,
            output_shape=self.output_shape,
            eval_fn=self.project,
            adj_fn=self.back_project,
        )

    def project(self, im: ArrayLike) -> snp.Array:
        """Compute X-ray projection."""
        return XRayTransform3D._project(im, self.matrices, self.det_shape)

    def back_project(self, proj: ArrayLike) -> snp.Array:
        """Compute X-ray back projection"""
        return XRayTransform3D._back_project(proj, self.matrices, self.input_shape)

    @staticmethod
    def _project(im: ArrayLike, matrices: ArrayLike, det_shape: Shape) -> snp.Array:
        r"""
        Args:
            im: Input image.
            matrix: (num_views, 2, 4) array of homogeneous projection matrices.
            det_shape: Shape of detector.
        """
        MAX_SLICE_LEN = 10
        slice_offsets = list(range(0, im.shape[0], MAX_SLICE_LEN))

        num_views = len(matrices)
        proj = jnp.zeros((num_views,) + det_shape, dtype=im.dtype)
        for view_ind, matrix in enumerate(matrices):
            for slice_offset in slice_offsets:
                proj = proj.at[view_ind].set(
                    XRayTransform3D._project_single(
                        im[slice_offset : slice_offset + MAX_SLICE_LEN],
                        matrix,
                        proj[view_ind],
                        slice_offset=slice_offset,
                    )
                )
        return proj

    @staticmethod
    @partial(jax.jit, donate_argnames="proj")
    def _project_single(
        im: ArrayLike, matrix: ArrayLike, proj: ArrayLike, slice_offset: int = 0
    ) -> snp.Array:
        r"""
        Args:
            im: Input image.
            matrix: (2, 4) homogeneous projection matrix.
            det_shape: Shape of detector.
        """

        ul_ind, ul_weight, ur_weight, ll_weight, lr_weight = XRayTransform3D._calc_weights(
            im.shape, matrix, proj.shape, slice_offset
        )
        proj = proj.at[ul_ind[0], ul_ind[1]].add(ul_weight * im, mode="drop")
        proj = proj.at[ul_ind[0] + 1, ul_ind[1]].add(ur_weight * im, mode="drop")
        proj = proj.at[ul_ind[0], ul_ind[1] + 1].add(ll_weight * im, mode="drop")
        proj = proj.at[ul_ind[0] + 1, ul_ind[1] + 1].add(lr_weight * im, mode="drop")
        return proj

    @staticmethod
    def _back_project(proj: ArrayLike, matrices: ArrayLike, input_shape: Shape) -> snp.Array:
        r"""
        Args:
            proj: Input (set of) projection(s).
            matrix: (num_views, 2, 4) array of homogeneous projection matrices.
            input_shape: Shape of desired back projection.
        """
        MAX_SLICE_LEN = 10
        slice_offsets = list(range(0, input_shape[0], MAX_SLICE_LEN))

        HTy = jnp.zeros(input_shape, dtype=proj.dtype)
        for view_ind, matrix in enumerate(matrices):
            for slice_offset in slice_offsets:
                HTy = HTy.at[slice_offset : slice_offset + MAX_SLICE_LEN].set(
                    XRayTransform3D._back_project_single(
                        proj[view_ind],
                        matrix,
                        HTy[slice_offset : slice_offset + MAX_SLICE_LEN],
                        slice_offset=slice_offset,
                    )
                )
                HTy.block_until_ready()  # prevent OOM

        return HTy

    @staticmethod
    @partial(jax.jit, donate_argnames="HTy")
    def _back_project_single(
        y: ArrayLike, matrix: ArrayLike, HTy: ArrayLike, slice_offset: int = 0
    ) -> snp.Array:
        ul_ind, ul_weight, ur_weight, ll_weight, lr_weight = XRayTransform3D._calc_weights(
            HTy.shape, matrix, y.shape, slice_offset
        )
        HTy = HTy + y[ul_ind[0], ul_ind[1]] * ul_weight
        HTy = HTy + y[ul_ind[0] + 1, ul_ind[1]] * ur_weight
        HTy = HTy + y[ul_ind[0], ul_ind[1] + 1] * ll_weight
        HTy = HTy + y[ul_ind[0] + 1, ul_ind[1] + 1] * lr_weight
        return HTy

    @staticmethod
    def _calc_weights(
        input_shape: Shape, matrix: snp.Array, det_shape: Shape, slice_offset: int = 0
    ) -> snp.Array:
        # pixel (0, 0, 0) has its center at (0.5, 0.5, 0.5)
        x = jnp.mgrid[: input_shape[0], : input_shape[1], : input_shape[2]] + 0.5  # (3, ...)
        x = x.at[0].add(slice_offset)

        Px = jnp.stack(
            (
                matrix[0, 0] * x[0] + matrix[0, 1] * x[1] + matrix[0, 2] * x[2] + matrix[0, 3],
                matrix[1, 0] * x[0] + matrix[1, 1] * x[1] + matrix[1, 2] * x[2] + matrix[1, 3],
            )
        )  # (2, ...)

        # calculate weight on 4 intersecting pixels
        w = 0.5  # assumed <= 1.0
        left_edge = Px - w / 2
        to_next = jnp.minimum(jnp.ceil(left_edge) - left_edge, w)
        ul_ind = jnp.floor(left_edge).astype("int32")

        ul_weight = to_next[0] * to_next[1] * (1 / w**2)
        ur_weight = (w - to_next[0]) * to_next[1] * (1 / w**2)
        ll_weight = to_next[0] * (w - to_next[1]) * (1 / w**2)
        lr_weight = (w - to_next[0]) * (w - to_next[1]) * (1 / w**2)

        # set weights to zero out of bounds
        ul_weight = jnp.where(
            (ul_ind[0] >= 0)
            * (ul_ind[0] < det_shape[0])
            * (ul_ind[1] >= 0)
            * (ul_ind[1] < det_shape[1]),
            ul_weight,
            0.0,
        )
        ur_weight = jnp.where(
            (ul_ind[0] + 1 >= 0)
            * (ul_ind[0] + 1 < det_shape[0])
            * (ul_ind[1] >= 0)
            * (ul_ind[1] < det_shape[1]),
            ur_weight,
            0.0,
        )
        ll_weight = jnp.where(
            (ul_ind[0] >= 0)
            * (ul_ind[0] < det_shape[0])
            * (ul_ind[1] + 1 >= 0)
            * (ul_ind[1] + 1 < det_shape[1]),
            ll_weight,
            0.0,
        )
        lr_weight = jnp.where(
            (ul_ind[0] + 1 >= 0)
            * (ul_ind[0] + 1 < det_shape[0])
            * (ul_ind[1] + 1 >= 0)
            * (ul_ind[1] + 1 < det_shape[1]),
            lr_weight,
            0.0,
        )

        return ul_ind, ul_weight, ur_weight, ll_weight, lr_weight

    @staticmethod
    def matrices_from_euler_angles(
        input_shape: Shape,
        output_shape: Shape,
        seq: str,
        angles: ArrayLike,
        degrees: bool = False,
        voxel_spacing: ArrayLike = None,
        det_spacing: ArrayLike = None,
    ) -> snp.Array:
        """
        Create a set of projection matrices from Euler angles. The
        input voxels will undergo the specified rotation and then be
        projected onto the global xy-plane.

        Args:
            input_shape: Shape of input image.
            output_shape: Shape of output (detector).
            str: Sequence of axes for rotation. Up to 3 characters belonging to the set {'X', 'Y', 'Z'}
                for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and
                intrinsic rotations cannot be mixed in one function call.
            angles: (num_views, N), N = 1, 2, or 3 Euler angles.
            degrees: If ``True``, angles are in degrees, otherwise radians. Default: ``True``, radians.
            voxel_spacing: (3,) array giving the spacing of image
                voxels.  Default: `[1.0, 1.0, 1.0]`. Experimental.
            det_spacing: (2,) array giving the spacing of detector
                pixels.  Default: `[1.0, 1.0]`. Experimental.


        Returns:
            (num_views, 2, 4) array of homogeneous projection matrices.
        """

        if voxel_spacing is None:
            voxel_spacing = np.ones(3)

        if det_spacing is None:
            det_spacing = np.ones(2)

        # make projection matrix: form a rotation matrix and chop off the last row
        matrices = Rotation.from_euler(seq, angles, degrees=degrees).as_matrix()
        matrices = matrices[:, :2, :]  # (num_views, 2, 3)

        # handle scaling
        M_voxel = np.diag(voxel_spacing)  # (3, 3)
        M_det = np.diag(1 / np.array(det_spacing))  # (2, 2)

        # idea: M_det * M * M_voxel, but with a leading batch dimension
        matrices = np.einsum("vmn,nn->vmn", matrices, M_voxel)
        matrices = np.einsum("mm,vmn->vmn", M_det, matrices)

        # add translation to line up the centers
        x0 = np.array(input_shape) / 2
        t = -np.einsum("vmn,n->vm", matrices, x0) + np.array(output_shape) / 2
        matrices = snp.concatenate((matrices, t[..., np.newaxis]), axis=2)

        return matrices
