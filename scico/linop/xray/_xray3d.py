# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""3D X-ray transform classes."""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax.typing import ArrayLike

import scico.numpy as snp
from scico.typing import DType, Shape
from scipy.spatial.transform import Rotation

from .._linop import LinearOperator


class XRayTransform3D(LinearOperator):
    r"""General-purpose, 3D, parallel ray X-ray projector.

    This projector approximates cubic voxels projecting onto
    rectangular pixels and provides a back projector that is the exact
    adjoint of the forward projector. It is written purely in JAX,
    allowing it to run on either CPU or GPU and minimizing host copies.

    For each view, the projection geometry is specified by an array
    with shape (2, 4) that specifies a :math:`2 \times 3` projection
    matrix and a :math:`2 \times 1` offset vector. Denoting the matrix
    by :math:`\mathbf{M}` and the offset by :math:`\mathbf{t}`, a voxel
    at array index `(i, j, k)` has its center projected to the detector
    coordinates

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
        batch_size: int = 8,
        input_dtype: DType = np.float32,
        input_device: xc.Device | Sharding | None = None,
        output_device: xc.Device | Sharding | None = None,
    ):
        r"""
        Args:
            input_shape: Input array shape.
            matrices: (num_views, 2, 4) array of homogeneous projection
               matrices.
            det_shape: Shape of detector.
            batch_size: Number of projections to compute in parallel.
                Higher is faster but more memory intensive.
            input_dtype: Input array dtype.
            input_device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                for input arrays.
            output_device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                for output arrays.
        """

        self.input_shape: Shape = input_shape
        self.matrices = jnp.asarray(matrices, dtype=np.float32)
        self.det_shape = tuple(det_shape)  # in case det_shape is a list
        self.batch_size = batch_size
        self.output_shape = (len(matrices), *det_shape)
        self.input_device = input_device
        self.output_device = output_device
        super().__init__(
            input_shape=input_shape,
            output_shape=self.output_shape,
            eval_fn=self.project,
            adj_fn=self.back_project,
            input_dtype=input_dtype,
            output_dtype=input_dtype,
        )

    def project(self, im: ArrayLike) -> snp.Array:
        """Compute X-ray projection."""
        return XRayTransform3D._project(
            im, self.matrices, self.det_shape, batch_size=self.batch_size, device=self.output_device
        )

    def back_project(self, proj: ArrayLike) -> snp.Array:
        """Compute X-ray back projection"""
        return XRayTransform3D._back_project(
            proj,
            self.matrices,
            self.input_shape,
            device=self.input_device,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("det_shape", "batch_size"))
    def _project(
        im: ArrayLike,
        matrices: ArrayLike,
        det_shape: Shape,
        batch_size: int = 8,
        device: xc.Device | Sharding | None = None,
    ) -> snp.Array:
        r"""
        Args:
            im: Input image.
            matrix: (num_views, 2, 4) array of homogeneous projection
                matrices.
            det_shape: Shape of detector.
            batch_size: Number of projections to compute in parallel.
                Higher is faster but more memory intensive.
            device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                to which the output will be committed.
        """

        def project_single_matrix(matrix):
            init_proj = jnp.zeros(det_shape, dtype=im.dtype, device=device)
            return XRayTransform3D._project_single(
                im,
                matrix,
                init_proj,
            )

        return jax.lax.map(project_single_matrix, matrices, batch_size=batch_size)

    @staticmethod
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
    @partial(jax.jit, static_argnames="input_shape")
    def _back_project(
        proj: ArrayLike,
        matrices: ArrayLike,
        input_shape: Shape,
        device: xc.Device | Sharding | None = None,
    ) -> snp.Array:
        r"""
        Args:
            proj: Input projection data of shape (num_views, *det_shape).
            matrix: (num_views, 2, 4) array of homogeneous projection matrices.
            input_shape: Shape of back projection.
            device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
                to which the output will be committed.
        """

        init_volume = jnp.zeros(input_shape, dtype=proj.dtype, device=device)

        def scan_func(volume, proj_matrix_tuple):
            proj, matrix = proj_matrix_tuple
            volume = XRayTransform3D._back_project_single(
                proj,
                matrix,
                volume,
            )
            return volume, None

        volume, _ = jax.lax.scan(scan_func, init_volume, (proj, matrices))

        return volume

    @staticmethod
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
            str: Sequence of axes for rotation. Up to 3 characters
                belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations.
                Extrinsic and intrinsic rotations cannot be mixed in one
                function call.
            angles: (num_views, N), N = 1, 2, or 3 Euler angles.
            degrees: If ``True``, angles are in degrees, otherwise
                radians. Default: ``True``, radians.
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
