# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform LinearOperators wrapping the ASTRA toolbox.

X-ray transform :class:`.LinearOperator` wrapping the parallel beam
projections in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
This package provides both C and CUDA implementations of core
functionality, but note that use of the CUDA/GPU implementation is
expected to result in GPU-host-GPU memory copies when transferring
JAX arrays. Other JAX features such as automatic differentiation are
not available.

Functions here refer to three coordinate systems: world coordinates,
volume coordinates, and detector coordinates. World coordinates are 3D
coordinates representing a point in physical space. Volume coordinates
refer to a position in the reconstruction volume, where the voxel with
its intensity value stored at `vol[i, j, k]` has its center at volume
coordinate (i+0.5, j+0.5, k+0.5) and side lengths of 1. Detector
coordinates refer to a position on the detector array, and the pixel at
`det[i, j]` has its center at detector coordinates (i+0.5, j+0.5) and
side lengths of one.

"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing

import jax
from jax.typing import ArrayLike

from scipy.spatial.transform import Rotation

try:
    import astra
except ModuleNotFoundError as e:
    if e.name == "astra":
        new_e = ModuleNotFoundError("Could not import astra; please install the ASTRA toolbox.")
        new_e.name = "astra"
        raise new_e from e
    else:
        raise e

try:
    from collections import Iterable  # type: ignore
except ImportError:
    import collections

    # Monkey patching required because latest astra release uses old module path for Iterable
    collections.Iterable = collections.abc.Iterable  # type: ignore

from scico.linop import LinearOperator
from scico.typing import Shape, TypeAlias

VolumeGeometry: TypeAlias = dict
ProjectionGeometry: TypeAlias = dict


def set_astra_gpu_index(idx: Union[int, Sequence[int]]):
    """Set the index/indices of GPU(s) to be used by astra.

    Args:
        idx: Index or indices of GPU(s).
    """
    astra.set_gpu_index(idx)


def _project_coords(
    x_volume: np.ndarray, vol_geom: VolumeGeometry, proj_geom: ProjectionGeometry
) -> np.ndarray:
    """
    Project volume coordinates into detector coordinates based on ASTRA
    geometry objects.

    Args:
        x_volume: (..., 3) vector(s) of volume coordinates.
        vol_geom: ASTRA volume geometry object.
        proj_geom: ASTRA projection geometry object.

    Returns:
        (num_angles, ..., 2) array of detector coordinates corresponding
        to projections of the points in `x_volume`.

    """
    det_shape = (proj_geom["DetectorRowCount"], proj_geom["DetectorColCount"])
    x_world = volume_coords_to_world_coords(x_volume, vol_geom=vol_geom)
    x_dets = []
    for vec in proj_geom["Vectors"]:
        ray, d, u, v = vec[0:3], vec[3:6], vec[6:9], vec[9:12]
        x_det = project_world_coordinates(x_world, ray, d, u, v, det_shape)
        x_dets.append(x_det)

    return np.stack(x_dets)


def project_world_coordinates(
    x: np.ndarray,
    ray: np.typing.ArrayLike,
    d: np.typing.ArrayLike,
    u: np.typing.ArrayLike,
    v: np.typing.ArrayLike,
    det_shape: Sequence[int],
) -> np.ndarray:
    """Project world coordinates along ray into the specified basis.

    Project world coordinates along `ray` into the basis described by `u`
    and `v` with center `d`.

    Args:
        x: (..., 3) vector(s) of world coordinates.
        ray: (3,) ray direction
        d: (3,) center of the detector
        u: (3,) vector from detector pixel (0,0) to (0,1), columns, x
        v: (3,) vector from detector pixel (0,0) to (1,0), rows, y

    Returns:
        (..., 2) vector(s) in the detector coordinates

    """
    Phi = np.stack((ray, u, v), axis=1)
    x = x - d  # express with respect to detector center
    alpha = np.linalg.pinv(Phi) @ x[..., :, np.newaxis]  # (3,3) times <stack of> (3,1)
    alpha = alpha[..., 0]  # squash from (..., 3, 1) to (..., 3)
    Palpha = alpha[..., 1:]  # throw away ray coordinate
    det_center_idx = (
        np.array(det_shape)[::-1] / 2 - 0.5
    )  # center of length-2 is index 0.5, length-3 -> index 1
    ind_xy = Palpha + det_center_idx
    ind_ij = ind_xy[..., ::-1]
    return ind_ij


def volume_coords_to_world_coords(idx: np.ndarray, vol_geom: VolumeGeometry) -> np.ndarray:
    """Convert a volume coordinate into a world coordinate.

    Convert a volume coordinate into a world coordinate using ASTRA
    conventions.

    Args:
        idx: (..., 2) or (..., 3) vector(s) of index coordinates.
        vol_geom: ASTRA volume geometry object.

    Returns:
        (..., 2) or (..., 3) vector(s) of world coordinates.

    """
    if "GridSliceCount" not in vol_geom:
        return _volume_index_to_astra_world_2d(idx, vol_geom)

    return _volume_index_to_astra_world_3d(idx, vol_geom)


def _volume_index_to_astra_world_2d(idx: np.ndarray, vol_geom: VolumeGeometry) -> np.ndarray:
    """Convert a 2D volume coordinate into a 2D world coordinate."""
    coord = idx[..., [1, 0]]  # x:col, y:row,
    nx = np.array(  # (x, y) order
        (
            vol_geom["GridColCount"],
            vol_geom["GridRowCount"],
        )
    )
    opt = vol_geom["option"]
    dx = np.array(
        (
            (opt["WindowMaxX"] - opt["WindowMinX"]) / nx[0],
            (opt["WindowMaxY"] - opt["WindowMinY"]) / nx[1],
        )
    )
    center_coord = nx / 2 - 0.5  # center of length-2 is index 0.5, center of length-3 is index 1
    return (coord - center_coord) * dx


def _volume_index_to_astra_world_3d(idx: np.ndarray, vol_geom: VolumeGeometry) -> np.ndarray:
    """Convert a 3D volume coordinate into a 3D world coordinate."""
    coord = idx[..., [2, 1, 0]]  # x:col, y:row, z:slice
    nx = np.array(  # (x, y, z) order
        (
            vol_geom["GridColCount"],
            vol_geom["GridRowCount"],
            vol_geom["GridSliceCount"],
        )
    )
    opt = vol_geom["option"]
    dx = np.array(
        (
            (opt["WindowMaxX"] - opt["WindowMinX"]) / nx[0],
            (opt["WindowMaxY"] - opt["WindowMinY"]) / nx[1],
            (opt["WindowMaxZ"] - opt["WindowMinZ"]) / nx[2],
        )
    )
    center_coord = nx / 2 - 0.5  # center of length-2 is index 0.5, center of length-3 is index 1
    return (coord - center_coord) * dx


class XRayTransform2D(LinearOperator):
    r"""2D parallel beam X-ray transform based on the ASTRA toolbox.

    Perform tomographic projection (also called X-ray projection) of an
    image at specified angles, using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
    """

    def __init__(
        self,
        input_shape: Shape,
        det_count: int,
        det_spacing: float,
        angles: np.ndarray,
        volume_geometry: Optional[List[float]] = None,
        device: str = "auto",
    ):
        """
        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom2d.html#projection-geometries>`__
               for more information.
            det_spacing: Spacing between detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom2d.html#projection-geometries>`__
               for more information..
            angles: Array of projection angles in radians.
            volume_geometry: Specification of the shape of the
               discretized reconstruction volume. Must either ``None``,
               in which case it is inferred from `input_shape`, or
               follow the syntax described in the
               `astra documentation <https://www.astra-toolbox.com/docs/geom2d.html#volume-geometries>`__.
            device: Specifies device for projection operation.
               One of ["auto", "gpu", "cpu"]. If "auto", a GPU is used if
               available, otherwise, the CPU is used.
        """

        self.num_dims = len(input_shape)
        if self.num_dims != 2:
            raise ValueError(
                f"Only 2D projections are supported, but 'input_shape' is {input_shape}."
            )
        if not isinstance(det_count, int):
            raise ValueError("Expected argument 'det_count' to be an int.")
        output_shape: Shape = (len(angles), det_count)

        # Set up all the ASTRA config
        self.det_spacing = det_spacing
        self.det_count = det_count
        self.angles: np.ndarray = np.array(angles)

        self.proj_geom: dict = astra.create_proj_geom(
            "parallel", det_spacing, det_count, self.angles
        )

        self.proj_id: Optional[int]
        self.input_shape: tuple = input_shape

        if volume_geometry is None:
            self.vol_geom = astra.create_vol_geom(*input_shape)
        else:
            if len(volume_geometry) == 4:
                self.vol_geom = astra.create_vol_geom(*input_shape, *volume_geometry)
            else:
                raise ValueError(
                    "Argument 'volume_geometry' must be a tuple of len 4."
                    "Please see the astra documentation for details."
                )

        if device in ["cpu", "gpu"]:
            # If cpu or gpu selected, attempt to comply (no checking to
            # confirm that a gpu is available to astra).
            self.device = device
        elif device == "auto":
            # If auto selected, use cpu or gpu depending on the default
            # jax device (for simplicity, no checking whether gpu is
            # available to astra when one is not available to jax).
            dev0 = jax.devices()[0]
            self.device = dev0.platform
        else:
            raise ValueError(f"Invalid 'device' specified; got {device}.")

        if self.device == "cpu":
            self.proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)
        elif self.device == "gpu":
            self.proj_id = astra.create_projector("cuda", self.proj_geom, self.vol_geom)

        # Wrap our non-jax function to indicate we will supply fwd/rev mode functions
        self._eval = jax.custom_vjp(self._proj)
        self._eval.defvjp(lambda x: (self._proj(x), None), lambda _, y: (self._bproj(y),))  # type: ignore
        self._adj = jax.custom_vjp(self._bproj)
        self._adj.defvjp(lambda y: (self._bproj(y), None), lambda _, x: (self._proj(x),))  # type: ignore

        super().__init__(
            input_shape=self.input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    def _proj(self, x: jax.Array) -> jax.Array:
        # apply the forward projector and generate a sinogram

        def f(x):
            x = _ensure_writeable(x)
            proj_id, result = astra.create_sino(x, self.proj_id)
            astra.data2d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.output_shape, self.output_dtype), x)

    def _bproj(self, y: jax.Array) -> jax.Array:
        # apply backprojector
        def f(y):
            y = _ensure_writeable(y)
            proj_id, result = astra.create_backprojection(y, self.proj_id)
            astra.data2d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), y)

    def fbp(self, sino: jax.Array, filter_type: str = "Ram-Lak") -> jax.Array:
        """Filtered back projection (FBP) reconstruction.

        Perform tomographic reconstruction using the filtered back
        projection (FBP) algorithm.

        Args:
            sino: Sinogram to reconstruct.
            filter_type: Select the filter to use. For a list of options
               see `cfg.FilterType` in the `ASTRA documentation
               <https://www.astra-toolbox.com/docs/algs/FBP_CUDA.html>`__.

        Returns:
            Reconstructed volume.
        """

        def f(sino):
            sino = _ensure_writeable(sino)
            sino_id = astra.data2d.create("-sino", self.proj_geom, sino)

            # create memory for result
            rec_id = astra.data2d.create("-vol", self.vol_geom)

            # start to populate config
            cfg = astra.astra_dict("FBP_CUDA" if self.device == "gpu" else "FBP")
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectorId"] = self.proj_id
            cfg["ProjectionDataId"] = sino_id
            cfg["option"] = {"FilterType": filter_type}

            # initialize algorithm; run
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            # get the result
            out = astra.data2d.get(rec_id)

            # cleanup FBP-specific arra
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sino_id)
            return out

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), sino)


def convert_from_scico_geometry(
    in_shape: Shape, matrices: ArrayLike, det_shape: Shape
) -> np.ndarray:
    """Convert SCICO projection matrices into ASTRA "parallel3d_vec" vectors.

    For 3D arrays,
    in ASTRA, the dimensions go (slices, rows, columns) and (z, y, x);
    in SCICO, the dimensions go (x, y, z).

    In ASTRA, the x-grid (recon) is centered on the origin and the y-grid (projection) can move.
    In SCICO, the x-grid origin is the center of x[0, 0, 0], the y-grid origin is the center
    of y[0, 0].

    See section "parallel3d_vec" in the
    `astra documentation <https://astra-toolbox.com/docs/geom3d.html#projection-geometries>`__.

    Args:
        in_shape: Shape of input image.
        matrices: (num_angles, 2, 4) array of homogeneous projection matrices.
        det_shape: Shape of detector.

    Returns:
        (num_angles, 12) vector array in the ASTRA "parallel3d_vec" convention.
    """
    # ray is perpendicular to projection axes
    ray = np.cross(matrices[:, 0, :3], matrices[:, 1, :3])
    # detector center comes from lifting the center index to 3D
    y_center = (np.array(det_shape) - 1) / 2
    x_center = (
        np.einsum("...mn,n->...m", matrices[..., :3], (np.array(in_shape) - 1) / 2)
        + matrices[..., 3]
    )
    d = np.einsum("...mn,...m->...n", matrices[..., :3], y_center - x_center)  # (V, 2, 3) x (V, 2)
    u = matrices[:, 1, :3]
    v = matrices[:, 0, :3]

    # handle different axis conventions
    ray = ray[:, [2, 1, 0]]
    d = d[:, [2, 1, 0]]
    u = u[:, [2, 1, 0]]
    v = v[:, [2, 1, 0]]

    vectors = np.concatenate((ray, d, u, v), axis=1)  # (v, 12)
    return vectors


def _astra_to_scico_geometry(vol_geom: VolumeGeometry, proj_geom: ProjectionGeometry) -> np.ndarray:
    """Convert ASTRA geometry objects into a SCICO projection matrix.

    Convert ASTRA volume and projection geometry into a SCICO X-ray
    projection matrix, assuming "parallel3d_vec" format.

    The approach is to locate 3 points in the volume domain,
    deduce the corresponding projection locations, and, then, solve a
    linear system to determine the affine relationship between them.

    Args:
        vol_geom: ASTRA volume geometry object.
        proj_geom: ASTRA projection geometry object.

    Returns:
        (num_angles, 2, 4) array of homogeneous projection matrices.

    """
    x_volume = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)  # (4, 3)
    x_dets = _project_coords(x_volume, vol_geom, proj_geom)  # (num_angles, 4, 2)

    x_volume_aug = np.concatenate((x_volume, np.ones((4, 1))), axis=1)  # (4, 4)
    matrices = []
    for x_det in x_dets:
        M = np.linalg.solve(x_volume_aug, x_det).T
        np.testing.assert_allclose(M @ x_volume_aug[0], x_det[0])
        matrices.append(M)

    return np.stack(matrices)


def convert_to_scico_geometry(
    input_shape: Shape,
    det_count: Tuple[int, int],
    det_spacing: Optional[Tuple[float, float]] = None,
    angles: Optional[np.ndarray] = None,
    vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert X-ray geometry specification to a SCICO projection matrix.

    The approach is to locate 3 points in the volume domain,
    deduce the corresponding projection locations, and, then, solve a
    linear system to determine the affine relationship between them.

    Args:
        input_shape: Shape of the input array.
        det_count: Number of detector elements. See the
           `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
           for more information.
        det_spacing: Spacing between detector elements. See the
           `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
           for more information.
        angles: Array of projection angles in radians. This parameter is
            mutually exclusive with `vectors`.
        vectors: Array of ASTRA geometry specification vectors. This
            parameter is mutually exclusive with `angles`.

    Returns:
        (num_angles, 2, 4) array of homogeneous projection matrices.

    """
    if angles is not None and vectors is not None:
        raise ValueError("Arguments 'angles' and 'vectors' are mutually exclusive.")
    if angles is None and vectors is None:
        raise ValueError("Exactly one of arguments 'angles' and 'vectors' must be provided.")
    vol_geom, proj_geom = XRayTransform3D.create_astra_geometry(
        input_shape, det_count, det_spacing=det_spacing, angles=angles, vectors=vectors
    )
    return _astra_to_scico_geometry(vol_geom, proj_geom)


class XRayTransform3D(LinearOperator):  # pragma: no cover
    r"""3D parallel beam X-ray transform based on the ASTRA toolbox.

    Perform tomographic projection (also called X-ray projection) of a
    volume at specified angles, using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
    The `3D geometries <https://astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
    "parallel3d" and "parallel3d_vec" are supported by this interface.
    Note that a CUDA GPU is required for the primary functionality of
    this class; if no GPU is available, initialization will fail with a
    :exc:`RuntimeError`.

    The volume is fixed with respect to the coordinate system, centered
    at the origin, as illustrated below:

    .. plot:: pyfigures/xray_3d_vol.py
       :align: center
       :include-source: False
       :show-source-link: False

    The voxels sides have unit length (in arbitrary units), which defines
    the scale for all other dimensions in the source-volume-detector
    configuration. Geometry axes `z`, `y`, and `x` correspond to volume
    array axes 0, 1, and 2 respectively. The projected array axes 0, 1,
    and 2 correspond respectively to detector rows, views, and detector
    columns.

    In the "parallel3d" case, the source and detector rotate clockwise
    about the `z` axis in the `x`-`y` plane, as illustrated below:

    .. plot:: pyfigures/xray_3d_ang.py
       :align: center
       :include-source: False
       :show-source-link: False
       :caption: Each radial arrow indicates the direction of the beam
          towards the detector (indicated in orange in the "light"
          display mode) and the arrow parallel to the detector indicates
          the direction of increasing pixel indices.

    In this case the `z` axis is in the same direction as the
    vertical/row axis of the detector and its projection corresponds to
    a vertical line in the center of the horizontal/column detector axis.
    Note that the view images must be displayed with the origin at the
    bottom left (i.e. vertically inverted from the top left origin image
    indexing convention) in order for the projections to correspond to
    the positive up/negative down orientation of the `z` axis in the
    figures here.

    In the "parallel3d_vec" case, each view is determined by the following
    vectors:

    .. list-table:: View definition vectors
       :widths: 10 90

       * - :math:`\mb{r}`
         - Direction of the parallel beam
       * - :math:`\mb{d}`
         - Center of the detector
       * - :math:`\mb{u}`
         - Vector from detector pixel (0,0) to (0,1) (direction of
           increasing detector column index)
       * - :math:`\mb{v}`
         - Vector from detector pixel (0,0) to (1,0) (direction of
           increasing detector row index)

    Note that the components of these vectors are in `x`, `y`, `z` order,
    not the `z`, `y`, `x` order of the volume axes.

    .. plot:: pyfigures/xray_3d_vec.py
       :align: center
       :include-source: False
       :show-source-link: False

    Vector :math:`\mb{r}` is not illustrated to avoid cluttering the
    figure, but will typically be directed toward the center of the
    detector (i.e. in the direction of :math:`\mb{d}` in the figure.)
    Since the volume-detector distance does not have a geometric effect
    for a parallel-beam configuration, :math:`\mb{d}` may be set to the
    zero vector when the detector and beam centers coincide (e.g., as in
    the case of the "parallel3d" geometry). Note that the view images
    must be displayed with the origin at the bottom left (i.e. vertically
    inverted from the top left origin image indexing convention) in order
    for the row indexing of the projections to correspond to the
    direction of :math:`\mb{v}` in the figure.

    These vectors are concatenated into a single row vector
    :math:`(\mb{r}, \mb{d}, \mb{u}, \mb{v})` to form the full
    geometry specification for a single view, and multiple such
    row vectors are stacked to specify the geometry for a set
    of views.
    """

    def __init__(
        self,
        input_shape: Shape,
        det_count: Tuple[int, int],
        det_spacing: Optional[Tuple[float, float]] = None,
        angles: Optional[np.ndarray] = None,
        vectors: Optional[np.ndarray] = None,
    ):
        """
        Keyword arguments `det_spacing` and `angles` should be specified
        to use the "parallel3d" geometry, and keyword argument `vectors`
        should be specified to use the "parallel3d_vec" geometry. These
        parameters are mutually exclusive.

        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
               for more information.
            det_spacing: Spacing between detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
               for more information.
            angles: Array of projection angles in radians. This
                parameter is  mutually exclusive with `vectors`.
            vectors: Array of ASTRA geometry specification vectors. This
                parameter is mutually exclusive with `angles`.

        Raises:
            RuntimeError: If a CUDA GPU is not available to the ASTRA
                toolbox.
        """
        if not astra.use_cuda():
            raise RuntimeError("CUDA GPU required but not available or not enabled.")

        if not (
            (det_spacing is not None and angles is not None and vectors is None)
            or (vectors is not None and det_spacing is None and angles is None)
        ):
            raise ValueError(
                "Keyword arguments 'det_spacing' and 'angles', or keyword argument "
                "'vectors' must be specified, but not both."
            )

        self.num_dims = len(input_shape)
        if self.num_dims != 3:
            raise ValueError(
                f"Only 3D projections are supported, but 'input_shape' is {input_shape}."
            )

        if not isinstance(det_count, (list, tuple)) or len(det_count) != 2:
            raise ValueError("Expected argument 'det_count' to be a tuple with 2 elements.")
        if angles is not None and vectors is not None:
            raise ValueError("Arguments 'angles' and 'vectors' are mutually exclusive.")
        if angles is None and vectors is None:
            raise ValueError(
                "Exactly one of the arguments 'angles' and 'vectors' must be provided."
            )
        if angles is not None:
            Nview = angles.size
            self.angles: Optional[np.ndarray] = np.array(angles)
            self.vectors: Optional[np.ndarray] = None
        if vectors is not None:
            Nview = vectors.shape[0]
            self.vectors = np.array(vectors)
            self.angles = None
        output_shape: Shape = (det_count[0], Nview, det_count[1])

        self.det_count = det_count
        assert isinstance(det_count, (list, tuple))
        self.input_shape: tuple = input_shape
        self.vol_geom, self.proj_geom = self.create_astra_geometry(
            input_shape,
            det_count,
            det_spacing=det_spacing,
            angles=self.angles,
            vectors=self.vectors,
        )

        # Wrap our non-jax function to indicate we will supply fwd/rev mode functions
        self._eval = jax.custom_vjp(self._proj)
        self._eval.defvjp(lambda x: (self._proj(x), None), lambda _, y: (self._bproj(y),))  # type: ignore
        self._adj = jax.custom_vjp(self._bproj)
        self._adj.defvjp(lambda y: (self._bproj(y), None), lambda _, x: (self._proj(x),))  # type: ignore

        super().__init__(
            input_shape=self.input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    @staticmethod
    def create_astra_geometry(
        input_shape: Shape,
        det_count: Tuple[int, int],
        det_spacing: Optional[Tuple[float, float]] = None,
        angles: Optional[np.ndarray] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> Tuple[VolumeGeometry, ProjectionGeometry]:
        """Create ASTRA 3D geometry objects.

        Keyword arguments `det_spacing` and `angles` should be specified
        to use the "parallel3d" geometry, and keyword argument `vectors`
        should be specified to use the "parallel3d_vec" geometry. These
        parameters are mutually exclusive.

        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
               for more information.
            det_spacing: Spacing between detector elements. See the
               `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
               for more information.
            angles: Array of projection angles in radians.
            vectors: Array of geometry specification vectors.

        Returns:
            A tuple `(vol_geom, proj_geom)` of ASTRA volume geometry and
            projection geometry objects.
        """
        vol_geom = astra.create_vol_geom(input_shape[1], input_shape[2], input_shape[0])
        if angles is not None:
            assert det_spacing is not None
            proj_geom = astra.create_proj_geom(
                "parallel3d",
                det_spacing[0],
                det_spacing[1],
                det_count[0],
                det_count[1],
                angles,
            )
        else:
            proj_geom = astra.create_proj_geom(
                "parallel3d_vec", det_count[0], det_count[1], vectors
            )
        return vol_geom, proj_geom

    def _proj(self, x: jax.Array) -> jax.Array:
        # apply the forward projector and generate a sinogram

        def f(x):
            x = _ensure_writeable(x)
            proj_id, result = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.output_shape, self.output_dtype), x)

    def _bproj(self, y: jax.Array) -> jax.Array:
        # apply backprojector
        def f(y):
            y = _ensure_writeable(y)
            proj_id, result = astra.create_backprojection3d_gpu(y, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), y)


def angle_to_vector(det_spacing: Tuple[float, float], angles: np.ndarray) -> np.ndarray:
    """Convert det_spacing and angles to vector geometry specification.

    Args:
        det_spacing: Spacing between detector elements. See the
            `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
            for more information.
        angles: Array of projection angles in radians.

    Returns:
        Array of geometry specification vectors.
    """
    vectors = np.zeros((angles.size, 12))
    vectors[:, 0] = np.sin(angles)
    vectors[:, 1] = -np.cos(angles)
    vectors[:, 6] = np.cos(angles) * det_spacing[0]
    vectors[:, 7] = np.sin(angles) * det_spacing[0]
    vectors[:, 11] = det_spacing[1]
    return vectors


def rotate_vectors(vectors: np.ndarray, rot: Rotation) -> np.ndarray:
    """Rotate geometry specification vectors.

    Rotate ASTRA "parallel3d_vec" geometry specification vectors.

    Args:
        vectors: Array of geometry specification vectors.
        rot: Rotation.

    Returns:
        Rotated geometry specification vectors.
    """
    rot_vecs = vectors.copy()
    for k in range(0, 12, 3):
        rot_vecs[:, k : k + 3] = rot.apply(rot_vecs[:, k : k + 3])
    return rot_vecs


def _ensure_writeable(x):
    """Ensure that `x.flags.writeable` is ``True``, copying if needed."""
    if hasattr(x, "flags"):  # x is a numpy array
        if not x.flags.writeable:
            try:
                x.setflags(write=True)
            except ValueError:
                x = x.copy()
    else:  # x is a jax array (which is immutable)
        x = np.array(x)
    return x
