# -*- coding: utf-8 -*-
# Copyright (C) 2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r"""X-ray cone beam transform LinearOperators wrapping the ASTRA toolbox.

X-ray cone beam transform :class:`.LinearOperator` wrapping the cone beam
projections in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
This package provides CUDA implementations of core functionality for cone
beam geometries.

The cone beam geometry uses the same coordinate system conventions as the
parallel beam geometry. The volume is fixed with respect to the coordinate
system, centered at the origin. Geometry axes `z`, `y`, and `x` correspond
to volume array axes 0, 1, and 2 respectively. The projected array axes 0,
1, and 2 correspond respectively to detector rows, views, and detector
columns.

In the "cone" case, the source and detector rotate clockwise about the `z`
axis in the `x`-`y` plane. The source-origin distance (SOD) and
source-detector distance (SDD) define the cone beam geometry.

In the "cone_vec" case, each view is determined by the following vectors:

.. list-table:: View definition vectors
   :widths: 10 90

   * - :math:`\mb{s}`
     - Position of the source
   * - :math:`\mb{d}`
     - Center of the detector
   * - :math:`\mb{u}`
     - Vector from detector pixel (0,0) to (0,1) (direction of
       increasing detector column index)
   * - :math:`\mb{v}`
     - Vector from detector pixel (0,0) to (1,0) (direction of
       increasing detector row index)

These vectors are concatenated into a single row vector
:math:`(\mb{s}, \mb{d}, \mb{u}, \mb{v})` to form the full
geometry specification for a single view.
"""

from typing import Optional, Tuple

import numpy as np

import jax

try:
    import astra
except ModuleNotFoundError as e:
    if e.name == "astra":
        new_e = ModuleNotFoundError("Could not import astra; please install the ASTRA toolbox.")
        new_e.name = "astra"
        raise new_e from e
    else:
        raise e

from scico.linop import LinearOperator
from scico.typing import Shape


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


class XRayTransform3DCone(LinearOperator):  # pragma: no cover
    r"""3D cone beam X-ray transform based on the ASTRA toolbox.

    Perform cone beam tomographic projection (also called X-ray projection)
    of a volume at specified angles, using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
    The `3D geometries <https://astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
    "cone" and "cone_vec" are supported by this interface.
    Note that a CUDA GPU is required for the functionality of this class;
    if no GPU is available, initialization will fail with a :exc:`RuntimeError`.

    The volume is fixed with respect to the coordinate system, centered
    at the origin. The voxel sides have unit length (in arbitrary units),
    which defines the scale for all other dimensions in the
    source-volume-detector configuration. Geometry axes `z`, `y`, and `x`
    correspond to volume array axes 0, 1, and 2 respectively. The projected
    array axes 0, 1, and 2 correspond respectively to detector rows, views,
    and detector columns.

    In the "cone" case, the source and detector rotate clockwise about the
    `z` axis in the `x`-`y` plane. The source is positioned at distance
    `source_dist` from the origin, and the detector is at distance
    `det_dist` from the origin (making the source-detector distance
    `source_dist + det_dist`).

    In the "cone_vec" case, each view is determined by vectors specifying:
    the source position :math:`\mb{s}`, detector center :math:`\mb{d}`,
    and detector basis vectors :math:`\mb{u}` and :math:`\mb{v}`.
    """

    def __init__(
        self,
        input_shape: Shape,
        det_count: Tuple[int, int],
        det_spacing: Optional[Tuple[float, float]] = None,
        det_offset: Optional[Tuple[float, float]] = None,
        angles: Optional[np.ndarray] = None,
        source_dist: Optional[float] = None,
        det_dist: Optional[float] = None,
        vectors: Optional[np.ndarray] = None,
    ):
        """
        .. _astra-proj-geom3: https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries

        Keyword arguments `det_spacing`, `angles`, `source_dist`, and
        `det_dist` should be specified to use the "cone" geometry, and
        keyword argument `vectors` should be specified to use the "cone_vec"
        geometry. These parameters are mutually exclusive.

        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements (rows, columns) in the
               `projection geometry <astra-proj-geom3_>`__.
            det_spacing: Spacing between detector elements (row spacing,
               column spacing) in the `projection geometry <astra-proj-geom3_>`__.
            det_offset: Offset of the detector center as a tuple
               (horizontal shift, vertical shift). Negative/positive
               values correspond to left/right and up/down detector
               shifts (i.e. right/left and down/up shifts of the
               projection within the image) respectively.
            angles: Array of projection angles in radians. This parameter
                is mutually exclusive with `vectors`.
            source_dist: Distance from the source to the origin (center
                of rotation). Required when using `angles`.
            det_dist: Distance from the origin to the detector. Required
                when using `angles`.
            vectors: Array of ASTRA geometry specification vectors. This
                parameter is mutually exclusive with `angles`. Each row
                should contain 12 values: source position (3), detector
                center (3), u vector (3), v vector (3).

        Raises:
            RuntimeError: If a CUDA GPU is not available to the ASTRA
                toolbox.
            ValueError: If invalid parameter combinations are provided.
        """
        if not astra.use_cuda():
            raise RuntimeError("CUDA GPU required but not available or not enabled.")

        # Validate parameter combinations
        if not (
            (
                det_spacing is not None
                and angles is not None
                and source_dist is not None
                and det_dist is not None
                and vectors is None
            )
            or (
                vectors is not None
                and det_spacing is None
                and angles is None
                and source_dist is None
                and det_dist is None
            )
        ):
            raise ValueError(
                "Either keyword arguments 'det_spacing', 'angles', 'source_dist', "
                "and 'det_dist', or keyword argument 'vectors' must be specified, "
                "but not both sets."
            )

        self.num_dims = len(input_shape)
        if self.num_dims != 3:
            raise ValueError(
                f"Only 3D projections are supported, but 'input_shape' is {input_shape}."
            )

        if not isinstance(det_count, (list, tuple)) or len(det_count) != 2:
            raise ValueError("Expected argument 'det_count' to be a tuple with 2 elements.")

        # Determine number of views and store geometry parameters
        if angles is not None:
            Nview = angles.size
            self.angles: Optional[np.ndarray] = np.array(angles)
            self.vectors: Optional[np.ndarray] = None
            self.source_dist = source_dist
            self.det_dist = det_dist
        else:
            assert vectors is not None
            Nview = vectors.shape[0]
            self.vectors = np.array(vectors)
            self.angles = None
            self.source_dist = None
            self.det_dist = None

        output_shape: Shape = (det_count[0], Nview, det_count[1])

        self.det_count = det_count
        self.det_offset = det_offset
        self.input_shape: tuple = input_shape

        # Create ASTRA geometry objects
        self.vol_geom, self.proj_geom = self.create_astra_geometry(
            input_shape,
            det_count,
            det_spacing=det_spacing,
            angles=self.angles,
            source_dist=self.source_dist,
            det_dist=self.det_dist,
            vectors=self.vectors,
        )

        # Apply detector offset if specified
        if det_offset is not None:
            self.proj_geom = astra.functions.geom_postalignment(self.proj_geom, det_offset)

        # Wrap our non-jax functions with custom_vjp for gradient support
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
        source_dist: Optional[float] = None,
        det_dist: Optional[float] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> Tuple[dict, dict]:
        """Create ASTRA 3D cone beam geometry objects.

        Keyword arguments `det_spacing`, `angles`, `source_dist`, and
        `det_dist` should be specified to use the "cone" geometry, and
        keyword argument `vectors` should be specified to use the "cone_vec"
        geometry. These parameters are mutually exclusive.

        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements (rows, columns) in the
               `projection geometry <astra-proj-geom3_>`__.
            det_spacing: Spacing between detector elements (row spacing,
               column spacing) in the `projection geometry <astra-proj-geom3_>`__.
            angles: Array of projection angles in radians.
            source_dist: Distance from the source to the origin.
            det_dist: Distance from the origin to the detector.
            vectors: Array of geometry specification vectors.

        Returns:
            A tuple `(vol_geom, proj_geom)` of ASTRA volume geometry and
            projection geometry objects.
        """
        # Create volume geometry (same for both cone types)
        vol_geom = astra.create_vol_geom(input_shape[1], input_shape[2], input_shape[0])

        # Create projection geometry based on parameters
        if angles is not None:
            assert det_spacing is not None
            assert source_dist is not None
            assert det_dist is not None
            proj_geom = astra.create_proj_geom(
                "cone",
                det_spacing[0],
                det_spacing[1],
                det_count[0],
                det_count[1],
                angles,
                source_dist,
                det_dist,
            )
        else:
            assert vectors is not None
            proj_geom = astra.create_proj_geom("cone_vec", det_count[0], det_count[1], vectors)

        return vol_geom, proj_geom

    def _proj(self, x: jax.Array) -> jax.Array:
        """Apply the forward projector and generate a sinogram."""

        def f(x):
            x = _ensure_writeable(x)
            proj_id, result = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.output_shape, self.output_dtype), x)

    def _bproj(self, y: jax.Array) -> jax.Array:
        """Apply backprojector."""

        def f(y):
            y = _ensure_writeable(y)
            proj_id, result = astra.create_backprojection3d_gpu(y, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), y)

    def fdk(self, sino: jax.Array, **kwargs) -> jax.Array:
        """Feldkamp-Davis-Kress (FDK) reconstruction.

        Perform tomographic reconstruction using the Feldkamp-Davis-Kress
        (FDK)algorithm.

        Args:
            sino: Sinogram to reconstruct.
            **kwargs: Specify algorithm options, described in in the
               `ASTRA documentation
               <https://www.astra-toolbox.com/docs/algs/FDK_CUDA.html>`__.

        Returns:
            Reconstructed volume.
        """

        def f(sino):
            sino = _ensure_writeable(sino)
            sino_id = astra.data3d.create("-sino", self.proj_geom, sino)

            # create memory for result
            rec_id = astra.data3d.create("-vol", self.vol_geom)

            # start to populate config
            cfg = astra.astra_dict("FDK_CUDA")
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sino_id
            cfg["option"] = kwargs

            # initialize algorithm; run
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            # get the result
            out = astra.data3d.get(rec_id)

            # cleanup FDK-specific array
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(rec_id)
            astra.data3d.delete(sino_id)
            return out

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), sino)


def angle_to_vector_cone(
    det_spacing: Tuple[float, float],
    angles: np.ndarray,
    source_dist: float,
    det_dist: float,
) -> np.ndarray:
    """Convert cone beam parameters to vector geometry specification.

    Convert detector spacing, angles, and distances to ASTRA "cone_vec"
    geometry specification vectors.

    Args:
        det_spacing: Spacing between detector elements (row spacing,
            column spacing). See the
            `astra documentation <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__
            for more information.
        angles: Array of projection angles in radians.
        source_dist: Distance from the source to the origin.
        det_dist: Distance from the origin to the detector.

    Returns:
        Array of geometry specification vectors with shape (num_angles, 12).
        Each row contains: source position (3), detector center (3),
        u vector (3), v vector (3).
    """
    vectors = np.zeros((angles.size, 12))

    # Source position (rotates around origin at distance source_dist)
    vectors[:, 0] = np.sin(angles) * source_dist  # x component
    vectors[:, 1] = -np.cos(angles) * source_dist  # y component
    vectors[:, 2] = 0  # z component

    # Detector center (rotates around origin at distance det_dist, opposite side)
    vectors[:, 3] = -np.sin(angles) * det_dist  # x component
    vectors[:, 4] = np.cos(angles) * det_dist  # y component
    vectors[:, 5] = 0  # z component

    # u vector (detector column direction, horizontal)
    vectors[:, 6] = np.cos(angles) * det_spacing[1]  # x component
    vectors[:, 7] = np.sin(angles) * det_spacing[1]  # y component
    vectors[:, 8] = 0  # z component

    # v vector (detector row direction, vertical)
    vectors[:, 9] = 0  # x component
    vectors[:, 10] = 0  # y component
    vectors[:, 11] = det_spacing[0]  # z component

    return vectors
