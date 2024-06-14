# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
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
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import jax
from jax.typing import ArrayLike

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
from scico.typing import Shape


def set_astra_gpu_index(idx: Union[int, Sequence[int]]):
    """Set the index/indices of GPU(s) to be used by astra.

    Args:
        idx: Index or indices of GPU(s).
    """
    astra.set_gpu_index(idx)


def convert_from_scico_geometry(
    in_shape: Shape, matrices: ArrayLike, det_shape: Shape
) -> ArrayLike:
    """
    Convert SCICO projection matrices into ASTRA "parallel3d_vec" vectors.

    For 3D arrays,
    in Astra, the dimensions go (slices, rows, columns) and (z, y, x);
    in SCICO, the dimensions go (x, y, z).

    In Astra, the x-grid (recon) is centered on the origin and the y-grid (projection) can move.
    In SCICO, the x-grid origin is the center of x[0, 0, 0], the y-grid origin is the center
    of y[0, 0].

    See https://astra-toolbox.com/docs/geom3d.html#projection-geometries parallel3d_vec.

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
    y_center = np.array(det_shape) / 2
    x_center = (
        np.einsum("...mn,n->...m", matrices[..., :3], np.array(in_shape) / 2) + matrices[..., 3]
    )
    d = np.einsum("...mn,...m->...n", matrices[..., :3], y_center - x_center)  # (V, 2, 3) x (V, 2)
    u = -matrices[:, 1, :3]
    v = -matrices[:, 0, :3]
    vectors = np.concatenate((ray, d, u, v), axis=1)  # (v, 12)
    return vectors


def convert_to_scico_geometry(vol_geom, proj_geom):
    """
    Convert ASTRA volume and projection geometry into a SCICO X-ray projection matrix, assuming
    "parallel3d_vec" format.

    Args:
        vol_geom: ASTRA volume geometry object.
        proj_geom: ASTRA projection geometry object.

    Returns:
        (num_angles, 2, 4) array of homogeneous projection matrices.

    """
    in_shape = (vol_geom["GridSliceCount"], vol_geom["GridRowCount"], vol_geom["GridColCount"])
    det_shape = (proj_geom["DetectorRowCount"], proj_geom["DetectorColCount"])
    vectors = proj_geom["Vectors"]
    _, d, u, v = vectors[:, 0:3], vectors[:, 3:6], vectors[:, 6:9], vectors[:, 9:12]
    matrices = -np.stack((v, u), axis=1)
    center_diff = np.einsum("...mn,...n->...m", matrices, d)  # y_center - x_center
    y_center = np.array(det_shape) / 2
    Px_center_t = -(center_diff - y_center)
    Px_center = np.einsum("...mn,n->...m", matrices, np.array(in_shape) / 2)
    t = Px_center_t - Px_center
    matrices = np.concatenate((matrices, t[..., np.newaxis]), axis=2)

    return matrices


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
                f"Only 2D projections are supported, but input_shape is {input_shape}."
            )
        if not isinstance(det_count, int):
            raise ValueError("Expected det_count to be an int.")
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
                    "volume_geometry must be a tuple of len 4."
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
            raise ValueError(f"Invalid device specified; got {device}.")

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


class XRayTransform3D(LinearOperator):  # pragma: no cover
    r"""3D parallel beam X-ray transform based on the ASTRA toolbox.

    Perform tomographic projection (also called X-ray projection) of a
    volume at specified angles, using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
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
        This class supports both "parallel3d" and "parallel3d_vec" astra
        `projection geometries <https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries>`__.
        Keyword arguments `det_spacing` and `angles` should be specified
        to use the former, and keyword argument `vectors` should be
        specified to use the latter. These options are mutually exclusive.

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
        """
        if not (
            (det_spacing is not None and angles is not None and vectors is None)
            or (vectors is not None and det_spacing is None and angles is None)
        ):
            raise ValueError(
                "Keyword arguments det_spacing and angles, or keyword argument "
                "vectors must be specified, but not both."
            )

        self.num_dims = len(input_shape)
        if self.num_dims != 3:
            raise ValueError(
                f"Only 3D projections are supported, but input_shape is {input_shape}."
            )

        if not isinstance(det_count, (list, tuple)) or len(det_count) != 2:
            raise ValueError("Expected det_count to be a tuple with 2 elements.")
        if angles is not None:
            Nview = angles.size
            self.angles: np.ndarray = np.array(angles)
        else:
            assert vectors is not None
            Nview = vectors.shape[0]
            self.vectors: np.ndarray = np.array(vectors)
        output_shape: Shape = (det_count[0], Nview, det_count[1])

        self.det_count = det_count
        assert isinstance(det_count, (list, tuple))
        if angles is not None:
            assert det_spacing is not None
            self.proj_geom = astra.create_proj_geom(
                "parallel3d",
                det_spacing[0],
                det_spacing[1],
                det_count[0],
                det_count[1],
                self.angles,
            )
        else:
            self.proj_geom = astra.create_proj_geom(
                "parallel3d_vec", det_count[0], det_count[1], self.vectors
            )

        self.input_shape: tuple = input_shape
        self.vol_geom = astra.create_vol_geom(input_shape[1], input_shape[2], input_shape[0])

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
