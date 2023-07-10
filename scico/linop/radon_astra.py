# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Radon transform LinearOperator wrapping the ASTRA toolbox.

Radon transform :class:`.LinearOperator` wrapping the parallel beam
projections in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
This package provides both C and CUDA implementations of core
functionality, but note that use of the CUDA/GPU implementation is
expected to result in GPU-host-GPU memory copies when transferring
JAX arrays. Other JAX features such as automatic differentiation are
not available.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

import jax
import jax.experimental.host_callback as hcb

try:
    import astra
except ModuleNotFoundError as e:
    if e.name == "astra":
        new_e = ModuleNotFoundError("Could not import astra; please install the ASTRA toolbox.")
        new_e.name = "astra"
        raise new_e from e
    else:
        raise e


from scico.typing import Shape

from ._linop import LinearOperator


class TomographicProjector(LinearOperator):
    r"""Parallel beam Radon transform based on the ASTRA toolbox.

    Perform tomographic projection (also called X-ray projection) of an
    image or volume at specified angles, using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
    """

    def __init__(
        self,
        input_shape: Shape,
        detector_spacing: Union[float, Tuple[float, float]],
        det_count: Union[int, Tuple[int, int]],
        angles: np.ndarray,
        volume_geometry: Optional[List[float]] = None,
        device: str = "auto",
    ):
        """
        Args:
            input_shape: Shape of the input array. Determines whether 2D
               or 3D algorithm is used.
            detector_spacing: Spacing between detector elements. See
               https://www.astra-toolbox.com/docs/geom2d.html#projection-geometries
               or
               https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries
               for more information.
            det_count: Number of detector elements. See
               https://www.astra-toolbox.com/docs/geom2d.html#projection-geometries
               or
               https://www.astra-toolbox.com/docs/geom3d.html#projection-geometries
               for more information.
            angles: Array of projection angles in radians.
            volume_geometry: Specification of the shape of the
               discretized reconstruction volume. Must either ``None``,
               in which case it is inferred from `input_shape`, or
               follow the astra syntax described in
               https://www.astra-toolbox.com/docs/geom2d.html#volume-geometries
               or
               https://www.astra-toolbox.com/docs/geom3d.html#d-geometries.
            device: Specifies device for projection operation.
               One of ["auto", "gpu", "cpu"]. If "auto", a GPU is used if
               available, otherwise, the CPU is used.
        """

        self.num_dims = len(input_shape)
        if self.num_dims not in [2, 3]:
            raise ValueError(
                f"Only 2D and 3D projections are supported, but `input_shape` is {input_shape}."
            )

        output_shape: Shape
        if self.num_dims == 2:
            output_shape = (len(angles), det_count)
        elif self.num_dims == 3:
            assert isinstance(det_count, (list, tuple))
            if len(det_count) != 2:
                raise ValueError("Expected `det_count` to have 2 elements")
            output_shape = (det_count[0], len(angles), det_count[1])

        # Set up all the ASTRA config
        self.detector_spacing = detector_spacing
        self.det_count = det_count
        self.angles: np.ndarray = np.array(angles)

        if self.num_dims == 2:
            self.proj_geom: dict = astra.create_proj_geom(
                "parallel", detector_spacing, det_count, self.angles
            )
        elif self.num_dims == 3:
            assert isinstance(detector_spacing, (list, tuple))
            assert isinstance(det_count, (list, tuple))
            if len(detector_spacing) != 2:
                raise ValueError("Expected `detector_spacing` to have 2 elements")
            self.proj_geom = astra.create_proj_geom(
                "parallel3d",
                detector_spacing[0],
                detector_spacing[1],
                det_count[0],
                det_count[1],
                self.angles,
            )

        self.proj_id: Optional[int]
        self.input_shape: tuple = input_shape

        if volume_geometry is not None:
            if (self.num_dims == 2 and len(volume_geometry) == 4) or (
                self.num_dims == 3 and len(volume_geometry) == 6
            ):
                self.vol_geom: dict = astra.create_vol_geom(*input_shape, *volume_geometry)
            else:
                raise ValueError(
                    "`volume_geometry` must be a tuple of len 4 (2D) or 6 (3D)."
                    "Please see the astra documentation for details."
                )
        else:
            if self.num_dims == 2:
                self.vol_geom = astra.create_vol_geom(*input_shape)
            elif self.num_dims == 3:
                self.vol_geom = astra.create_vol_geom(
                    input_shape[1], input_shape[2], input_shape[0]
                )

        dev0 = jax.devices()[0]
        if dev0.platform == "cpu" or device == "cpu":
            self.device = "cpu"
        elif dev0.platform == "gpu" and device in ["gpu", "auto"]:
            self.device = "gpu"
        else:
            raise ValueError(f"Invalid device specified; got {device}.")

        if self.num_dims == 3 and self.device == "cpu":
            raise ValueError("No CPU algorithm exists for 3D tomography.")

        if self.num_dims == 3:
            # not needed for astra's 3D algorithm
            self.proj_id = None
        elif self.num_dims == 2:
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
        # Applies the forward projector and generates a sinogram

        def f(x):
            if x.flags.writeable == False:
                x.flags.writeable = True
            if self.num_dims == 2:
                proj_id, result = astra.create_sino(x, self.proj_id)
                astra.data2d.delete(proj_id)
            elif self.num_dims == 3:
                proj_id, result = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)
                astra.data3d.delete(proj_id)
            return result

        return hcb.call(
            f, x, result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype)
        )

    def _bproj(self, y: jax.Array) -> jax.Array:
        # applies backprojector
        def f(y):
            if y.flags.writeable == False:
                y.flags.writeable = True
            if self.num_dims == 2:
                proj_id, result = astra.create_backprojection(y, self.proj_id)
                astra.data2d.delete(proj_id)
            elif self.num_dims == 3:
                proj_id, result = astra.create_backprojection3d_gpu(
                    y, self.proj_geom, self.vol_geom
                )
                astra.data3d.delete(proj_id)
            return result

        return hcb.call(f, y, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype))

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

        if self.num_dims == 3:
            raise NotImplementedError("3D FBP is not implemented")

        # Just use the CPU FBP alg for now; hitting memory issues with GPU one.
        def f(sino):
            if sino.flags.writeable == False:
                sino.flags.writeable = True
            sino_id = astra.data2d.create("-sino", self.proj_geom, sino)

            # create memory for result
            rec_id = astra.data2d.create("-vol", self.vol_geom)

            # start to populate config
            cfg = astra.astra_dict("FBP")
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

        return hcb.call(
            f, sino, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype)
        )
