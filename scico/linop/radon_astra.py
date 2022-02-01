# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Radon transform LinearOperator wrapping the ASTRA toolbox.

Radon transform LinearOperator wrapping the parallel beam projections in
the `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
"""


from typing import List, Optional

import numpy as np

import jax
import jax.experimental.host_callback as hcb

try:
    import astra
except ImportError:
    raise ImportError("Could not import astra; please install the ASTRA toolbox.")


from jaxlib.xla_extension import GpuDevice

from scico.typing import JaxArray, Shape

from ._linop import LinearOperator


class ParallelBeamProjector(LinearOperator):
    r"""Parallel beam Radon transform based on the ASTRA toolbox.

    Perform tomographic projection of an image at specified angles,
    using the
    `ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
    """

    def __init__(
        self,
        input_shape: Shape,
        detector_spacing: float,
        det_count: int,
        angles: np.ndarray,
        volume_geometry: Optional[List[float]] = None,
        device: str = "auto",
    ):
        """
        Args:
            input_shape: Shape of the input array.
            volume_geometry: Defines the shape and size of the
                discretized reconstruction volume. Must either `None`, or
                of the form (min_x, max_x, min_y, max_y). If `None`,
                volume pixels are squares with sides of unit length, and
                the volume is centered around the origin. If not None,
                the extents of the volume can be specified arbitrarily.
                The default, None, corresponds to
                ``volume_geometry = [cols, -cols/2, cols/2, -rows/2, rows/2]``.
                Note: For usage with GPU code, the volume must be
                centered around the origin and pixels must be square.
                This is not always explicitly checked in all functions,
                so not following these requirements may have
                unpredictable results. See `original ASTRA documentation
                <https://www.astra-toolbox.com/docs/geom2d.html#volume-geometries>`_.
            detector_spacing: Spacing between detector elements.
            det_count: Number of detector elements.
            angles: Array of projection angles.
            device: Specifies device for projection operation.
                One of ["auto", "gpu", "cpu"]. If "auto", a GPU is used
                if available. Otherwise, the CPU is used.
        """

        # Set up all the ASTRA config
        self.detector_spacing: float = detector_spacing
        self.det_count: int = det_count
        self.angles: np.ndarray = angles

        self.proj_geom: dict = astra.create_proj_geom(
            "parallel", detector_spacing, det_count, angles
        )
        self.proj_id: int
        self.input_shape: tuple = input_shape

        if volume_geometry is not None:
            if len(volume_geometry) == 4:
                self.vol_geom: dict = astra.create_vol_geom(*input_shape, *volume_geometry)
            else:
                raise AssertionError(
                    "Volume_geometry must be the shape of the volume as a tuple of len 4 "
                    "containing the volume geometry dimensions. Please see documentation "
                    "for specifics."
                )
        else:
            self.vol_geom: dict = astra.create_vol_geom(*input_shape)

        dev0 = jax.devices()[0]
        if dev0.device_kind == "cpu" or device == "cpu":
            self.proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)
        elif isinstance(dev0, GpuDevice) and device in ["gpu", "auto"]:
            self.proj_id = astra.create_projector("cuda", self.proj_geom, self.vol_geom)
        else:
            raise ValueError(f"Invalid device specified; got {device}")

        # Wrap our non-jax function to indicate we will supply fwd/rev mode functions
        self._eval = jax.custom_vjp(self._proj)
        self._eval.defvjp(lambda x: (self._proj(x), None), lambda _, y: (self._bproj(y),))
        self._adj = jax.custom_vjp(self._bproj)
        self._adj.defvjp(lambda y: (self._bproj(y), None), lambda _, x: (self._proj(x),))

        super().__init__(
            input_shape=self.input_shape,
            output_shape=(len(angles), det_count),
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    def _proj(self, x: JaxArray) -> JaxArray:
        # Applies the forward projector and generates a sinogram

        def f(x):
            if x.flags.writeable == False:
                x.flags.writeable = True
            proj_id, result = astra.create_sino(x, self.proj_id)
            astra.data2d.delete(proj_id)
            return result

        return hcb.call(
            f, x, result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype)
        )

    def _bproj(self, y: JaxArray) -> JaxArray:
        # applies backprojector
        def f(y):
            if y.flags.writeable == False:
                y.flags.writeable = True
            proj_id, result = astra.create_backprojection(y, self.proj_id)
            astra.data2d.delete(proj_id)
            return result

        return hcb.call(f, y, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype))

    def fbp(self, sino: JaxArray, filter_type: str = "Ram-Lak") -> JaxArray:
        """Perform tomographic reconstruction using the filtered back
        projection (FBP) algorithm.

        Args:
            sino: Sinogram to reconstruct.
            filter_type: Which filter to use, see `cfg.FilterType` in
               `<https://www.astra-toolbox.com/docs/algs/FBP_CUDA.html>`_.
        """

        # Just use the CPU FBP alg for now; hitting memory issues with GPU one.
        def f(sino):
            if sino.flags.writeable == False:
                sino.flags.writeable = True
            proj_id = astra.create_projector("line", self.proj_geom, self.vol_geom)
            sino_id = astra.data2d.create("-sino", self.proj_geom, sino)

            # create memory for result
            rec_id = astra.data2d.create("-vol", self.vol_geom)

            # start to populate config
            cfg = astra.astra_dict("FBP")
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectorId"] = proj_id
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
            return out

        return hcb.call(
            f, sino, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype)
        )
