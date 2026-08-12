# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""LinearOperators wrapping the ASTRA 2D parallel beam X-ray transform.

X-ray transform :class:`.LinearOperator` wrapping the 2D parallel beam
projections in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
This package provides both C and CUDA implementations of core
functionality, but note that use of the CUDA/GPU implementation
involves GPU-host-GPU memory copies when transferring JAX arrays. Other
JAX features such as automatic differentiation are not available.
"""

from typing import List, Optional

import numpy as np
import numpy.typing

import jax

import astra

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
        det_offset: float = 0.0,
        volume_geometry: Optional[List[float]] = None,
        device: str = "auto",
    ):
        """
        .. _astra-proj-geom2: https://www.astra-toolbox.com/docs/geom2d.html#projection-geometries

        Args:
            input_shape: Shape of the input array.
            det_count: Number of detector elements in the
               `projection geometry <astra-proj-geom2_>`__.
            det_spacing: Spacing between detector elements in the
               `projection geometry <astra-proj-geom2_>`__.
            angles: Array of projection angles in radians.
            det_offset: Offset of the detector center. Negative/positive
               values correspond to left/right detector shifts (i.e.
               right/left shifts of the projection within the image)
               respectively. Note that :meth:`.fbp` cannot be used when
               this offset is non-zero.
            volume_geometry: Specification of the shape of the
               discretized reconstruction volume. Must either be ``None``,
               in which case it is inferred from `input_shape`, or be a
               list of ``int`` or ``float`` scalars corresponding to the
               valid parameters of :func:`astra.creators.create_vol_geom`.
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
        self.det_offset = det_offset
        self.angles: np.ndarray = np.array(angles)

        self.proj_geom: dict = astra.create_proj_geom(
            "parallel", det_spacing, det_count, self.angles
        )
        if det_offset != 0.0:
            self.proj_geom = astra.functions.geom_postalignment(self.proj_geom, det_offset)

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
            x = np.array(x)
            proj_id, result = astra.create_sino(x, self.proj_id)
            astra.data2d.delete(proj_id)
            return result

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.output_shape, self.output_dtype), x)

    def _bproj(self, y: jax.Array) -> jax.Array:
        # apply backprojector
        def f(y):
            y = np.array(y)
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
               see ``cfg.FilterType`` in the `ASTRA documentation
               <https://www.astra-toolbox.com/docs/algs/FBP_CUDA.html>`__.

        Returns:
            Reconstructed volume.
        """

        if self.det_offset != 0.0:
            raise ValueError(
                "The fbp method may not be called when the detector offset" " is non-zero."
            )

        def f(sino):
            sino = np.array(sino)
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

            # cleanup FBP-specific array
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sino_id)
            return out

        return jax.pure_callback(f, jax.ShapeDtypeStruct(self.input_shape, self.input_dtype), sino)
