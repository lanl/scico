from typing import List, Optional

import jax
import jax.experimental.host_callback as hcb
import numpy as np
from astra import algorithm

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


class TomographicProjector3D(LinearOperator):
    r"""
    TBD
    """

    def __init__(
            self,
            input_shape: Shape,
            detector_spacing_x: float,
            detector_spacing_y: float,
            detector_row_count: int,
            detector_col_count: int,
            angles: np.ndarray,
            volume_geometry: Optional[List[float]] = None
            # device can only be gpu
    ):
        """
        TBD
        """

        self.detector_spacing_x: float = detector_spacing_x
        self.detector_spacing_y: float = detector_spacing_y
        self.detector_row_count: int = detector_row_count
        self.detector_col_count: int = detector_col_count
        self.angles: np.ndarray = angles

        self.proj_geom = astra.create_proj_geom("parallel3d", detector_spacing_x, detector_spacing_y, detector_row_count,
                                                detector_col_count, angles)
        self.proj_id: int
        self.input_shape: tuple = input_shape

        if volume_geometry is not None:
            if len(volume_geometry) == 6:
                self.vol_geom: dict = astra.create_vol_geom(*input_shape, *volume_geometry)
            else:
                raise ValueError(
                    "volume_geometry must be the shape of the volume as a tuple of len 6 "
                    "containing the volume geometry dimensions. Please see documentation "
                    "for details."
                )
        else:
            (Nz, Ny, Nx) = input_shape
            self.vol_geom = astra.create_vol_geom(Ny, Nx, Nz)

        # hardcoded to use cuda3d (the only option), might not be necessary/needed
        # self.proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)

        # Wrap our non-jax function to indicate we will supply fwd/rev mode functions
        self._eval = jax.custom_vjp(self._proj)
        self._eval.defvjp(lambda x: (self._proj(x), None), lambda _, y: (self._bproj(y),))  # type: ignore
        self._adj = jax.custom_vjp(self._bproj)
        self._adj.defvjp(lambda y: (self._bproj(y), None), lambda _, x: (self._proj(x),))  # type: ignore

        super().__init__(
            input_shape=input_shape,
            output_shape=(detector_row_count, len(angles), detector_col_count),
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    def _proj(self, x: jax.Array) -> jax.Array:
        # Applies the forward projector and generates a sinogram
        def f(x):
            if not x.flags.writeable:
                x.flags.writeable = True
            proj_id, result = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        # print("output shape:", self.output_shape)
        return hcb.call(f, x, result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype))

    def _bproj(self, y: jax.Array) -> jax.Array:
        # applies backprojector
        def f(y):
            if not y.flags.writeable:
                y.flags.writeable = True
            proj_id, result = astra.create_backprojection3d_gpu(y, self.proj_geom, self.vol_geom)
            astra.data3d.delete(proj_id)
            return result

        return hcb.call(f, y, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype))

    def sirt(self, sino: jax.Array) -> jax.Array:
        def f(sino):
            if not sino.flags.writeable:
                sino.flags.writeable = True
            proj_id = astra.data3d.create('-sino', self.proj_geom, sino)
            rec_id = astra.data3d.create('-vol', self.vol_geom)
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ProjectionDataId'] = proj_id
            cfg['ReconstructionDataId'] = rec_id
            alg_id = algorithm.create(cfg)
            algorithm.run(alg_id, 50)

            out = astra.data3d.get(rec_id)

            algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(rec_id)

            return out
        # return f(sino)
        return hcb.call(f, sino, result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype))
