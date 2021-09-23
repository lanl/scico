# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Radon tranform LinearOperator wrapping the Super-Voxel Model-Based
Iterative Reconstruction (svmbir) package.

For more information about svmbir, see https://svmbir.readthedocs.io/.
"""

import numpy as np

import jax
import jax.experimental.host_callback

try:
    import svmbir
except ImportError:
    raise ImportError(
        "Could not import svmbir, please refer to INSTALL.rst "
        "for instructions on how to install the SVMBIR."
    )

import scico.numpy as snp
from scico.loss import WeightedSquaredL2Loss
from scico.typing import JaxArray, Shape

from ._linop import Diagonal, LinearOperator


class ParallelBeamProjector(LinearOperator):
    r"""Parallel beam x-ray projector."""

    def __init__(
        self,
        input_shape: Shape,
        angles: np.ndarray,
        num_channels: int,
    ):
        """
        Args:
            input_shape: Shape of the input array.
            angles: Array of projeciton angles in radians, should be increasing.
            num_channels: Number of pixels in the sinogram
        """
        self.input_shape = input_shape
        self.angles = angles
        self.num_channels = num_channels

        # set up custom_vjp for _eval and _adj so jax.grad works on them
        self._eval = jax.custom_vjp(lambda x: self._proj_hcb(x))
        self._eval.defvjp(lambda x: (self._proj_hcb(x), None), lambda _, y: (self._bproj_hcb(y),))

        self._adj = jax.custom_vjp(lambda y: self._bproj_hcb(y))
        self._adj.defvjp(lambda y: (self._bproj_hcb(y), None), lambda _, x: (self._proj_hcb(x),))

        super().__init__(
            input_shape=self.input_shape,
            output_shape=(len(angles), input_shape[0], num_channels),
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    @staticmethod
    def _proj(x: JaxArray, angles: JaxArray, num_channels: int) -> JaxArray:
        return svmbir.project(np.array(x), np.array(angles), num_channels, verbose=0)

    def _proj_hcb(self, x):
        # host callback wrapper for _proj
        return jax.experimental.host_callback.call(
            lambda x: self._proj(x, self.angles, self.num_channels),
            x,
            result_shape=jax.ShapeDtypeStruct(self.output_shape, self.output_dtype),
        )

    @staticmethod
    def _bproj(y: JaxArray, angles: JaxArray, num_rows: int, num_cols: int):
        return svmbir.backproject(np.array(y), np.array(angles), num_rows, num_cols, verbose=0)

    def _bproj_hcb(self, y):
        # host callback wrapper for _bproj
        return jax.experimental.host_callback.call(
            lambda y: self._bproj(y, self.angles, self.input_shape[1], self.input_shape[2]),
            y,
            result_shape=jax.ShapeDtypeStruct(self.input_shape, self.input_dtype),
        )


class SvmbirWeightedSquaredL2Loss(WeightedSquaredL2Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.A, ParallelBeamProjector):
            raise ValueError(
                "`LinearOperator` A must be a `radon_svmbir.ParallelBeamProjector`"
                "to instantiate a `SvmbirWeightedSquaredL2Loss`."
            )

        if not isinstance(self.weight_op, Diagonal):
            raise ValueError(
                f"`weight_op` must be `Diagonal` but instead got {type(self.weight_op)}"
            )

        self.weights = (
            snp.conj(self.weight_op.diagonal) * self.weight_op.diagonal
        )  # because weight_op is W^{1/2}

        self.has_prox = True

    def prox(self, v: JaxArray, lam: float) -> JaxArray:
        sigma_p = snp.sqrt(lam)
        result = svmbir.recon(
            np.array(self.y),
            np.array(self.A.angles),
            weights=np.array(self.weights),
            prox_image=np.array(v),
            num_rows=self.A.input_shape[1],
            num_cols=self.A.input_shape[2],
            sigma_p=np.float(sigma_p),
            sigma_y=1.0,
            positivity=False,
            verbose=0,
        )
        return result
        # return jax.experimental.host_callback.call(
        #     lambda x: svmbir.recon(
        #         self.y,
        #         self.A.angles,
        #         prox_image=x
        #     ),
        #     v, result_shape=jax.ShapeDtypeStruct(self.A.input_shape, self.A.input_dtype)
        # )
