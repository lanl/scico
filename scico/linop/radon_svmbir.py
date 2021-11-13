# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Radon transform LinearOperator wrapping the svmbir package.

Radon transform LinearOperator wrapping the
`svmbir <https://github.com/cabouman/svmbir>`_ package.
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

from ._linop import LinearOperator


class ParallelBeamProjector(LinearOperator):
    r"""Parallel beam projector based on svmbir."""

    def __init__(
        self,
        input_shape: Shape,
        angles: np.ndarray,
        num_channels: int,
    ):
        """
        Args:
            input_shape: Shape of the input array.
            angles: Array of projection angles in radians, should be
              increasing.
            num_channels: Number of pixels in the sinogram
        """
        self.angles = angles
        self.num_channels = num_channels

        if len(input_shape) == 2:  # 2D input
            self.svmbir_input_shape = (1,) + input_shape
            output_shape = (len(angles), num_channels)
            self.svmbir_output_shape = output_shape[0:1] + (1,) + output_shape[1:2]
        elif len(input_shape) == 3:  # 3D input
            self.svmbir_input_shape = input_shape
            output_shape = (len(angles), input_shape[0], num_channels)
            self.svmbir_output_shape = output_shape
        else:
            raise ValueError(
                f"Only 2D and 3D inputs are supported, but input_shape was {input_shape}"
            )

        # Set up custom_vjp for _eval and _adj so jax.grad works on them.
        self._eval = jax.custom_vjp(lambda x: self._proj_hcb(x))
        self._eval.defvjp(lambda x: (self._proj_hcb(x), None), lambda _, y: (self._bproj_hcb(y),))

        self._adj = jax.custom_vjp(lambda y: self._bproj_hcb(y))
        self._adj.defvjp(lambda y: (self._bproj_hcb(y), None), lambda _, x: (self._proj_hcb(x),))

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            adj_fn=self._adj,
            jit=False,
        )

    @staticmethod
    def _proj(x: JaxArray, angles: JaxArray, num_channels: int) -> JaxArray:
        return svmbir.project(np.array(x), np.array(angles), num_channels, verbose=0)

    def _proj_hcb(self, x):
        x = x.reshape(self.svmbir_input_shape)
        # host callback wrapper for _proj
        y = jax.experimental.host_callback.call(
            lambda x: self._proj(x, self.angles, self.num_channels),
            x,
            result_shape=jax.ShapeDtypeStruct(self.svmbir_output_shape, self.output_dtype),
        )
        return y.reshape(self.output_shape)

    @staticmethod
    def _bproj(y: JaxArray, angles: JaxArray, num_rows: int, num_cols: int):
        return svmbir.backproject(np.array(y), np.array(angles), num_rows, num_cols, verbose=0)

    def _bproj_hcb(self, y):
        y = y.reshape(self.svmbir_output_shape)
        # host callback wrapper for _bproj
        x = jax.experimental.host_callback.call(
            lambda y: self._bproj(
                y, self.angles, self.svmbir_input_shape[1], self.svmbir_input_shape[2]
            ),
            y,
            result_shape=jax.ShapeDtypeStruct(self.svmbir_input_shape, self.input_dtype),
        )
        return x.reshape(self.input_shape)


class SVMBIRWeightedSquaredL2Loss(WeightedSquaredL2Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.A, ParallelBeamProjector):
            raise ValueError(
                "`LinearOperator` A must be a `radon_svmbir.ParallelBeamProjector`"
                "to instantiate a `SVMBIRWeightedSquaredL2Loss`."
            )

        self.has_prox = True

    def prox(self, v: JaxArray, lam: float) -> JaxArray:
        v = v.reshape(self.A.svmbir_input_shape)
        y = self.y.reshape(self.A.svmbir_output_shape)
        weights = self.W.diagonal.reshape(self.A.svmbir_output_shape)
        sigma_p = snp.sqrt(lam)
        result = svmbir.recon(
            np.array(y),
            np.array(self.A.angles),
            weights=np.array(weights),
            prox_image=np.array(v),
            num_rows=self.A.svmbir_input_shape[1],
            num_cols=self.A.svmbir_input_shape[2],
            sigma_p=float(sigma_p),
            sigma_y=1.0,
            positivity=False,
            verbose=0,
        )
        return result.reshape(self.A.input_shape)


def _unsqueeze(x: JaxArray, input_shape: Shape) -> JaxArray:
    """If x is 2D, make it 3D according to SVMBIR's convention."""
    if len(input_shape) == 2:
        x = x[snp.newaxis, :, :]
    return x
