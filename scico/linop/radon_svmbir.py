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

from typing import Optional

import numpy as np

import jax
import jax.experimental.host_callback

import scico.numpy as snp
from scico.loss import WeightedSquaredL2Loss
from scico.typing import JaxArray, Shape

from ._linop import LinearOperator

try:
    import svmbir
except ImportError:
    raise ImportError("Could not import svmbir; please install it.")


class ParallelBeamProjector(LinearOperator):
    r"""Parallel beam Radon transform based on svmbir.

    Perform tomographic projection of an image at specified angles, using
    the `svmbir <https://github.com/cabouman/svmbir>`_ package. The
    ``is_masked`` option selects whether a valid region for projections
    (pixels outside this region are ignored when performing the
    projection) is active. This region of validity is also respected by
    :meth:`.SVMBIRWeightedSquaredL2Loss.prox` when
    :class:`.SVMBIRWeightedSquaredL2Loss` is initialized with a
    :class:`ParallelBeamProjector` with this option enabled.
    """

    def __init__(
        self,
        input_shape: Shape,
        angles: np.ndarray,
        num_channels: int,
        is_masked: Optional[bool] = False,
    ):
        """
        Args:
            input_shape: Shape of the input array.
            angles: Array of projection angles in radians, should be
                increasing.
            num_channels: Number of pixels in the sinogram
            is_masked:  If True, the valid region of the image is
                determined by a mask defined as the circle inscribed
                within the image boundary. Otherwise, the whole image
                array is taken into account by projections.
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

        if is_masked:
            self.roi_radius = None
        else:
            self.roi_radius = max(self.svmbir_input_shape[1], self.svmbir_input_shape[2])

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
    def _proj(
        x: JaxArray, angles: JaxArray, num_channels: int, roi_radius: Optional[float] = None
    ) -> JaxArray:
        return svmbir.project(
            np.array(x), np.array(angles), num_channels, verbose=0, roi_radius=roi_radius
        )

    def _proj_hcb(self, x):
        x = x.reshape(self.svmbir_input_shape)
        # host callback wrapper for _proj
        y = jax.experimental.host_callback.call(
            lambda x: self._proj(x, self.angles, self.num_channels, self.roi_radius),
            x,
            result_shape=jax.ShapeDtypeStruct(self.svmbir_output_shape, self.output_dtype),
        )
        return y.reshape(self.output_shape)

    @staticmethod
    def _bproj(
        y: JaxArray,
        angles: JaxArray,
        num_rows: int,
        num_cols: int,
        roi_radius: Optional[float] = None,
    ):
        return svmbir.backproject(
            np.array(y), np.array(angles), num_rows, num_cols, verbose=0, roi_radius=roi_radius
        )

    def _bproj_hcb(self, y):
        y = y.reshape(self.svmbir_output_shape)
        # host callback wrapper for _bproj
        x = jax.experimental.host_callback.call(
            lambda y: self._bproj(
                y,
                self.angles,
                self.svmbir_input_shape[1],
                self.svmbir_input_shape[2],
                self.roi_radius,
            ),
            y,
            result_shape=jax.ShapeDtypeStruct(self.svmbir_input_shape, self.input_dtype),
        )
        return x.reshape(self.input_shape)


class SVMBIRWeightedSquaredL2Loss(WeightedSquaredL2Loss):
    r"""Weighted squared :math:`\ell_2` loss with svmbir CT projector.

    Weighted squared :math:`\ell_2` loss of a CT reconstruction problem,

    .. math::
        \alpha \norm{\mb{y} - A(\mb{x})}_W^2 =
        \alpha \left(\mb{y} - A(\mb{x})\right)^T W \left(\mb{y} -
        A(\mb{x})\right) \;,

    where :math:`A` is a :class:`.ParallelBeamProjector`,
    :math:`\alpha` is the scaling parameter and :math:`W` is an instance
    of :class:`scico.linop.Diagonal`. If :math:`W` is None, it is set to
    :class:`scico.linop.Identity`.

    When ``positivity=True``, the prox projects onto the non-negative
    quadrant, but the the loss, :math:`\alpha l(\mb{y}, A(\mb{x}))`,
    is unaffected by this setting and still evaluates to finite values
    when :math:`\mb{x}` is not in the non-negative quadrant.

    """

    def __init__(
        self,
        *args,
        prox_kwargs: dict = {"maxiter": 1000, "ctol": 0.001},
        positivity: bool = False,
        **kwargs,
    ):
        r"""Initialize a :class:`SVMBIRWeightedSquaredL2Loss` object.

        Args:
            y : Sinogram measurement.
            A : Forward operator.
            scale : Scaling parameter.
            W:  Weighting diagonal operator. Must be non-negative.
                If None, defaults to :class:`.Identity`.
            prox_kwargs: Dictionary of arguments passed to the
                :meth:`svmbir.recon` prox routine. Note that omitting
                this dictionary will cause the default dictionary to be
                used, however omitting entries in the passed dictionary
                causes the defaults of the underlying :meth:`svmbir.recon`
                prox routine to be used.
            positivity: Enforce positivity in the prox operation. The
                loss is not affected.
        """
        super().__init__(*args, **kwargs)

        if not isinstance(self.A, ParallelBeamProjector):
            raise ValueError("LinearOperator A must be a radon_svmbir.ParallelBeamProjector.")

        self.has_prox = True

        if prox_kwargs is None:
            prox_kwargs = dict()

        svmbir_prox_args = dict()
        if "maxiter" in prox_kwargs:
            svmbir_prox_args["max_iterations"] = prox_kwargs["maxiter"]
        if "ctol" in prox_kwargs:
            svmbir_prox_args["stop_threshold"] = prox_kwargs["ctol"]
        self.svmbir_prox_args = svmbir_prox_args

        self.positivity = positivity

    def prox(self, v: JaxArray, lam: float, **kwargs) -> JaxArray:
        v = v.reshape(self.A.svmbir_input_shape)
        y = self.y.reshape(self.A.svmbir_output_shape)
        weights = self.W.diagonal.reshape(self.A.svmbir_output_shape)
        sigma_p = snp.sqrt(lam)
        if "v0" in kwargs and kwargs["v0"] is not None:
            v0 = np.reshape(np.array(kwargs["v0"]), self.A.svmbir_input_shape)
        else:
            v0 = 0.0

        # change: stop, mask-rad, init
        result = svmbir.recon(
            np.array(y),
            np.array(self.A.angles),
            weights=np.array(weights),
            prox_image=np.array(v),
            num_rows=self.A.svmbir_input_shape[1],
            num_cols=self.A.svmbir_input_shape[2],
            roi_radius=self.A.roi_radius,
            sigma_p=float(sigma_p),
            sigma_y=1.0,
            positivity=self.positivity,
            verbose=0,
            init_image=v0,
            **self.svmbir_prox_args,
        )
        if np.sum(np.isnan(result)):
            raise ValueError("Result contains NaNs")

        return jax.device_put(result.reshape(self.A.input_shape))


def _unsqueeze(x: JaxArray, input_shape: Shape) -> JaxArray:
    """If x is 2D, make it 3D according to SVMBIR's convention."""
    if len(input_shape) == 2:
        x = x[snp.newaxis, :, :]
    return x
