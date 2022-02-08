# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
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
from scico.loss import Loss, WeightedSquaredL2Loss
from scico.typing import JaxArray, Shape

from ._linop import Diagonal, Identity, LinearOperator

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
        center_offset: Optional[float] = 0.0,
        is_masked: Optional[bool] = False,
    ):
        """
        Args:
            input_shape: Shape of the input array.
            angles: Array of projection angles in radians, should be
                increasing.
            num_channels: Number of pixels in the sinogram.
            center_offset: Position of the detector center relative to the center-of-rotation,
                in units of pixels
            is_masked: If ``True``, the valid region of the image is
                determined by a mask defined as the circle inscribed
                within the image boundary. Otherwise, the whole image
                array is taken into account by projections.
        """
        self.angles = angles
        self.num_channels = num_channels
        self.center_offset = center_offset

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

        self.is_masked = is_masked
        if self.is_masked:
            self.roi_radius = None
        else:
            self.roi_radius = max(self.svmbir_input_shape[1], self.svmbir_input_shape[2])

        # Set up custom_vjp for _eval and _adj so jax.grad works on them.
        self._eval = jax.custom_vjp(self._proj_hcb)
        self._eval.defvjp(lambda x: (self._proj_hcb(x), None), lambda _, y: (self._bproj_hcb(y),))

        self._adj = jax.custom_vjp(self._bproj_hcb)
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
        x: JaxArray,
        angles: JaxArray,
        num_channels: int,
        center_offset: Optional[float] = 0.0,
        roi_radius: Optional[float] = None,
    ) -> JaxArray:
        return jax.device_put(
            svmbir.project(
                np.array(x),
                np.array(angles),
                num_channels,
                verbose=0,
                center_offset=center_offset,
                roi_radius=roi_radius,
            )
        )

    def _proj_hcb(self, x):
        x = x.reshape(self.svmbir_input_shape)
        # host callback wrapper for _proj
        y = jax.experimental.host_callback.call(
            lambda x: self._proj(
                x,
                self.angles,
                self.num_channels,
                center_offset=self.center_offset,
                roi_radius=self.roi_radius,
            ),
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
        center_offset: Optional[float] = 0.0,
        roi_radius: Optional[float] = None,
    ):
        return jax.device_put(
            svmbir.backproject(
                np.array(y),
                np.array(angles),
                num_rows,
                num_cols,
                verbose=0,
                center_offset=center_offset,
                roi_radius=roi_radius,
            )
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
                center_offset=self.center_offset,
                roi_radius=self.roi_radius,
            ),
            y,
            result_shape=jax.ShapeDtypeStruct(self.svmbir_input_shape, self.input_dtype),
        )
        return x.reshape(self.input_shape)


class SVMBIRExtendedLoss(Loss):
    r"""Extended Weighted squared :math:`\ell_2` loss with svmbir CT projector.

    Generalization of the weighted squared :math:`\ell_2` loss of a CT reconstruction problem,

    .. math::
        \alpha \norm{\mb{y} - A(\mb{x})}_W^2 =
        \alpha \left(\mb{y} - A(\mb{x})\right)^T W \left(\mb{y} -
        A(\mb{x})\right) \;,

    where :math:`A` is a :class:`.ParallelBeamProjector`,
    :math:`\alpha` is the scaling parameter and :math:`W` is an instance
    of :class:`scico.linop.Diagonal`. If :math:`W` is None, it is set to
    :class:`scico.linop.Identity`.

    The extended loss differs from a typical weighted squared :math:`\ell_2` loss
    is the following aspects.
    When ``positivity=True``, the prox projects onto the non-negative
    orthant and the loss is infinite if any element of the input is negative.
    When the ``is_masked`` option of the associated :class:`.ParallelBeamProjector` is `True`,
    the reconstruction is computed over a masked region of the image as described
    in class :class:`.ParallelBeamProjector`.
    """

    def __init__(
        self,
        *args,
        prox_kwargs: Optional[dict] = None,
        positivity: bool = False,
        W: Optional[Diagonal] = None,
        **kwargs,
    ):
        r"""Initialize a :class:`SVMBIRExtendedLoss` object.

        Args:
            y: Sinogram measurement.
            A: Forward operator.
            scale: Scaling parameter.
            W:  Weighting diagonal operator. Must be non-negative.
                If ``None``, defaults to :class:`.Identity`.
            prox_kwargs: Dictionary of arguments passed to the
                :meth:`svmbir.recon` prox routine. Defaults to
                {"maxiter": 1000, "ctol": 0.001}.
            positivity: Enforce positivity in the prox operation. The
                loss is infinite if any element of the input is negative.
        """
        super().__init__(*args, **kwargs)

        if not isinstance(self.A, ParallelBeamProjector):
            raise ValueError("LinearOperator A must be a radon_svmbir.ParallelBeamProjector.")

        self.has_prox = True

        if prox_kwargs is None:
            prox_kwargs = {}

        default_prox_args = {"maxiter": 1000, "ctol": 0.001}
        default_prox_args.update(prox_kwargs)

        svmbir_prox_args = {}
        if "maxiter" in default_prox_args:
            svmbir_prox_args["max_iterations"] = default_prox_args["maxiter"]
        if "ctol" in default_prox_args:
            svmbir_prox_args["stop_threshold"] = default_prox_args["ctol"]
        self.svmbir_prox_args = svmbir_prox_args

        self.positivity = positivity

        if W is None:
            self.W = Identity(self.y.shape)
        elif isinstance(W, Diagonal):
            if snp.all(W.diagonal >= 0):
                self.W = W
            else:
                raise Exception(f"The weights, W, must be non-negative.")
        else:
            raise TypeError(f"W must be None or a linop.Diagonal, got {type(W)}")

    def __call__(self, x: JaxArray) -> float:

        if self.positivity and snp.sum(x < 0) > 0:
            return snp.inf
        else:
            return self.scale * (self.W.diagonal * snp.abs(self.y - self.A(x)) ** 2).sum()

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
            center_offset=self.A.center_offset,
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


class SVMBIRWeightedSquaredL2Loss(SVMBIRExtendedLoss, WeightedSquaredL2Loss):
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
    """

    def __init__(
        self,
        *args,
        prox_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""Initialize a :class:`SVMBIRWeightedSquaredL2Loss` object.

        Args:
            y: Sinogram measurement.
            A: Forward operator.
            scale: Scaling parameter.
            W:  Weighting diagonal operator. Must be non-negative.
                If None, defaults to :class:`.Identity`.
            prox_kwargs: Dictionary of arguments passed to the
                :meth:`svmbir.recon` prox routine. Defaults to
                {"maxiter": 1000, "ctol": 0.001}.
        """
        super().__init__(*args, **kwargs, prox_kwargs=prox_kwargs, positivity=False)

        if self.A.is_masked:
            raise ValueError(
                "is_masked must be false for the ParallelBeamProjector in SVMBIRWeightedSquaredL2Loss."
            )


def _unsqueeze(x: JaxArray, input_shape: Shape) -> JaxArray:
    """If x is 2D, make it 3D according to the SVMBIR convention."""
    if len(input_shape) == 2:
        x = x[snp.newaxis, :, :]
    return x
