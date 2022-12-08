# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Interfaces to standard denoisers."""


from typing import Any

import numpy as np

from jax.experimental import host_callback as hcb

try:
    pass
except ImportError:
    have_bm3d = False
    BM3DProfile = Any
else:
    have_bm3d = True
    from bm3d.profiles import BM3DProfile  # type: ignore

try:
    pass
except ImportError:
    have_bm4d = False
    BM4DProfile = Any
else:
    have_bm4d = True
    from bm4d.profiles import BM4DProfile  # type: ignore

import scico.numpy as snp
from scico._flax import DnCNNNet, load_weights
from scico.data import _flax_data_path
from scico.typing import JaxArray

from ._flax import FlaxMap


def bm3d(x: JaxArray, sigma: float, is_rgb: bool = False):
    r"""An interface to the BM3D denoiser :cite:`dabov-2008-image`.

    BM3D denoising is performed using the
    `code <https://pypi.org/project/bm3d>`__ released with
    :cite:`makinen-2019-exact`. Since this package is an interface
    to compiled C code, JAX features such as automatic differentiation
    and support for GPU devices are not available.

    Args:
        x: Input image. Expected to be a 2D array (gray-scale denoising)
            or 3D array (color denoising). Higher-dimensional arrays are
            tolerated only if the additional dimensions are singletons.
            For color denoising, the color channel is assumed to be in the
            last non-singleton dimension.
        sigma: Noise standard deviation.
        is_rgb: Flag indicating use of BM3D with a color transform.
            Default: ``False``.

    Returns:
        Denoised output.
    """
    if not have_bm3d:
        raise RuntimeError("Package bm3d is required for use of this function.")

    if is_rgb is True:
        bm3d_eval = tunibm3d.bm3d_rgb
    else:
        bm3d_eval = tunibm3d.bm3d

    if snp.util.is_complex_dtype(x.dtype):
        raise TypeError(f"BM3D requires real-valued inputs, got {x.dtype}.")

    # Support arrays with more than three axes when the additional axes are singletons.
    x_in_shape = x.shape

    if isinstance(x.ndim, tuple) or x.ndim < 2:
        raise ValueError(
            "BM3D requires two-dimensional or three dimensional inputs; got ndim = {x.ndim}."
        )

    # This check is also performed inside the BM3D call, but due to the host_callback,
    # no exception is raised and the program will crash with no traceback.
    # NOTE: if BM3D is extended to allow for different profiles, the block size must be
    #       updated; this presumes 'np' profile (bs=8)
    if np.min(x.shape[:2]) < 8:
        raise ValueError(
            "Two leading dimensions of input cannot be smaller than block size "
            f"(8); got image size = {x.shape}."
        )

    if x.ndim > 3:
        if all(k == 1 for k in x.shape[3:]):
            x = x.squeeze()
        else:
            raise ValueError(
                "Arrays with more than three axes are only supported when "
                " the additional axes are singletons."
            )

    y = hcb.call(lambda args: bm3d_eval(*args).astype(x.dtype), (x, sigma), result_shape=x)

    # undo squeezing, if neccessary
    y = y.reshape(x_in_shape)

    return y


def bm4d(x: JaxArray, sigma: float):
    r"""An interface to the BM4D denoiser :cite:`maggioni-2012-nonlocal`.

    BM4D denoising is performed using the
    `code <https://pypi.org/project/bm4d/>`__ released by the authors of
    :cite:`maggioni-2012-nonlocal`. Since this package is an interface
    to compiled C code, JAX features such as automatic differentiation
    and support for GPU devices are not available.

    Args:
        x: Input image. Expected to be a 3D array. Higher-dimensional
            arrays are tolerated only if the additional dimensions are
            singletons.
        sigma: Noise standard deviation.

    Returns:
        Denoised output.
    """
    if not have_bm4d:
        raise RuntimeError("Package bm4d is required for use of this function.")

    bm4d_eval = tunibm4d.bm4d

    if snp.util.is_complex_dtype(x.dtype):
        raise TypeError(f"BM4D requires real-valued inputs, got {x.dtype}.")

    # Support arrays with more than three axes when the additional axes are singletons.
    x_in_shape = x.shape

    if isinstance(x.ndim, tuple) or x.ndim < 3:
        raise ValueError(f"BM4D requires three-dimensional inputs; got ndim = {x.ndim}.")

    # This check is also performed inside the BM4D call, but due to the host_callback,
    # no exception is raised and the program will crash with no traceback.
    # NOTE: if BM4D is extended to allow for different profiles, the block size must be
    #       updated; this presumes 'np' profile (bs=8)
    if np.min(x.shape[:3]) < 8:
        raise ValueError(
            "Three leading dimensions of input cannot be smaller than block size "
            f"(8); got image size = {x.shape}."
        )

    if x.ndim > 3:
        if all(k == 1 for k in x.shape[3:]):
            x = x.squeeze()
        else:
            raise ValueError(
                "Arrays with more than three axes are only supported when "
                " the additional axes are singletons."
            )

    y = hcb.call(lambda args: bm4d_eval(*args).astype(x.dtype), (x, sigma), result_shape=x)

    # undo squeezing, if neccessary
    y = y.reshape(x_in_shape)

    return y


class DnCNN(FlaxMap):
    """Flax implementation of the DnCNN denoiser.

    A flax implementation of the DnCNN denoiser :cite:`zhang-2017-dncnn`.
    Note that :class:`.DnCNNNet` represents an untrained form of the
    generic DnCNN CNN structure, while this class represents a trained
    form with six or seventeen layers.

    The standard DnCNN as proposed in :cite:`zhang-2017-dncnn` is a blind
    denoiser without a noise level input. This implementation also supports
    a custom variant that includes a noise level input (as in
    :cite:`zhang-2021-plug`, but using a different network architecture).
    """

    def __init__(self, variant: str = "6M"):
        """
        Note that all DnCNN models are trained for single-channel image
        input. Multi-channel input is supported via independent denoising
        of each channel.

        Args:
            variant: Identify the DnCNN model to be used. Options are
                '6L', '6M' (default), '6H', '6N', '17L', '17M', '17H',
                and '17N', where the integer indicates the number of
                layers in the network, and the postfix indicates the
                training noise standard deviation (with respect to data
                in the range [0, 1]): L (low) = 0.06, M (mid) = 0.1,
                H (high) = 0.2, or N indicating that a noise level input
                is available.
        """

        self.variant = variant

        if variant not in ["6L", "6M", "6H", "17L", "17M", "17H", "6N", "17N"]:
            raise ValueError(f"Invalid value {variant} of parameter variant.")
        if variant[0] == "6":
            nlayer = 6
        else:
            nlayer = 17

        channels = 2 if variant in ["6N", "17N"] else 1

        model = DnCNNNet(depth=nlayer, channels=channels, num_filters=64, dtype=np.float32)
        variables = load_weights(_flax_data_path("dncnn%s.npz" % variant))
        super().__init__(model, variables)

    def __call__(self, x: JaxArray, sigma: Optional[float] = None) -> JaxArray:
        r"""Apply DnCNN denoiser.

        Args:
            x: Input array.
            sigma: Noise standard deviation (for variants `6N` and `17N`).

        Returns:
            Denoised output.
        """
        if sigma is not None and self.variant not in ["6N", "17N"]:
            raise ValueError(
                "A non-default value for the sigma parameter may "
                "only be specified when the variant is 6N or 17N"
                f"; got variant = {self.variant}."
            )

        if sigma is None and self.variant in ["6N", "17N"]:
            raise ValueError(
                "A float value must be specified for the sigma "
                "parameter when the variant is 6N or 17N."
            )

        if snp.util.is_complex_dtype(x.dtype):
            raise TypeError(f"DnCNN requries real-valued inputs, got {x.dtype}.")

        if isinstance(x.ndim, tuple) or x.ndim < 2:
            raise ValueError(
                "DnCNN requires two-dimensional (M, N) or three-dimensional (M, N, C)"
                f" inputs; got ndim = {x.ndim}."
            )

        x_in_shape = x.shape
        if x.ndim > 3:
            if all(k == 1 for k in x.shape[3:]):
                x = x.squeeze()
            else:
                raise ValueError(
                    "Arrays with more than three axes are only supported when"
                    " the additional axes are singletons."
                )

        if x.ndim == 3:
            y = snp.swapaxes(x, 0, -1)

            if sigma is not None:
                y = snp.stack([y, snp.ones_like(y) * sigma], -1)
            else:
                y = y[..., np.newaxis]

            # swap channel axis to batch axis and add singleton axis at end
            y = super().__call__(y)
            # drop singleton axis and swap axes back to original positions
            y = snp.swapaxes(y[..., 0], 0, -1)

        else:
            if sigma is not None:
                x = snp.stack([x, snp.ones_like(x) * sigma], -1)
                x = x[np.newaxis, ...]

            y = super().__call__(x)

            if sigma is not None:
                y = y[0, ..., 0]

        y = y.reshape(x_in_shape)

        return y
