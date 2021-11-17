# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals that are indicator functions/constraints."""

from typing import Optional

import numpy as np

from jax.experimental import host_callback as hcb

from bm3d import bm3d, bm3d_rgb

import scico.numpy as snp
from scico.blockarray import BlockArray
from scico.data import _flax_data_path
from scico.flax import DnCNNNet, load_weights
from scico.typing import JaxArray

from ._flax import FlaxMap
from ._functional import Functional

__author__ = """Luke Pfister <luke.pfister@gmail.com>"""


class BM3D(Functional):
    r"""Functional whose prox applies the BM3D denoising algorithm :cite:`dabov-2008-image`.

    The BM3D algorithm is computed using the `code <https://pypi.org/project/bm3d>`__ released
    with :cite:`makinen-2019-exact`.
    """

    has_eval = False
    has_prox = True
    is_smooth = False

    def __init__(self, is_rgb: Optional[bool] = False):
        r"""Initialize a :class:`BM3D` object.

        Args:
            is_rgb : Flag indicating use of BM3D with a color transform.  Default: False.
        """

        if is_rgb is True:
            self.is_rgb = True
            self.bm3d_eval = bm3d_rgb
        else:
            self.is_rgb = False
            self.bm3d_eval = bm3d

        super().__init__()

    def prox(self, x: JaxArray, lam: float = 1.0, **kwargs) -> JaxArray:
        r"""Apply BM3D denoiser with noise level ``lam``.

        Args:
            x : input image.
            lam : noise level.

        Returns:
            BM3D denoised output.
        """

        # BM3D only works on (NxN) or (NxNxC) arrays
        # In future, may want to apply prox along an "axis"
        # But that time isn't now
        if isinstance(x, BlockArray):
            raise NotImplementedError

        if np.iscomplexobj(x):
            raise TypeError(f"BM3D requries real-valued inputs, got {x.dtype}")

        # Support arrays with more than three axes when the additional axes are singletons
        x_in_shape = x.shape

        if x.ndim < 2:
            raise ValueError(
                f"BM3D requires two dimensional (M, N) or three dimensional (M, N, C)"
                " inputs; got ndim = {x.ndim}"
            )

        # this check is also performed inside the BM3D call, but due to the host_callback,
        # no exception is raised and the program will crash with no traceback.
        # NOTE:  if BM3D is extended to allow for different profiles, the block size must be
        #        updated; this presumes 'np' profile (bs=8)
        if np.min(x.shape[:2]) < 8:
            raise ValueError(
                f"Two leading dimensions of input cannot be smaller than block size "
                f"(8);  got image size = {x.shape}"
            )

        if x.ndim > 3:
            if all(k == 1 for k in x.shape[3:]):
                x = x.squeeze()
            else:
                raise ValueError(
                    "Arrays with more than three axes are only supported when "
                    " the additional axes are singletons"
                )

        y = hcb.call(lambda args: self.bm3d_eval(*args).astype(x.dtype), (x, lam), result_shape=x)

        # undo squeezing, if neccessary
        y = y.reshape(x_in_shape)

        return y


class DnCNN(FlaxMap):
    """Flax implementation of the DnCNN denoiser :cite:`zhang-2017-dncnn`.

    Note that :class:`.flax.DnCNNNet` represents an untrained form of the
    generic DnCNN CNN structure, while this class represents a trained
    form with six or seventeen layers.
    """

    def __init__(self, variant: Optional[str] = "6M"):
        """Initialize a :class:`DnCNN` object.

        Note that all DnCNN models are trained for single-channel image
        input. Multi-channel input is supported via independent denoising
        of each channel.

        Args:
            variant : Identify the DnCNN model to be used. Options are
                '6L', '6M' (default), '6H', '17L', '17M', and '17H',
                where the integer indicates the number of layers in the
                network, and the postfix indicates the training noise
                standard deviation: L (low) = 0.06, M (mid) = 0.1,
                H (high) = 0.2, where the standard deviations are
                with respect to data in the range [0, 1].
        """
        if variant not in ["6L", "6M", "6H", "17L", "17M", "17H"]:
            raise RuntimeError(f"Invalid value of parameter variant: {variant}")
        if variant[0] == "6":
            nlayer = 6
        else:
            nlayer = 17
        model = DnCNNNet(depth=nlayer, channels=1, num_filters=64, dtype=np.float32)
        variables = load_weights(_flax_data_path("dncnn%s.npz" % variant))
        super().__init__(model, variables)

    def prox(self, x: JaxArray, lam: float = 1) -> JaxArray:
        r"""Apply DnCNN denoiser.

        *Warning*: The ``lam`` parameter is ignored, and has no effect on
        the output.

        Args:
            x : input.
            lam : noise estimate (ignored).

        Returns:
            DnCNN denoised output.
        """
        if np.iscomplexobj(x):
            raise TypeError(f"DnCNN requries real-valued inputs, got {x.dtype}")

        if x.ndim < 2:
            raise ValueError(
                f"DnCNN requires two dimensional (M, N) or three dimensional (M, N, C)"
                " inputs; got ndim = {x.ndim}"
            )

        x_in_shape = x.shape
        if x.ndim > 3:
            if all(k == 1 for k in x.shape[3:]):
                x = x.squeeze()
            else:
                raise ValueError(
                    "Arrays with more than three axes are only supported when "
                    " the additional axes are singletons"
                )

        if x.ndim == 3:
            # swap channel axis to batch axis and add singleton axis at end
            y = super().prox(snp.swapaxes(x, 0, -1)[..., np.newaxis], lam)
            # drop singleton axis and swap axes back to original positions
            y = snp.swapaxes(y[..., 0], 0, -1)
        else:
            y = super().prox(x, lam)

        y = y.reshape(x_in_shape)

        return y
