# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Pseudo-functionals that have denoisers as their proximal operators."""

from scico import denoiser
from scico.typing import JaxArray

from ._functional import Functional


class BM3D(Functional):
    r"""Pseudo-functional whose prox applies the BM3D denoising algorithm.

    A pseudo-functional that has the BM3D algorithm
    :cite:`dabov-2008-image` as its proximal operator, which calls
    :func:`.denoiser.bm3d`.
    """

    has_eval = False
    has_prox = True

    def __init__(self, is_rgb: bool = False):
        r"""Initialize a :class:`BM3D` object.

        Args:
            is_rgb: Flag indicating use of BM3D with a color transform.
                    Default: ``False``.
        """

        self.is_rgb = is_rgb
        super().__init__()

    def prox(self, x: JaxArray, lam: float = 1.0, **kwargs) -> JaxArray:
        r"""Apply BM3D denoiser.

        Args:
            x: Input image.
            lam: Noise parameter.

        Returns:
            Denoised output.
        """
        return denoiser.bm3d(x, lam, self.is_rgb)


class DnCNN(Functional):
    """Pseudo-functional whose prox applies the DnCNN denoising algorithm.

    A pseudo-functional that has the DnCNN algorithm
    :cite:`zhang-2017-dncnn` as its proximal operator, implemented via
    :class:`.denoiser.DnCNN`.
    """

    has_eval = False
    has_prox = True

    def __init__(self, variant: str = "6M"):
        """

        Args:
            variant: Identify the DnCNN model to be used. See
               :class:`.denoiser.DnCNN` for valid values.
        """
        self.dncnn = denoiser.DnCNN(variant)

    def prox(self, x: JaxArray, lam: float = 1.0, **kwargs) -> JaxArray:
        r"""Apply DnCNN denoiser.

        *Warning*: The `lam` parameter is ignored, and has no effect on
        the output.

        Args:
            x: Input array.
            lam: Noise parameter (ignored).

        Returns:
            Denoised output.
        """
        return self.dncnn(x)
