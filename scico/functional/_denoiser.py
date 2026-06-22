# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Pseudo-functionals that have denoisers as their proximal operators."""

from typing import Union

from scico import denoiser
from scico.numpy import Array

from ._functional import Functional


class BM3D(Functional):
    r"""Pseudo-functional whose prox applies the BM3D denoising algorithm.

    A pseudo-functional that has the BM3D algorithm
    :cite:`dabov-2008-image` as its proximal operator, which calls
    :func:`.denoiser.bm3d`. Since this function provides an interface
    to compiled C code, JAX features such as automatic differentiation
    and support for GPU devices are not available.
    """

    has_eval = False
    has_prox = True

    def __init__(self, is_rgb: bool = False, profile: Union[denoiser.BM3DProfile, str] = "np"):
        r"""Initialize a :class:`BM3D` object.

        Args:
            is_rgb: Flag indicating use of BM3D with a color transform.
                    Default: ``False``.
            profile: Parameter configuration for BM3D.
        """

        self.is_rgb = is_rgb
        self.profile = profile
        super().__init__()

    def prox(self, x: Array, lam: float = 1.0, **kwargs) -> Array:  # type: ignore
        r"""Apply BM3D denoiser.

        Args:
            x: Input image.
            lam: Noise parameter.
            **kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Denoised output.
        """
        return denoiser.bm3d(x, lam, self.is_rgb, profile=self.profile)


class BM4D(Functional):
    r"""Pseudo-functional whose prox applies the BM4D denoising algorithm.

    A pseudo-functional that has the BM4D algorithm
    :cite:`maggioni-2012-nonlocal` as its proximal operator, which calls
    :func:`.denoiser.bm4d`. Since this function provides an interface
    to compiled C code, JAX features such as automatic differentiation
    and support for GPU devices are not available.
    """

    has_eval = False
    has_prox = True

    def __init__(self, profile: Union[denoiser.BM4DProfile, str] = "np"):
        r"""Initialize a :class:`BM4D` object.

        Args:
            profile: Parameter configuration for BM4D.
        """
        self.profile = profile
        super().__init__()

    def prox(self, x: Array, lam: float = 1.0, **kwargs) -> Array:  # type: ignore
        r"""Apply BM4D denoiser.

        Args:
            x: Input image.
            lam: Noise parameter.
            **kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Denoised output.
        """
        return denoiser.bm4d(x, lam, profile=self.profile)


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
        if self.dncnn.is_blind:

            def denoise(x, sigma):
                return self.dncnn(x)

        else:

            def denoise(x, sigma):
                return self.dncnn(x, sigma)

        self._denoise = denoise

    def prox(self, x: Array, lam: float = 1.0, **kwargs) -> Array:  # type: ignore
        r"""Apply DnCNN denoiser.

        *Warning*: The `lam` parameter is ignored, and has no effect on
        the output for :class:`.DnCNN` objects initialized with
        :code:`variant` parameter values other than `6N` and `17N`.

        Args:
            x: Input array.
            lam: Noise parameter (ignored).
            **kwargs: Additional arguments that may be used by derived
                classes.

        Returns:
            Denoised output.
        """
        return self._denoise(x, lam)
