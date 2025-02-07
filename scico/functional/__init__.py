# -*- coding: utf-8 -*-
# Copyright (C) 2021-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals and functionals classes."""

import sys

# isort: off
from ._functional import (
    Functional,
    ComposedFunctional,
    ScaledFunctional,
    SeparableFunctional,
    ZeroFunctional,
)
from ._norm import (
    HuberNorm,
    L0Norm,
    L1Norm,
    SquaredL2Norm,
    L2Norm,
    L21Norm,
    NuclearNorm,
    L1MinusL2Norm,
)
from ._tvnorm import AnisotropicTVNorm, IsotropicTVNorm, TVNorm
from ._proxavg import ProximalAverage
from ._indicator import NonNegativeIndicator, L2BallIndicator
from ._denoiser import BM3D, BM4D, DnCNN
from ._dist import SetDistance, SquaredSetDistance


__all__ = [
    "AnisotropicTVNorm",
    "IsotropicTVNorm",
    "TVNorm",
    "Functional",
    "ComposedFunctional",
    "ScaledFunctional",
    "SeparableFunctional",
    "ZeroFunctional",
    "HuberNorm",
    "L0Norm",
    "L1Norm",
    "SquaredL2Norm",
    "L2Norm",
    "L21Norm",
    "L1MinusL2Norm",
    "NonNegativeIndicator",
    "NuclearNorm",
    "L2BallIndicator",
    "ProximalAverage",
    "SetDistance",
    "SquaredSetDistance",
    "BM3D",
    "BM4D",
    "DnCNN",
]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
