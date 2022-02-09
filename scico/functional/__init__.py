# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionals and functionals classes."""

import sys

# isort: off
from ._functional import Functional, ScaledFunctional, SeparableFunctional, ZeroFunctional
from ._norm import L0Norm, L1Norm, SquaredL2Norm, L2Norm, L21Norm, NuclearNorm
from ._indicator import NonNegativeIndicator, L2BallIndicator
from ._denoiser import BM3D, DnCNN


__all__ = [
    "Functional",
    "ScaledFunctional",
    "SeparableFunctional",
    "ZeroFunctional",
    "L0Norm",
    "L1Norm",
    "SquaredL2Norm",
    "L2Norm",
    "L21Norm",
    "NonNegativeIndicator",
    "NuclearNorm",
    "L2BallIndicator",
    "BM3D",
    "DnCNN",
]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
