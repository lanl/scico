# -*- coding: utf-8 -*-
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""PGM solvers and auxiliary classes."""

import sys

# isort: off
from ._pgmaux import (
    PGMStepSize,
    BBStepSize,
    AdaptiveBBStepSize,
    LineSearchStepSize,
    RobustLineSearchStepSize,
)
from ._pgm import PGM, AcceleratedPGM

__all__ = [
    "PGMStepSize",
    "BBStepSize",
    "AdaptiveBBStepSize",
    "LineSearchStepSize",
    "RobustLineSearchStepSize",
    "PGM",
    "AcceleratedPGM",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
