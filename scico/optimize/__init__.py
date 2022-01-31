# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Optimization algorithms."""

import sys

# isort: off
from .admm import ADMM
from ._ladmm import LinearizedADMM
from .pgm import PGM, AcceleratedPGM
from ._primaldual import PDHG

__all__ = ["ADMM", "LinearizedADMM", "PGM", "AcceleratedPGM", "PDHG"]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
