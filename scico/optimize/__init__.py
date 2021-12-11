# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Optimization algorithms."""

# isort: off
from _admm import ADMM
from _ladmm import LinearizedADMM
from _pgm import PGM, AcceleratedPGM
from _primaldual import PDHG
from _solver import minimize, minimize_scalar, cg

__all__ = [
    "ADMM",
    "LinearizedADMM",
    "PGM",
    "AcceleratedPGM",
    "PDHG",
    "minimize",
    "minimize_scalar",
    "cg",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
