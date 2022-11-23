# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Non-linear Proximal ADMM solver and support functions."""

import sys

from ._nlpadmm import NonLinearPADMM, estimate_parameters

__all__ = [
    "NonLinearPADMM",
    "estimate_parameters",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
