# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator functions and classes."""

import sys

# isort: off
from scico._generic_operators import LinearOperator
from ._linop import Diagonal, Identity, power_iteration, operator_norm, Sum, Slice, valid_adjoint
from ._matrix import MatrixOperator
from ._diff import FiniteDifference
from ._convolve import Convolve, ConvolveByX
from ._circconv import CircularConvolve
from ._dft import DFT
from ._stack import LinearOperatorStack


__all__ = [
    "LinearOperator",
    "Identity",
    "Diagonal",
    "MatrixOperator",
    "FiniteDifference",
    "Convolve",
    "CircularConvolve",
    "DFT",
    "LinearOperatorStack",
    "Sum",
    "Slice",
    "power_iteration",
    "operator_norm",
    "valid_adjoint",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
