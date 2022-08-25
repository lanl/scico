# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator functions and classes."""

import sys

from ._circconv import CircularConvolve
from ._convolve import Convolve, ConvolveByX
from ._dft import DFT
from ._diff import FiniteDifference, SingleAxisFiniteDifference
from ._func import Pad, Reshape, Sum, Transpose, linop_from_function
from ._linop import ComposedLinearOperator, LinearOperator
from ._matrix import MatrixOperator
from ._misc import Diagonal, Identity, Slice
from ._stack import DiagonalStack, VerticalStack
from ._util import operator_norm, power_iteration, valid_adjoint

__all__ = [
    "CircularConvolve",
    "Convolve",
    "DFT",
    "Diagonal",
    "FiniteDifference",
    "SingleAxisFiniteDifference",
    "Identity",
    "VerticalStack",
    "DiagonalStack",
    "MatrixOperator",
    "Pad",
    "Reshape",
    "Slice",
    "Sum",
    "Transpose",
    "LinearOperator",
    "ComposedLinearOperator",
    "linop_from_function",
    "operator_norm",
    "power_iteration",
    "valid_adjoint",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
