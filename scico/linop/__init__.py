# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator functions and classes."""

import sys

from ._circconv import CircularConvolve
from ._convolve import Convolve, ConvolveByX
from ._dft import DFT
from ._diag import Diagonal, Identity, ScaledIdentity
from ._diff import FiniteDifference, SingleAxisFiniteDifference
from ._func import Crop, Pad, Reshape, Slice, Sum, Transpose, linop_from_function
from ._grad import (
    CylindricalGradient,
    PolarGradient,
    ProjectedGradient,
    SphericalGradient,
)
from ._linop import ComposedLinearOperator, LinearOperator
from ._matrix import MatrixOperator
from ._stack import DiagonalReplicated, DiagonalStack, VerticalStack, linop_over_axes
from ._util import jacobian, operator_norm, power_iteration, valid_adjoint

__all__ = [
    "CircularConvolve",
    "Convolve",
    "DFT",
    "Diagonal",
    "FiniteDifference",
    "ProjectedGradient",
    "PolarGradient",
    "CylindricalGradient",
    "SphericalGradient",
    "SingleAxisFiniteDifference",
    "Identity",
    "DiagonalReplicated",
    "VerticalStack",
    "DiagonalStack",
    "MatrixOperator",
    "Pad",
    "Crop",
    "Reshape",
    "ScaledIdentity",
    "Slice",
    "Sum",
    "Transpose",
    "LinearOperator",
    "ComposedLinearOperator",
    "linop_from_function",
    "linop_over_axes",
    "operator_norm",
    "power_iteration",
    "valid_adjoint",
    "jacobian",
]

# Imported items in __all__ appear to originate in top-level linop module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
