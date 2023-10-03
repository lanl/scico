# -*- coding: utf-8 -*-
# Copyright (C) 2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform classes.

The tomographic projections that are frequently referred to as Radon
transforms are referred to as X-ray transforms in SCICO. While the Radon
transform is far more well-known than the X-ray transform, which is the
same as the Radon transform for projections in two dimensions, these two
transform differ in higher numbers of dimensions, and it is the X-ray
transform that is the appropriate mathematical model for beam attenuation
based imaging in three or more dimensions.
"""

import sys

from ._xray import Parallel2dProjector, XRayTransform

__all__ = [
    "XRayTransform",
    "Parallel2dProjector",
]
