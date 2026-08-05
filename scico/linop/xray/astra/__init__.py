# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""LinearOperators wrapping the ASTRA toolbox X-ray transforms.

X-ray transform :class:`.LinearOperator` classes wrapping the X-ray
projections in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
This package provides both C and CUDA implementations of core
functionality, but note that use of the CUDA/GPU implementation
involves GPU-host-GPU memory copies when transferring JAX arrays. Other
JAX features such as automatic differentiation are not available.
"""

import sys
from typing import Sequence, Union

try:
    import astra
except ModuleNotFoundError as e:
    if e.name == "astra":
        new_e = ModuleNotFoundError("Could not import astra; please install the ASTRA toolbox.")
        new_e.name = "astra"
        raise new_e from e
    else:
        raise e


def set_astra_gpu_index(idx: Union[int, Sequence[int]]):
    """Set the index/indices of GPU(s) to be used by astra.

    Args:
        idx: Index or indices of GPU(s).
    """
    astra.set_gpu_index(idx)


from ._astra_2d import XRayTransform2D
from ._astra_3d import (
    XRayTransform3D,
    angle_to_vector,
    convert_from_scico_geometry,
    convert_to_scico_geometry,
    rotate_vectors,
    volume_coords_to_world_coords,
)
from ._astra_cone import XRayTransform3DCone, angle_to_vector_cone

__all__ = [
    "set_astra_gpu_index",
    "convert_from_scico_geometry",
    "convert_to_scico_geometry",
    "volume_coords_to_world_coords",
    "angle_to_vector",
    "rotate_vectors",
    "angle_to_vector_cone",
    "XRayTransform2D",
    "XRayTransform3D",
    "XRayTransform3DCone",
]


# Imported items in __all__ appear to originate in top-level astra module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
