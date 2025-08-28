# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r"""X-ray transform classes.

The tomographic projections that are frequently referred to as Radon
transforms are referred to as X-ray transforms in SCICO. While the Radon
transform is far more well-known than the X-ray transform, which is the
same as the Radon transform for projections in two dimensions, these two
transform differ in higher numbers of dimensions, and it is the X-ray
transform that is the appropriate mathematical model for beam attenuation
based imaging in three or more dimensions.

SCICO includes its own integrated 2D and 3D X-ray transforms, and also
provides interfaces to those implemented in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_
and the `svmbir <https://github.com/cabouman/svmbir>`_ package.


**2D Transforms**

The SCICO, ASTRA, and svmbir transforms use different conventions for
view angle directions, as illustrated in the figure below.

.. plot:: pyfigures/xray_2d_geom.py
   :align: center
   :include-source: False
   :show-source-link: False
   :caption: Comparison of 2D X-ray projector geometries. The radial
      arrows are directed towards the locations of the corresponding
      detectors, with the direction of increasing pixel indices indicated
      by the arrows on the dotted lines parallel to the detectors.

|

The conversion from the SCICO projection angle convention to those of the
other two transforms is

.. math::

   \begin{aligned}
   \theta_{\text{astra}} &= \theta_{\text{scico}} - \frac{\pi}{2} \\
   \theta_{\text{svmbir}} &= 2 \pi - \theta_{\text{scico}} \;.
   \end{aligned}


**3D Transforms**

There are more significant differences in the interfaces for the 3D SCICO
and ASTRA transforms. The SCICO 3D transform :class:`.xray.XRayTransform3D`
defines the projection geometry in terms of a set of projection matrices,
while the geometry for the ASTRA 3D transform
:class:`.astra.XRayTransform3D` may either be specified in terms of a set
of view angles, or via a more general set of vectors specifying projection
direction and detector orientation. A number of support functions are
provided for convering between these conventions.

Note that the SCICO transform is implemented in JAX and can be run on
both CPU and GPU devices, while the ASTRA transform is implemented in
CUDA, and can only be run on GPU devices.
"""

import sys

from ._util import (
    center_image,
    image_alignment_rotation,
    image_centroid,
    rotate_volume,
    volume_alignment_rotation,
)
from ._xray import XRayTransform2D, XRayTransform3D

__all__ = [
    "XRayTransform2D",
    "XRayTransform3D",
    "image_centroid",
    "center_image",
    "rotate_volume",
    "image_alignment_rotation",
    "volume_alignment_rotation",
]


# Imported items in __all__ appear to originate in top-level xray module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
