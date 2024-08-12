# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 by SCICO Developers
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

SCICO includes its own integrated 2D X-ray transform, and also provides
interfaces to those implemented in the
`ASTRA toolbox <https://github.com/astra-toolbox/astra-toolbox>`_
and the `svmbir <https://github.com/cabouman/svmbir>`_ package. Each of
these transforms uses a different convention for view angle directions,
as illustrated in the figure below.

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

"""

import sys

from ._xray import Parallel2dProjector, XRayTransform

__all__ = [
    "XRayTransform",
    "Parallel2dProjector",
]
