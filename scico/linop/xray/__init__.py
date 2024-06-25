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

.. plot:: figures/xray_2d_geom.py
   :align: center
   :include-source: False
   :show-source-link: False
   :caption: Comparison of 2D X-ray projector geometries. The red arrows
      are are directed towards the detector, which is oriented with pixel
      indices ordered in the same direction as clockwise rotation (e.g.
      in the "scico" geometry, the :math:`\theta=0` projection
      corresponds to row sums ordered from the top to the bottom of the
      figure, while the :math:`\theta=\pi` projection
      corresponds to row sums ordered from the bottom to the top of the
      figure).

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

from ._xray import Parallel2dProjector, Parallel3dProjector, XRayTransform

__all__ = [
    "Parallel2dProjector",
    "Parallel3dProjector",
    "XRayTransform",
]
