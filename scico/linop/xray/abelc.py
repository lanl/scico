# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Cone beam Abel transform LinearOperator.

Cone beam Abel transform :class:`.LinearOperator` based on code
modified from the `axitom <https://github.com/PolymerGuy/AXITOM>`_
package.
"""

from typing import Tuple

import numpy as np

from scico.typing import Shape

from .._linop import LinearOperator
from ._axitom import config, project


class AbelTransformCone(LinearOperator):
    r"""Cone beam Abel transform.

    Compute cone beam Abel transform of a cylindricaly symmetric volume
    represented by a 2D central slice, which is rotated about axis 1 to
    generate a 3D volume for projection. The implementation is based on
    code modified from the `axitom <https://github.com/PolymerGuy/AXITOM>`_
    package.
    """

    def __init__(
        self,
        output_shape: Shape,
        det_size: Tuple[float, float],
        obj_dist: float,
        det_dist: float,
        sym_center: float = 0,
    ):
        """
        Args:
            output_shape: Shape of the output array (projection).
            det_size: Tuple of detector size values in mm.
            obj_dist: Source-object distance in mm.
            det_dist: Source-detector distance in mm.
            sym_center: Position of the rotation axis in pixels, with 0
              corresponding to the center of the image.
        """
        self.config = config.Config(*output_shape, *det_size, det_dist, obj_dist, sym_center)
        input_shape = (self.config.detector_us.size, self.config.detector_vs.size)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=lambda x: project._forward_project(x, self.config),
            jit=True,
        )
