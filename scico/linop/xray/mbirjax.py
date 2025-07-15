# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform LinearOperator wrapping the mbirjax package.

X-ray transform :class:`.LinearOperator` wrapping the
`mbirjax <https://github.com/cabouman/mbirjax>`_ package..
"""


import numpy as np

import scico.numpy as snp
from scico.typing import Shape

from .._linop import LinearOperator

try:
    # Attempt to undo mbirjax interference in matplotlib backend
    import matplotlib

    mpl_backend = matplotlib.get_backend()
    import mbirjax

    matplotlib.use(mpl_backend)
except ImportError:
    raise ImportError("Could not import mbirjax; please install it.")


class XRayTransformParallel(LinearOperator):
    r"""Parallel beam X-ray transform based on mbirjax.

    Perform parallel beam tomographic projection of an image at specified
    angles, using the `mbirjax <https://github.com/cabouman/mbirjax>`_
    package.
    """

    def __init__(self, output_shape: Shape, angles: snp.Array, jit: bool = False, **kwargs):
        """
        Args:
            output_shape: Shape of the output array (sinogram).
            angles: Array of projection angles in radians, should be
                increasing.
            jit: If ``True``, call :meth:`.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling :meth:`.jit` after the
                :class:`LinearOperator` is created.
        """
        self.model = mbirjax.ParallelBeamModel(output_shape, angles)
        self.model.set_params(no_warning=False, no_compile=False, **kwargs)
        input_shape = self.model.get_params("recon_shape")
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=self.model.forward_project,
            adj_fn=self.model.back_project,
            jit=jit,
        )


class XRayTransformCone(LinearOperator):
    r"""Cone beam X-ray transform based on mbirjax.

    Perform cone beam tomographic projection of an image at specified
    angles, using the `mbirjax <https://github.com/cabouman/mbirjax>`_
    package.
    """

    def __init__(
        self,
        output_shape: Shape,
        angles: snp.Array,
        iso_dist: float,
        det_dist: float,
        jit: bool = False,
        **kwargs,
    ):
        """
        Args:
            output_shape: Shape of the output array (sinogram).
            angles: Array of projection angles in radians, should be
                increasing.
            iso_dist: Distance in arbitrary length units (ALU) from
                source to imaging isocenter.
            det_dist: Distance in arbitrary length units (ALU) from
                source to detector.
            jit: If ``True``, call :meth:`.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling :meth:`.jit` after the
                :class:`LinearOperator` is created.
        """
        self.model = mbirjax.ConeBeamModel(output_shape, angles, det_dist, iso_dist)
        self.model.set_params(no_warning=False, no_compile=False, **kwargs)
        input_shape = self.model.get_params("recon_shape")
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=self.model.forward_project,
            adj_fn=self.model.back_project,
            jit=jit,
        )
