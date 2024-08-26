#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Large-scale CT Projection
=========================

This example demonstrates using SCICO's X-ray projector on a large-scale
volume.

"""

import numpy as np

import jax

from scico.examples import create_block_phantom
from scico.linop import Parallel3dProjector

N = 1000
num_views = 10

in_shape = (N, N, N)
x = create_block_phantom(in_shape)

det_shape = (N, N)

rot_X = 90.0 - 16.0
rot_Y = np.linspace(0, 180, num_views, endpoint=False)
angles = np.stack(np.broadcast_arrays(rot_X, rot_Y), axis=-1)
matrices = Parallel3dProjector.matrices_from_euler_angles(
    in_shape, det_shape, "XY", angles, degrees=True
)


H = Parallel3dProjector(in_shape, matrices, det_shape)

proj = H @ x
jax.block_until_ready(proj)
