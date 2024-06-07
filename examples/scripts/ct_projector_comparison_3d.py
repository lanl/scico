#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


r"""
X-ray Transform Comparison in 3D
================================

This example compares SCICO's native 3D X-ray transform algorithm
to that of the ASTRA toolbox.
"""

import numpy as np

import jax
import jax.numpy as jnp

import scico.linop.xray
import scico.linop.xray.astra as astra
from scico import plot
from scico.examples import create_block_phantom
from scico.linop import Parallel3dProjector, XRayTransform
from scico.util import Timer
from scipy.spatial.transform import Rotation

"""
Create a ground truth image and set detector dimensions.
"""
N = 64
# use rectangular volume to check whether it is handled correctly
in_shape = (N + 1, N + 2, N + 3)
x = create_block_phantom(in_shape)
x = jnp.array(x)

diagonal_length = int(jnp.ceil(jnp.sqrt(3) * N))
# use rectangular detector to check whether it is handled correctly
out_shape = (diagonal_length, diagonal_length + 1)


"""
Set up SCICO projection
"""
num_angles = 7

# make projection matrix: form a rotation matrix and chop off the last row
rot_X = 90.0 - 16.0
rot_Y = np.linspace(0, 180, num_angles, endpoint=False)
P = jnp.stack([Rotation.from_euler("XY", [rot_X, y], degrees=True).as_matrix() for y in rot_Y])
P = P[:, :2, :]

# add translation
x0 = jnp.array(in_shape) / 2
t = -jnp.tensordot(P, x0, axes=[2, 0]) + jnp.array(out_shape) / 2
P = jnp.concatenate((P, t[..., np.newaxis]), axis=2)


"""
Specify geometry using SCICO conventions and project
"""
num_repeats = 3

timer_scico = Timer()
timer_scico.start("init")
H_scico = XRayTransform(Parallel3dProjector(in_shape, P, out_shape))
timer_scico.stop("init")

timer_scico.start("first_fwd")
y_scico = H_scico @ x
jax.block_until_ready(y_scico)
timer_scico.stop("first_fwd")

timer_scico.start("first_fwd")
y_scico = H_scico @ x
timer_scico.stop("first_fwd")

timer_scico.start("avg_fwd")
for _ in range(num_repeats):
    y_scico = H_scico @ x
    jax.block_until_ready(y_scico)
timer_scico.stop("avg_fwd")
timer_scico.td["avg_fwd"] /= num_repeats


"""
Convert SCICO geometry to ASTRA and project
"""

P_to_astra_vectors = scico.linop.xray.P_to_vectors(in_shape, P, out_shape)

timer_astra = Timer()
timer_astra.start("astra_init")
H_astra_from_scico = astra.XRayTransform3D(
    input_shape=in_shape, det_count=out_shape, vectors=P_to_astra_vectors
)
timer_astra.stop("astra_init")

timer_astra.start("first_fwd")
y_astra_from_scico = H_astra_from_scico @ x
jax.block_until_ready(y_scico)
timer_astra.stop("first_fwd")

timer_astra.start("first_fwd")
y_astra_from_scico = H_scico @ x
timer_astra.stop("first_fwd")

timer_astra.start("avg_fwd")
for _ in range(num_repeats):
    y_astra_from_scico = H_scico @ x
    jax.block_until_ready(y_astra_from_scico)
timer_astra.stop("avg_fwd")
timer_astra.td["avg_fwd"] /= num_repeats


"""
Specify geometry with ASTRA conventions and project
"""

angles = np.linspace(0, np.pi, num_angles)  # evenly spaced projection angles
det_spacing = [1.0, 1.0]
vectors = astra.angle_to_vector(det_spacing, angles)

H_astra = astra.XRayTransform3D(input_shape=in_shape, det_count=out_shape, vectors=vectors)

y_astra = H_astra @ x

"""
Convert ASTRA geometry to SCICO and project
"""

P_from_astra = scico.linop.xray.astra_to_scico(H_astra.vol_geom, H_astra.proj_geom)
H_scico_from_astra = XRayTransform(Parallel3dProjector(in_shape, P_from_astra, out_shape))

y_scico_from_astra = H_scico_from_astra @ x

"""
Show projections.
"""
fig, ax = plot.subplots(nrows=3, ncols=2, figsize=(8, 6))
plot.imview(y_scico[0], title="SCICO projections", cbar=None, fig=fig, ax=ax[0, 0])
plot.imview(y_scico[2], cbar=None, fig=fig, ax=ax[1, 0])
plot.imview(y_scico[4], cbar=None, fig=fig, ax=ax[2, 0])
plot.imview(y_astra_from_scico[0], title="ASTRA projections", cbar=None, fig=fig, ax=ax[0, 1])
plot.imview(y_astra_from_scico[2], cbar=None, fig=fig, ax=ax[1, 1])
plot.imview(y_astra_from_scico[4], cbar=None, fig=fig, ax=ax[2, 1])
fig.suptitle("Using SCICO conventions")
fig.tight_layout()
fig.show()

fig, ax = plot.subplots(nrows=3, ncols=2, figsize=(8, 6))
plot.imview(y_scico_from_astra[0], title="SCICO projections", cbar=None, fig=fig, ax=ax[0, 0])
plot.imview(y_scico_from_astra[2], cbar=None, fig=fig, ax=ax[1, 0])
plot.imview(y_scico_from_astra[4], cbar=None, fig=fig, ax=ax[2, 0])
plot.imview(y_astra[0], title="ASTRA projections", cbar=None, fig=fig, ax=ax[0, 1])
plot.imview(y_astra[2], cbar=None, fig=fig, ax=ax[1, 1])
plot.imview(y_astra[4], cbar=None, fig=fig, ax=ax[2, 1])
fig.suptitle("Using ASTRA conventions")
fig.tight_layout()
fig.show()


input("\nWaiting for input to close figures and exit")
