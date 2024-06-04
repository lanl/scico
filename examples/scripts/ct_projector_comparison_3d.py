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

timer = Timer()

timer.start("scico_init")
H_scico = XRayTransform(Parallel3dProjector(in_shape, P, out_shape))
timer.stop("scico_init")


timer.start("scico_first_fwd")
y_scico = H_scico @ x
jax.block_until_ready(y_scico)
timer.stop("scico_first_fwd")

timer.start("scico_first_fwd")
y_scico = H_scico @ x
timer.stop("scico_first_fwd")

timer.start("scico_avg_fwd")
for _ in range(num_repeats):
    y_scico = H_scico @ x
    jax.block_until_ready(y_scico)
timer.stop("scico_avg_fwd")
timer.td["scico_avg_fwd"] /= num_repeats


"""
Convert SCICO geometry to ASTRA and project
"""

P_to_astra_vectors = scico.linop.xray.P_to_vectors(in_shape, P, out_shape)

timer.start("astra_init")
H_astra_from_scico = astra.XRayTransform3D(
    input_shape=in_shape, det_count=out_shape, vectors=P_to_astra_vectors
)
timer.stop("astra_init")


"""
Specify geometry with ASTRA conventions and project
"""

angles = np.linspace(0, np.pi, num_angles)  # evenly spaced projection angles
det_spacing = [1.0, 1.0]
vectors = astra.angle_to_vector(det_spacing, angles)

H_astra = astra.XRayTransform3D(input_shape=in_shape, det_count=out_shape, vectors=vectors)

"""
Convert ASTRA geometry to SCICO and project
"""

P_from_astra = scico.linop.xray.astra_to_scico(H_astra.vol_geom, H_astra.proj_geom)
H_scico_from_astra = XRayTransform(Parallel3dProjector(in_shape, P_from_astra, out_shape))


# """
# Time first back projection, which might include JIT overhead.
# """
# y = np.zeros(H.output_shape, dtype=np.float32)
# y[num_angles // 3, det_count // 2] = 1.0
# y = jnp.array(y)

# HTys = {}
# for name, H in projectors.items():
#     timer_label = f"{name}_first_back"
#     timer.start(timer_label)
#     HTys[name] = H.T @ y
#     jax.block_until_ready(ys[name])
#     timer.stop(timer_label)


# """
# Compute average time for back projection.
# """
# num_repeats = 3
# for name, H in projectors.items():
#     timer_label = f"{name}_avg_back"
#     timer.start(timer_label)
#     for _ in range(num_repeats):
#         HTys[name] = H.T @ y
#         jax.block_until_ready(ys[name])
#     timer.stop(timer_label)
#     timer.td[timer_label] /= num_repeats


# """
# Display timing results.

# On our server, when using the GPU, the SCICO projector (both forward
# and backward) is faster than ASTRA. When using the CPU, it is slower
# for forward projection and faster for back projection. The SCICO object
# initialization and first back projection are slow due to JIT
# overhead.

# On our server, using the GPU:
# ```
# init         astra    4.81e-02 s
# init         scico    2.53e-01 s

# first  fwd   astra    4.44e-02 s
# first  fwd   scico    2.82e-02 s

# first  back  astra    3.31e-02 s
# first  back  scico    2.80e-01 s

# avg    fwd   astra    4.76e-02 s
# avg    fwd   scico    2.83e-02 s

# avg    back  astra    3.96e-02 s
# avg    back  scico    6.80e-04 s
# ```

# Using the CPU:
# ```
# init         astra    1.72e-02 s
# init         scico    2.88e+00 s

# first  fwd   astra    1.02e+00 s
# first  fwd   scico    2.40e+00 s

# first  back  astra    1.03e+00 s
# first  back  scico    3.53e+00 s

# avg    fwd   astra    1.03e+00 s
# avg    fwd   scico    2.54e+00 s

# avg    back  astra    1.01e+00 s
# avg    back  scico    5.98e-01 s
# ```
# """
# print(f"init         astra    {timer.td['astra_init']:.2e} s")
# print(f"init         scico    {timer.td['scico_init']:.2e} s")
# print("")
# for tstr in ("first", "avg"):
#     for dstr in ("fwd", "back"):
#         for pstr in ("astra", "scico"):
#             print(
#                 f"{tstr:5s}  {dstr:4s}  {pstr}    {timer.td[pstr + '_' + tstr + '_' + dstr]:.2e} s"
#             )
#         print()


"""
Show projections.
"""
fig, ax = plot.subplots(nrows=3, ncols=2, figsize=(8, 6))
plot.imview(y_scico[0], title="SCICO projections", cbar=None, fig=fig, ax=ax[0, 0])
plot.imview(y_scico[2], cbar=None, fig=fig, ax=ax[1, 0])
plot.imview(y_scico[4], cbar=None, fig=fig, ax=ax[2, 0])
plot.imview(y_scico[0], title="ASTRA projections", cbar=None, fig=fig, ax=ax[0, 1])  # TODO fix
plot.imview(y_scico[1], cbar=None, fig=fig, ax=ax[1, 1])
plot.imview(y_scico[2], cbar=None, fig=fig, ax=ax[2, 1])
fig.suptitle("Using SCICO conventions")
fig.tight_layout()
fig.show()

fig, ax = plot.subplots(nrows=3, ncols=2, figsize=(8, 6))
plot.imview(y_scico[0], title="SCICO projections", cbar=None, fig=fig, ax=ax[0, 0])
plot.imview(y_scico[1], cbar=None, fig=fig, ax=ax[1, 0])
plot.imview(y_scico[2], cbar=None, fig=fig, ax=ax[2, 0])
plot.imview(y_scico[0], title="ASTRA projections", cbar=None, fig=fig, ax=ax[0, 1])  # TODO fix
plot.imview(y_scico[1], cbar=None, fig=fig, ax=ax[1, 1])
plot.imview(y_scico[2], cbar=None, fig=fig, ax=ax[2, 1])
fig.suptitle("Using ASTRA conventions")
fig.tight_layout()
fig.show()


input("\nWaiting for input to close figures and exit")
