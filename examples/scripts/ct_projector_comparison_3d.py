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
from jax.typing import ArrayLike

import scico.linop.xray.astra as astra
from scico import plot
from scico.examples import create_block_phantom
from scico.linop import Parallel3dProjector, XRayTransform
from scico.typing import Shape
from scico.util import Timer

"""
Create a ground truth image.
"""
N = 64
in_shape = (N // 2, N + 1, N + 2)

det_count = int(jnp.ceil(jnp.sqrt(3) * N))
# out_shape = (det_count, det_count + 1)
out_shape = (det_count, det_count * 2)

x_gt = create_block_phantom(in_shape)
x_gt = jnp.array(x_gt)


"""
Project with SCICO
"""
from scipy.spatial.transform import Rotation

num_angles = 7

# projection matrix: rotation matrix, chop off last row...
rot_X = 90.0 - 16.0
rot_Y = np.linspace(0, 180, num_angles, endpoint=False)
P = jnp.stack([Rotation.from_euler("XY", [rot_X, y], degrees=True).as_matrix() for y in rot_Y])
P = P[:, :2, :]

# add translation
x0 = jnp.array(in_shape) / 2
t = -jnp.tensordot(P, x0, axes=[2, 0]) + jnp.array(out_shape) / 2
P = jnp.concatenate((P, t[..., np.newaxis]), axis=2)


def P_to_vectors(in_shape: Shape, P: ArrayLike, det_shape: Shape) -> ArrayLike:
    """
    For 3D arrays,
    in Astra, the dimensions go (slices, rows, columns) and (z, y, x);
    in SCICO, the dimensions go (x, y, z).

    For 2D arrays,
    in Astra, the dimensions go (rows, columns) and (y, x);
    in SCICO, the dimensions go (x, y).

    In Astra, the x-grid (recon) is centered on the origin and the y-grid (projection) can move.
    In SCICO, the x-grid origin is x[0, 0, 0], the y-grid origin is y[0, 0].

    See https://astra-toolbox.com/docs/geom3d.html#projection-geometries parallel3d_vec.
    """
    # TODO add tests
    # ray is perpendicular to projection axes
    ray = np.cross(P[:, 0, :3], P[:, 1, :3])
    # detector center comes from lifting the center index to 3D
    y_center = np.array(det_shape) / 2
    x_center = np.einsum("...mn,n->...m", P[..., :3], np.array(in_shape) / 2) + P[..., 3]
    d = np.einsum("...mn,...m->...n", P[..., :3], y_center - x_center)  # (V, 2, 3) x (V, 2)
    u = P[:, 1, :3]
    v = P[:, 0, :3]
    vectors = np.concatenate((ray, d, u, v), axis=1)  # (v, 12)
    return vectors


def astra_to_scico(vol_geom, proj_geom):
    # TODO add tests
    in_shape = (vol_geom["GridColCount"], vol_geom["GridRowCount"], vol_geom["GridSliceCount"])
    det_shape = (proj_geom["DetectorRowCount"], proj_geom["DetectorColCount"])
    vectors = proj_geom["Vectors"]
    _, d, u, v = vectors[:, 0:3], vectors[:, 3:6], vectors[:, 6:9], vectors[:, 9:12]
    P = np.stack((v, u), axis=1)
    center_diff = np.einsum("...mn,...n->...m", P, d)  # y_center - x_center
    y_center = np.array(det_shape) / 2
    Px_center_t = -(center_diff - y_center)
    Px_center = np.einsum("...mn,n->...m", P, np.array(in_shape) / 2)
    t = Px_center_t - Px_center
    P = np.concatenate((P, t[..., np.newaxis]), axis=2)
    return P


P_to_astra_vectors = P_to_vectors(in_shape, P, out_shape)
nx, ny, nz = in_shape
n_rows = ny
n_cols = nx
n_slices = nz

vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices)
proj_geom = astra.create_proj_geom("parallel3d_vec", out_shape[0], out_shape[1], P_to_astra_vectors)

P_from_astra = astra_to_scico(vol_geom, proj_geom)


import astra

proj_id, proj_data = astra.create_sino3d_gpu(
    np.array(x_gt).transpose(2, 1, 0), proj_geom, vol_geom
)  # [row, view, col]
np.save("astra.npy", proj_data.transpose(1, 0, 2))


timer = Timer()

projectors = {}
timer.start("scico_init")
projectors["scico"] = XRayTransform(Parallel3dProjector(in_shape, P, out_shape))
timer.stop("scico_init")


proj = projectors["scico"] @ x_gt
np.save("scico.npy", proj)

viewer.layers.clear()
proj_scico = np.load("temp/scico.npy")
proj_astra = np.load("temp/astra.npy")
viewer.add_image(proj_astra)
viewer.add_image(proj_scico, opacity=0.5, colormap="blue")

exit

timer.start("astra_init")
projectors["astra"] = astra.XRayTransform2D(
    (N, N), det_count=det_count, det_spacing=1.0, angles=angles - jnp.pi / 2.0
)
timer.stop("astra_init")


"""
Time first projector application, which might include JIT overhead.
"""
ys = {}
for name, H in projectors.items():
    timer_label = f"{name}_first_fwd"
    timer.start(timer_label)
    ys[name] = H @ x_gt
    jax.block_until_ready(ys[name])
    timer.stop(timer_label)


"""
Compute average time for a projector application.
"""
num_repeats = 3
for name, H in projectors.items():
    timer_label = f"{name}_avg_fwd"
    timer.start(timer_label)
    for _ in range(num_repeats):
        ys[name] = H @ x_gt
        jax.block_until_ready(ys[name])
    timer.stop(timer_label)
    timer.td[timer_label] /= num_repeats


"""
Time first back projection, which might include JIT overhead.
"""
y = np.zeros(H.output_shape, dtype=np.float32)
y[num_angles // 3, det_count // 2] = 1.0
y = jnp.array(y)

HTys = {}
for name, H in projectors.items():
    timer_label = f"{name}_first_back"
    timer.start(timer_label)
    HTys[name] = H.T @ y
    jax.block_until_ready(ys[name])
    timer.stop(timer_label)


"""
Compute average time for back projection.
"""
num_repeats = 3
for name, H in projectors.items():
    timer_label = f"{name}_avg_back"
    timer.start(timer_label)
    for _ in range(num_repeats):
        HTys[name] = H.T @ y
        jax.block_until_ready(ys[name])
    timer.stop(timer_label)
    timer.td[timer_label] /= num_repeats


"""
Display timing results.

On our server, when using the GPU, the SCICO projector (both forward
and backward) is faster than ASTRA. When using the CPU, it is slower
for forward projection and faster for back projection. The SCICO object
initialization and first back projection are slow due to JIT
overhead.

On our server, using the GPU:
```
init         astra    4.81e-02 s
init         scico    2.53e-01 s

first  fwd   astra    4.44e-02 s
first  fwd   scico    2.82e-02 s

first  back  astra    3.31e-02 s
first  back  scico    2.80e-01 s

avg    fwd   astra    4.76e-02 s
avg    fwd   scico    2.83e-02 s

avg    back  astra    3.96e-02 s
avg    back  scico    6.80e-04 s
```

Using the CPU:
```
init         astra    1.72e-02 s
init         scico    2.88e+00 s

first  fwd   astra    1.02e+00 s
first  fwd   scico    2.40e+00 s

first  back  astra    1.03e+00 s
first  back  scico    3.53e+00 s

avg    fwd   astra    1.03e+00 s
avg    fwd   scico    2.54e+00 s

avg    back  astra    1.01e+00 s
avg    back  scico    5.98e-01 s
```
"""
print(f"init         astra    {timer.td['astra_init']:.2e} s")
print(f"init         scico    {timer.td['scico_init']:.2e} s")
print("")
for tstr in ("first", "avg"):
    for dstr in ("fwd", "back"):
        for pstr in ("astra", "scico"):
            print(
                f"{tstr:5s}  {dstr:4s}  {pstr}    {timer.td[pstr + '_' + tstr + '_' + dstr]:.2e} s"
            )
        print()


"""
Show projections.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 6))
plot.imview(ys["scico"], title="SCICO projection", cbar=None, fig=fig, ax=ax[0])
plot.imview(ys["astra"], title="ASTRA projection", cbar=None, fig=fig, ax=ax[1])
fig.show()


"""
Show back projections of a single detector element, i.e., a line.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 6))
plot.imview(HTys["scico"], title="SCICO back projection (zoom)", cbar=None, fig=fig, ax=ax[0])
plot.imview(HTys["astra"], title="ASTRA back projection (zoom)", cbar=None, fig=fig, ax=ax[1])
for ax_i in ax:
    ax_i.set_xlim(2 * N / 5, N - 2 * N / 5)
    ax_i.set_ylim(2 * N / 5, N - 2 * N / 5)
fig.show()


input("\nWaiting for input to close figures and exit")
