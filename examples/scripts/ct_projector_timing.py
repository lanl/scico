#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


import time

import jax
import jax.numpy as jnp
from xdesign import Foam, discrete_phantom

from scico.linop import XRayProject, ParallelFixedAxis2dProjector
from scico.linop.radon_astra import TomographicProjector
from scico.util import Timer


N = 512
num_angles = 512

det_count = int(jnp.ceil(jnp.sqrt(2 * N**2)))

x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)

angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)

method_names = ["scico", "astra"]
timer = Timer(
    [n + "_init" for n in method_names]
    + [n + "_first_proj" for n in method_names]
    + [n + "_avg_proj" for n in method_names]
)

projectors = {}
timer.start("scico_init")
projectors["scico"] = XRayProject(ParallelFixedAxis2dProjector((N, N), angles))
timer.stop("scico_init")

timer.start("astra_init")
projectors["astra"] = TomographicProjector(
    (N, N), detector_spacing=1.0, det_count=det_count, angles=angles
)
timer.stop("astra_init")

ys = {}
for name, H in projectors.items():
    timer_label = f"{name}_first_proj"
    timer.start(timer_label)
    ys[name] = H @ x_gt
    timer.stop(timer_label)


num_repeats = 3
for name, H in projectors.items():
    timer_label = f"{name}_avg_proj"
    timer.start(timer_label)
    for _ in range(num_repeats):
        ys[name] = H @ x_gt
    timer.stop(timer_label)
    timer.td[timer_label] /= num_repeats


print(timer)

"""
with way 2:
Label               Accum.       Current
-------------------------------------------
astra_avg_proj      7.30e-01 s   Stopped
astra_first_proj    7.41e-01 s   Stopped
astra_init          4.63e-03 s   Stopped
scico_avg_proj      9.96e-01 s   Stopped
scico_first_proj    9.98e-01 s   Stopped
scico_init          8.02e+00 s   Stopped
"""

fig, ax = plt.subplots()
ax.imshow(ys["scico"])
fig.show()

fig, ax = plt.subplots()
ax.imshow(ys["astra"])
fig.show()
