#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.


r"""
X-ray Projector Comparison
==========================

This example compares SCICO's native X-ray projection algorithm
to that of the ASTRA Toolbox.
"""

import jax
import jax.numpy as jnp

from xdesign import Foam, discrete_phantom

from scico import plot
from scico.linop import ParallelFixedAxis2dProjector, XRayProject
from scico.linop.radon_astra import TomographicProjector
from scico.util import Timer

"""
Create a ground truth image.
"""

N = 512


det_count = int(jnp.ceil(jnp.sqrt(2 * N**2)))

x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)

"""
Time projector instantiation.
"""

num_angles = 500
angles = jnp.linspace(0, jnp.pi, num=num_angles, endpoint=False)


timer = Timer()

projectors = {}
timer.start("scico_init")
projectors["scico"] = XRayProject(ParallelFixedAxis2dProjector((N, N), angles))
timer.stop("scico_init")

timer.start("astra_init")
projectors["astra"] = TomographicProjector(
    (N, N), detector_spacing=1.0, det_count=det_count, angles=angles
)
timer.stop("astra_init")

"""
Time first projector application, which might include JIT overhead.
"""

ys = {}
for name, H in projectors.items():
    timer_label = f"{name}_first_proj"
    timer.start(timer_label)
    ys[name] = H @ x_gt
    jax.block_until_ready(ys[name])
    timer.stop(timer_label)


"""
Compute average time for a projector application.
"""

num_repeats = 3
for name, H in projectors.items():
    timer_label = f"{name}_avg_proj"
    timer.start(timer_label)
    for _ in range(num_repeats):
        ys[name] = H @ x_gt
        jax.block_until_ready(ys[name])
    timer.stop(timer_label)
    timer.td[timer_label] /= num_repeats

"""
Display timing results.

On our server, using the GPU:


Using the CPU:

"""

print(timer)

"""
Show projections.
"""

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 5))
plot.imview(ys["scico"], title="SCICO projection", cbar=None, fig=fig, ax=ax[0])
plot.imview(ys["astra"], title="ASTRA projection", cbar=None, fig=fig, ax=ax[1])
fig.show()

input("\nWaiting for input to close figures and exit")
