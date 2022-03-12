#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
CT Data Generation for NN Training
==================================

This example demonstrates how to
generate CT synthetic data for
using in training neural network
models. If desired, a basic
reconstruction can be generated
using filtered back projection (FBP).
"""

from time import time

import numpy as np
import jax
import jax.numpy as jnp

from scico import plot
from scico.linop.radon_astra import ParallelBeamProjector
from scico.examples_flax import generate_foam2_images, distributed_data_generation

"""
Prepare parallel processing. Set an
arbitrary processor count (only
applies if GPU is not available).
"""
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = jax.lib.xla_bridge.get_backend().platform
print("Platform: ", platform)


"""
Prepare output path.
"""
prefix = "./dtct/"
os.mkdir(prefix)


"""
Generate data.
"""
N = 256  # phantom size
train_nimg = 512  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg
start_time = time()
imgsshd = distributed_data_generation(generate_foam2_images, N, nimg)
nproc = jax.device_count()
time_dtgen = time() - start_time
imgs = imgsshd.reshape((-1, N, N, 1))
print(f"{'Data Generation':8s}{'time[s]:':2s}{time_dtgen:>5.2f}")

"""
Configure a CT projection operator
to generate synthetic measurements.
"""
n_projection = 180  # CT views
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
gt_sh = (N, N)
detector_spacing = 1
A = ParallelBeamProjector(gt_sh, detector_spacing, N, angles)  # Radon transform operator

"""
Compute sinograms in parallel.
"""
a_map = lambda v: jnp.atleast_3d(A @ v.squeeze())
sinoshd = jax.pmap(lambda i: jax.lax.map(a_map, imgsshd[i]))(jnp.arange(nproc))
time_sino = time() - start_time
sino = sinoshd.reshape((-1, n_projection, N, 1))
print(f"{'Sinogram Generation':8s}{'time[s]:':2s}{time_sino:>5.2f}")

"""
Compute filter back-project in
parallel.
"""
afbp_map = lambda v: jnp.atleast_3d(A.fbp(v.squeeze()))
start_time = time()
fbpshd = jax.pmap(lambda i: jax.lax.map(afbp_map, sinoshd[i]))(jnp.arange(nproc))
time_fbp = time() - start_time
fbp = fbpshd.reshape((-1, N, N, 1))
print(f"{'FBP Generation':8s}{'time[s]:':2s}{time_fbp:>5.2f}")

"""
Separate training and testing
partitions. Store images, sinograms
and filter back-projections.
"""
np.savez(
    prefix + "foam2ct_train.npz",
    img=imgs[:train_nimg],
    sino=sino[:train_nimg],
    fbp=fbp[:train_nimg],
)
np.savez(
    prefix + "foam2ct_test.npz", img=imgs[train_nimg:], sino=sino[train_nimg:], fbp=fbp[train_nimg:]
)

"""
Plot randomly selected sample.
"""
indx_tr = np.random.randint(0, train_nimg)
indx_te = np.random.randint(0, test_nimg)
fig, axes = plot.subplots(nrows=2, ncols=3, figsize=(11, 9))
plot.imview(imgs[indx_tr, ..., 0], title="Ground truth - Training Sample", fig=fig, ax=axes[0, 0])
plot.imview(sino[indx_tr, ..., 0], title="Sinogram - Training Sample", fig=fig, ax=axes[0, 1])
plot.imview(
    fbp[indx_tr, ..., 0],
    title="FBP - Training Sample",
    fig=fig,
    ax=axes[0, 2],
)
plot.imview(
    imgs[train_nimg + indx_te, ..., 0],
    title="Ground truth - Testing Sample",
    fig=fig,
    ax=axes[1, 0],
)
plot.imview(
    sino[train_nimg + indx_te, ..., 0], title="Sinogram - Testing Sample", fig=fig, ax=axes[1, 1]
)
plot.imview(
    fbp[train_nimg + indx_te, ..., 0],
    title="FBP - Testing Sample",
    fig=fig,
    ax=axes[1, 2],
)
fig.suptitle(r"Training and Testing samples")
fig.tight_layout()
fig.colorbar(
    axes[1, 0].get_images()[0],
    ax=axes,
    location="bottom",
    shrink=0.5,
    pad=0.05,
    label="Arbitrary Units",
)
fig.show()

input("\nWaiting for input to close figures and exit")
