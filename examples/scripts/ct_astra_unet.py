#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
CT with UNet for Denoising of FBP
=================================

This example demonstrates the training and application of a UNet to denoise previously filtered back projections for CT reconstruction.
"""

from time import time

import numpy as np

import jax
import jax.numpy as jnp

from xdesign import UnitCircle, SimpleMaterial, discrete_phantom

from scico import plot
from scico.linop.radon_astra import ParallelBeamProjector
from scico import flax as sflax
from scico.metric import snr

"""
Prepare parallel processing. Set an arbitrary processor count (only applies if GPU is not available).
"""
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
platform = jax.lib.xla_bridge.get_backend().platform
print("Platform: ", platform)


"""
Generate training and testing data.
"""
N = 128  # 256  # phantom size

"""Define functionality to generate phantom with structure similar to foam with 2 different attenuation properties."""


class Foam2(UnitCircle):
    def __init__(self, size_range=[0.05, 0.01], gap=0, porosity=1):
        super(Foam2, self).__init__(radius=0.5, material=SimpleMaterial(1.0))
        if porosity < 0 or porosity > 1:
            raise ValueError("Porosity must be in the range [0,1).")
        self.sprinkle(
            300, size_range, gap, material=SimpleMaterial(10), max_density=porosity / 2.0
        ) + self.sprinkle(300, size_range, gap, material=SimpleMaterial(20), max_density=porosity)


"""
Split data generation to enable parallel processing.
"""


def generate_images(seed, size, ndata):
    # key = jax.random.PRNGKey(seed)
    saux = np.zeros((ndata, size, size, 1))
    for i in range(ndata):
        saux[i, ..., 0] = discrete_phantom(
            Foam2(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=size
        )
    return saux


"""
Configure a CT projection operator to generate synthetic measurements.
"""
n_projection = 180  # relatively few-view CT
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
gt_sh = (N, N)
# A = 1 / N * ParallelBeamProjector(gt_sh, 1, N, angles)  # Radon transform operator
A = ParallelBeamProjector(gt_sh, 1, N, angles)  # Radon transform operator

"""
Configure training and testing data generation and set number of processes for parallel execution.
"""
train_n_img = 200  # 40
test_n_img = 24
num_ims = train_n_img + test_n_img
nproc = jax.device_count()
ndata_per_proc = int(num_ims / nproc)
seeds = np.arange(nproc)

"""
Generate data.
First: generate foam images in parallel.
"""
ims = jax.pmap(generate_images, static_broadcasted_argnums=(1, 2))(seeds, N, ndata_per_proc)
ims_sharded = ims
ims = ims.reshape((-1, N, N, 1))

"""
Second: compute sinograms in parallel.
"""
a_map = lambda v: jnp.atleast_3d(A @ v.squeeze())
sinoaux = jax.pmap(lambda i: jax.lax.map(a_map, ims_sharded[i]))(np.arange(nproc))
sinoaux_sharded = sinoaux
sinoaux = sinoaux.reshape((-1, n_projection, N, 1))

"""
Third: compute filter back-project in parallel.
"""
afbp_map = lambda v: jnp.atleast_3d(A.fbp(v.squeeze()))
sfbpaux = jax.pmap(lambda i: jax.lax.map(afbp_map, sinoaux_sharded[i]))(np.arange(nproc))
sfbpaux = sfbpaux.reshape((-1, N, N, 1))

"""
Build training and testing structures. Inputs are the filter back-projected sinograms and outpus are the original generated foams. Keep training and testing partitions.
"""
train_ds = {"image": sfbpaux[:train_n_img], "label": ims[:train_n_img]}
test_ds = {"image": sfbpaux[train_n_img:], "label": ims[train_n_img:]}


"""
Define configuration dictionary for model and training loop.
"""
batch_size = 16
epochs = 200
dconf: sflax.ConfigDict = {
    "seed": 0,
    "depth": 2,
    "num_filters": 64,
    "block_depth": 3,
    "opt_type": "ADAM",
    "momentum": 0.9,
    "batch_size": batch_size,
    "num_epochs": epochs,
    "base_learning_rate": 1e-3,
    "warmup_epochs": 0,
    "num_train_steps": -1,
    "steps_per_eval": -1,
    "steps_per_epoch": 1,
    "log_every_steps": 1000,
}

"""
Construct UNet model.
"""
channels = train_ds["image"].shape[-1]
model = sflax.UNet(dconf["depth"], channels, dconf["num_filters"])

"""
Run training loop.
"""
workdir = "./temp/"
print(f"{'JAX process: '}{jax.process_index()}{' / '}{jax.process_count()}")
print(f"{'JAX local devices: '}{jax.local_devices()}")

start_time = time()
modvar = sflax.train_and_evaluate(
    dconf, workdir, model, train_ds, test_ds, checkpointing=True, log=True
)
time_train = time() - start_time

"""
Evaluate on testing data.
"""
start_time = time()
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"])
time_eval = time() - start_time

"""
Compare trained model in terms of reconstruction time and data fidelity.
"""
snr_eval = snr(test_ds["label"], output)
print(
    f"{'UNet':8s}{'epochs:':2s}{epochs:>5d}{'':3s}{'time[s]:':2s}{time_train:>5.2f}{'':3s}{'SNR:':4s}{snr_eval:>5.2f}{' dB'}"
)


# Plot comparison
fig, axes = plot.subplots(nrows=1, ncols=3, figsize=(12, 4.5))
plot.imview(test_ds["label"][0, ..., 0], title="Ground truth", fig=fig, ax=axes[0])
plot.imview(test_ds["image"][0, ..., 0], title=r"FBP", fig=fig, ax=axes[1])
plot.imview(
    output[0, ..., 0],
    title=r"UNet Prediction",
    fig=fig,
    ax=axes[2],
)
fig.suptitle(r"Compare FBP-based Prediction")
fig.tight_layout()
fig.colorbar(
    axes[2].get_images()[0],
    ax=axes,
    location="right",
    shrink=1.0,
    pad=0.05,
    label="Arbitrary Units",
)
fig.show()

input("\nWaiting for input to close figures and exit")
