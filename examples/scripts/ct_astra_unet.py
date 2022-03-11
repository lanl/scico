#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
CT with UNet for Denoising of FBP
=================================

This example demonstrates the training and application of a UNet to denoise previously filtered back projections for CT reconstruction inspired by :cite:`jin-2017-unet`.
"""

from time import time

import numpy as np

import jax

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
Get and load training and testing data.
"""
# ToDo: Replace by download from repo ??
path_in = "./dtout/"
trdt = np.load(path_in + "foam2ct_train.npz")
ttdt = np.load(path_in + "foam2ct_test.npz")
N = trdt["img"].shape[1]   # image size
n_projection = trdt["sino"].shape[1]  # CT views

"""
Rebuild CT projection operator used to generate synthetic measurements.
"""
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
gt_sh = (N, N)
A = ParallelBeamProjector(gt_sh, 1, N, angles)  # Radon transform operator

"""
Build training and testing structures. Inputs are the filter back-projected sinograms and outpus are the original generated foams. Keep training and testing partitions.
"""
train_ds = {"image": trdt["fbp"], "label": trdt["img"]}
test_ds = {"image": ttdt["fbp"], "label": ttdt["img"]}

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
model = sflax.UNet(depth=dconf["depth"],
            channels=channels,
            num_filters=dconf["num_filters"],
            block_depth=dconf["block_depth"])

"""
Run training loop.
"""
workdir = "./unet_out/"
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
