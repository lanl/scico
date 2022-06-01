#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
CT with UNet for Denoising of FBP
=================================

This example demonstrates the training and application of UNet
to denoise previously filtered back projections (FBP) for CT
reconstruction inspired by :cite:`jin-2017-unet`.
"""

import os
from time import time

import jax

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scico import flax as sflax
from scico import metric, plot
from scico.flax.examples import load_ct_data

"""
Prepare parallel processing. Set an arbitrary processor
count (only applies if GPU is not available).
"""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = jax.lib.xla_bridge.get_backend().platform
print("Platform: ", platform)


"""
Read data from cache or generate if not available.
"""
N = 256  # phantom size
train_nimg = 536  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg
n_projection = 45  # CT views

trdt, ttdt = load_ct_data(train_nimg, test_nimg, N, n_projection, verbose=True)

"""
Build training and testing structures. Inputs are the filter
back-projected sinograms and outpus are the original generated foams.
Keep training and testing partitions.
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
    "block_depth": 2,
    "opt_type": "SGD",
    "momentum": 0.9,
    "batch_size": batch_size,
    "num_epochs": epochs,
    "base_learning_rate": 1e-2,
    "warmup_epochs": 0,
    "num_train_steps": -1,
    "steps_per_eval": -1,
    "log_every_steps": 400,
}

"""
Construct UNet model.
"""
channels = train_ds["image"].shape[-1]
model = sflax.UNet(
    depth=dconf["depth"],
    channels=channels,
    num_filters=dconf["num_filters"],
    block_depth=dconf["block_depth"],
)

"""
Run training loop.
"""
workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "unet_ct_out")
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
output = jax.numpy.clip(output, a_min=0, a_max=1.0)

"""
Compare trained model in terms of reconstruction time
and data fidelity.
"""
snr_eval = metric.snr(test_ds["label"], output)
psnr_eval = metric.psnr(test_ds["label"], output)
print(
    f"{'UNet training':15s}{'epochs:':2s}{epochs:>5d}{'':21s}{'time[s]:':10s}{time_train:>5.2f}{'':3s}"
)
print(
    f"{'UNet testing':15s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}{'':3s}{'time[s]:':10s}{time_eval:>5.2f}"
)


# Plot comparison
key = jax.random.PRNGKey(123)
indx = jax.random.randint(key, shape=(1,), minval=0, maxval=test_nimg)[0]

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    test_ds["image"][indx, ..., 0],
    title="FBP Reconstruction: \nSNR: %.2f (dB), MAE: %.3f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], test_ds["image"][indx, ..., 0]),
        metric.mae(test_ds["label"][indx, ..., 0], test_ds["image"][indx, ..., 0]),
    ),
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    output[indx, ..., 0],
    title="UNet Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
        metric.mae(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
    ),
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

input("\nWaiting for input to close figures and exit")