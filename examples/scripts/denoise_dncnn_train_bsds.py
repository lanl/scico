#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Training of DnCNN for Denoising
===============================

This example demonstrates the training and application of the DnCNN model
from :cite:`zhang-2017-dncnn` to denoise images that have been corrupted
with additive Gaussian noise.
"""

# isort: off
import os
from time import time

import numpy as np

# Set an arbitrary processor count (only applies if GPU is not available).
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

try:
    from jax.extend.backend import get_backend  # introduced in jax 0.4.33
except ImportError:
    from jax.lib.xla_bridge import get_backend

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scico import flax as sflax
from scico import metric, plot
from scico.flax.examples import load_image_data


platform = get_backend().platform
print("Platform: ", platform)


"""
Read data from cache or generate if not available.
"""
size = 40  # patch size
train_nimg = 400  # number of training images
test_nimg = 16  # number of testing images
nimg = train_nimg + test_nimg
gray = True  # use gray scale images
data_mode = "dn"  # Denoising problem
noise_level = 0.1  # Standard deviation of noise
noise_range = False  # Use fixed noise level
stride = 23  # Stride to sample multiple patches from each image

train_ds, test_ds = load_image_data(
    train_nimg,
    test_nimg,
    size,
    gray,
    data_mode,
    verbose=True,
    noise_level=noise_level,
    noise_range=noise_range,
    stride=stride,
)


"""
Define configuration dictionary for model and training loop.

Parameters have been selected for demonstration purposes and relatively
short training. The depth of the model has been reduced to 6, instead of
the 17 of the original model. The suggested settings can be found in the
original paper.
"""
# model configuration
model_conf = {
    "depth": 6,
    "num_filters": 64,
}
# training configuration
train_conf: sflax.ConfigDict = {
    "seed": 0,
    "opt_type": "ADAM",
    "batch_size": 128,
    "num_epochs": 50,
    "base_learning_rate": 1e-3,
    "warmup_epochs": 0,
    "log_every_steps": 5000,
    "log": True,
    "checkpointing": True,
}


"""
Construct DnCNN model.
"""
channels = train_ds["image"].shape[-1]
model = sflax.DnCNNNet(
    depth=model_conf["depth"],
    channels=channels,
    num_filters=model_conf["num_filters"],
)


"""
Run training loop.
"""
workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "dncnn_out")
train_conf["workdir"] = workdir
print(f"\nJAX local devices: {jax.local_devices()}\n")

trainer = sflax.BasicFlaxTrainer(
    train_conf,
    model,
    train_ds,
    test_ds,
)
modvar, stats_object = trainer.train()


"""
Evaluate on testing data.
"""
test_patches = 720
start_time = time()
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"][:test_patches])
time_eval = time() - start_time
output = np.clip(output, a_min=0, a_max=1.0)


"""
Evaluate trained model in terms of reconstruction time and data fidelity.
"""
snr_eval = metric.snr(test_ds["label"][:test_patches], output)
psnr_eval = metric.psnr(test_ds["label"][:test_patches], output)
print(
    f"{'DnCNNNet training':18s}{'epochs:':2s}{train_conf['num_epochs']:>5d}"
    f"{'':21s}{'time[s]:':10s}{trainer.train_time:>7.2f}"
)
print(
    f"{'DnCNNNet testing':18s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}"
    f"{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}{'':3s}{'time[s]:':10s}{time_eval:>7.2f}"
)


"""
Plot comparison. Note that plots may display unidentifiable image
fragments due to the small patch size.
"""
np.random.seed(123)
indx = np.random.randint(0, high=test_patches)

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    test_ds["image"][indx, ..., 0],
    title="Noisy: \nSNR: %.2f (dB), PSNR: %.2f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], test_ds["image"][indx, ..., 0]),
        metric.psnr(test_ds["label"][indx, ..., 0], test_ds["image"][indx, ..., 0]),
    ),
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    output[indx, ..., 0],
    title="DnCNNNet Reconstruction\nSNR: %.2f (dB), PSNR: %.2f"
    % (
        metric.snr(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
        metric.psnr(test_ds["label"][indx, ..., 0], output[indx, ..., 0]),
    ),
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()


"""
Plot convergence statistics. Statistics are generated only if a training
cycle was done (i.e. if not reading final epoch results from checkpoint).
"""
if stats_object is not None and len(stats_object.iterations) > 0:
    hist = stats_object.history(transpose=True)
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plot.plot(
        np.vstack((hist.Train_Loss, hist.Eval_Loss)).T,
        x=hist.Epoch,
        ptyp="semilogy",
        title="Loss function",
        xlbl="Epoch",
        ylbl="Loss value",
        lgnd=("Train", "Test"),
        fig=fig,
        ax=ax[0],
    )
    plot.plot(
        np.vstack((hist.Train_SNR, hist.Eval_SNR)).T,
        x=hist.Epoch,
        title="Metric",
        xlbl="Epoch",
        ylbl="SNR (dB)",
        lgnd=("Train", "Test"),
        fig=fig,
        ax=ax[1],
    )
    fig.show()


input("\nWaiting for input to close figures and exit")
