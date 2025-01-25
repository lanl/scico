#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Variational Autoencoder
=======================

This example demonstrates the training of a variational autoencoder.

The output of the final stage is the set of generated images.
"""

# isort: off
import os
from functools import partial
from time import time

import numpy as np


import jax

try:
    from jax.extend.backend import get_backend  # introduced in jax 0.4.33
except ImportError:
    from jax.lib.xla_bridge import get_backend


import flax.linen as nn

import optax

from scico import plot
from scico import flax as sflax
from scico.flax.autoencoders.varautoencoders import ConvVAE
from scico.flax.autoencoders.state import create_vae_train_state
from scico.flax.autoencoders.steps import build_sample_fn, eval_step_vae, train_step_vae
from scico.flax.autoencoders.diagnostics import stats_obj
from scico.flax.train.learning_rate import create_cosine_lr_schedule

"""
Prepare parallel processing. Set an arbitrary processor count (only
applies if GPU is not available).
"""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = get_backend().platform
print("Platform: ", platform)


"""
Read data from specified path.
"""
npy_train_file = "~/.cache/scico/examples/data/foam2_2400x64x64.npy"
dt_in = np.load(npy_train_file)
print("Read data shape: ", dt_in.shape)

"""
Augment given data by reflection transformations.
"""
data_all = np.vstack([dt_in, dt_in[:, ::-1, ...], dt_in[..., ::-1, :]])
print("Augmented data shape: ", data_all.shape)

"""
Build training structure. Inputs are generated foams.
"""
numtr = 100
train_ds = {"image": data_all[:numtr]}
test_ds = {"image": data_all[-16:]}


"""
Define configuration dictionary for model and training loop.

Parameters have been selected for demonstration purposes.
"""
# training configuration
train_conf: sflax.ConfigDict = {
    "seed": 12345,
    "opt_type": "SGD",
    "momentum": 0.9,
    "batch_size": 16,
    "num_epochs": 500,
    # "base_learning_rate": 1e-4, # DenseVAE
    "base_learning_rate": 5e-5,  # ConvVAE
    "warmup_epochs": 100,
    "log_every_steps": 40,
    "log": True,
    #    "checkpointing": True,
}


"""
Print configuration of distributed run.
"""
print(f"\nJAX process: {jax.process_index()}{' / '}{jax.process_count()}")
print(f"JAX local devices: {jax.local_devices()}\n")


"""
Construct MLP VAE model.
model, using only one iteration (depth) in model and few CG iterations
for faster intialization. Run first stage (initialization) training
loop followed by a second stage (depth iterations) training loop.
"""
size = train_ds["image"].shape[1]
channels = train_ds["image"].shape[-1]
latent_dim = 256  # 64

# model = DenseVAE(
#    out_shape = (size, size,),
#    channels = channels,
#    encoder_widths = [4096, 1024], #[64, 32, 16],
#    latent_dim = latent_dim,
#    decoder_widths = [1024, 4096], #[16, 32, 64],
#    activation_fn = nn.leaky_relu,
# )
model = ConvVAE(
    out_shape=(
        size,
        size,
    ),
    channels=channels,
    encoder_filters=[16, 4],
    latent_dim=latent_dim,
    decoder_filters=[4, 16],
    encoder_activation_fn=nn.leaky_relu,
    decoder_activation_fn=nn.leaky_relu,
)
print("Model defined")
print(model)

kl_weight = 0.5

workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "dvae_out")
train_conf["workdir"] = workdir
train_conf["create_train_state"] = create_vae_train_state
train_conf["train_step_fn"] = partial(train_step_vae, kl_weight=kl_weight)
train_conf["criterion"] = optax.l2_loss
train_conf["eval_step_fn"] = partial(eval_step_vae, kl_weight=kl_weight)
train_conf["stats_obj"] = stats_obj()
train_conf["lr_schedule"] = create_cosine_lr_schedule  # ConvVAE


# Construct training object
trainer = sflax.BasicFlaxTrainer(
    train_conf,
    model,
    train_ds,
    test_ds,
)

key = jax.random.PRNGKey(0x1234)
start_time = time()
modvar, stats_object = trainer.train(key)
time_train = time() - start_time


"""
Generate new samples.
"""
sample_fn = build_sample_fn(model, modvar)

num_samples = 16
h = 4
w = 4

key, z_key = jax.random.split(key)
z = jax.random.normal(z_key, (num_samples, latent_dim))
sample = sample_fn(z)
print("z shape: ", z.shape)
print("sample shape: ", sample.shape)


"""
Evaluate trained model in terms of reconstruction time
and data fidelity.
"""
total_epochs = train_conf["num_epochs"]
total_time_train = time_train
print(
    f"{'VAE training':18s}{'epochs:':2s}{total_epochs:>5d}{'':21s}"
    f"{'time[s]:':10s}{total_time_train:>7.2f}"
)

"""
Plot samples.
"""
from numpy import einsum
import numpy as np

sample_ = einsum("ikjl", np.asarray(sample).reshape(h, w, size, size)).reshape(size * h, size * w)
fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(7, 7))
plot.imview(sample_, title="Samples", cbar=None, fig=fig, ax=ax)
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
        np.vstack((hist.Train_KL, hist.Eval_KL)).T,
        x=hist.Epoch,
        title="Metric",
        xlbl="Epoch",
        ylbl="KL Divergence",
        lgnd=("Train", "Test"),
        fig=fig,
        ax=ax[1],
    )
    fig.show()

input("\nWaiting for input to close figures and exit")
