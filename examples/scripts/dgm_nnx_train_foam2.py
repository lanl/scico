#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Diffusion Generative Model
==========================

This example demonstrates the training of a diffusion generative model.

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

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = get_backend().platform
print("Platform: ", platform)

from flax import nnx

from scico import plot
from scico import flax as sflax
from scico.flax.train.losses import huber_loss
from scico.flax_nnx.diffusion.models import ConditionalUNet
from scico.flax_nnx.diffusion.sampling import Euler_Maruyama_sampler as sampler
from scico.flax_nnx.diffusion.state import create_train_state
from scico.flax_nnx.diffusion.steps import eval_step_diffusion, train_step_diffusion
from scico.flax.diffusion.diagnostics import stats_obj
from scico.flax_nnx.utils import save_model

"""
Prepare parallel processing. Set an arbitrary processor count (only
applies if GPU is not available).
"""
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# platform = get_backend().platform
# print("Platform: ", platform)


"""
Read data from specified path.
"""
npy_train_file = os.path.expanduser("~/.cache/scico/examples/data/foam2_2400x64x64.npy")
dt_in = np.load(npy_train_file)
print("Read data shape: ", dt_in.shape)

"""
Augment given data by reflection transformations.
"""
data_all = np.vstack([dt_in, dt_in[:, ::-1, ...], dt_in[..., ::-1, :]])
print("Augmented data shape: ", data_all.shape)
init_sample = np.array(data_all[:80])
# Shift range to [-1, 1]
init_sample = init_sample * 2.0 - 1.0


"""
Build training structure. Inputs are generated foams.
"""
train_ds = {"image": init_sample[:-16]}
test_ds = {"image": init_sample[-16:]}


"""
Define configuration dictionary for model and training loop.

Parameters have been selected for demonstration purposes.
"""
# training configuration
train_conf: sflax.ConfigDict = {
    "seed": 12345,
    "opt_type": "ADAM",
    "batch_size": 16,
    "num_epochs": 100,
    "base_learning_rate": 1e-4,
    "warmup_epochs": 0,
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
Construct diffusion model.
"""
shape = train_ds["image"].shape[1:-1]
channels = train_ds["image"].shape[-1]

model = ConditionalUNet(
    shape=shape,
    channels=channels,
    init_channels=train_ds["image"].shape[1],
    dim_mults=(
        1.5,
        2,
        2.5,
    ),
    kernel_size=(5, 5),
    rngs=nnx.Rngs(train_conf["seed"]),
)

# print("Model defined")
# print(model)
# nnx.display(model)

stddev_prior = 6.9

workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "diff_nnx_out")
train_conf["workdir"] = workdir
train_conf["create_train_state"] = create_train_state
train_conf["train_step_fn"] = partial(train_step_diffusion, stddev_prior=stddev_prior)
train_conf["criterion"] = huber_loss
train_conf["eval_step_fn"] = partial(eval_step_diffusion, stddev_prior=stddev_prior)
train_conf["stats_obj"] = stats_obj()

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
xshape = train_ds["image"].shape[1:]
num_samples = 16
h = 4
w = 4
num_steps = 300
sample, sample_path = sampler(key, model, stddev_prior, xshape, num_steps, num_samples)
print("sample shape: ", sample.shape)

"""
Save model.
"""
save_model(model, workdir)
# to load:
# model = load_model(workdir, model)

"""
Evaluate trained model in terms of reconstruction time
and data fidelity.
"""
total_epochs = train_conf["num_epochs"]
total_time_train = time_train
print(
    f"{'Diffusion training':20s}{'epochs:':2s}{total_epochs:>5d}{'':21s}"
    f"{'time[s]:':10s}{total_time_train:>7.2f}"
)

"""
Plot samples.
"""
from numpy import einsum
import numpy as np

sample_ = einsum("ikjl", np.asarray(sample).reshape(h, w, shape[0], shape[1])).reshape(
    shape[0] * h, shape[1] * w
)
fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(7, 7))
plot.imview(sample_, title="Samples", cbar=None, fig=fig, ax=ax)
fig.show()

"""
Plot convergence statistics. Statistics are generated only if a training
cycle was done (i.e. if not reading final epoch results from checkpoint).
"""
if stats_object is not None and len(stats_object.iterations) > 0:
    hist = stats_object.history(transpose=True)
    fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(9, 5))
    plot.plot(
        np.vstack((hist.Train_Loss, hist.Eval_Loss)).T,
        x=hist.Epoch,
        ptyp="semilogy",
        title="Loss function",
        xlbl="Epoch",
        ylbl="Loss value",
        lgnd=("Train", "Test"),
        fig=fig,
        ax=ax,
    )

input("\nWaiting for input to close figures and exit")
