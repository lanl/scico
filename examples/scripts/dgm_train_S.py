#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Diffusion Generative Model
==========================

This example demonstrates the training of a diffusion generative model.

The output of the final stage is the set of generated samples.
"""

# isort: off
import os
from functools import partial
from time import time

import numpy as np


import jax
import jax.numpy as jnp

try:
    from jax.extend.backend import get_backend  # introduced in jax 0.4.33
except ImportError:
    from jax.lib.xla_bridge import get_backend


from sklearn.datasets import make_s_curve

from scico import plot
from scico import flax as sflax
from scico.flax.train.losses import huber_loss
from scico.flax.diffusion.models import MLPScore
from scico.flax.diffusion.sampling import Euler_Maruyama_sampler as sampler
from scico.flax.diffusion.state import create_train_state
from scico.flax.diffusion.steps import eval_step_diffusion, train_step_diffusion
from scico.flax.diffusion.diagnostics import stats_obj

"""
Prepare parallel processing. Set an arbitrary processor count (only
applies if GPU is not available).
"""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = get_backend().platform
print("Platform: ", platform)


"""
Generate S curve from scikit-data function.
"""

nsamples = 5000
noise = 0.0
X, Y = make_s_curve(n_samples=nsamples, noise=noise)
init_sample = jnp.array(X)[:, [0, 2]]
scaling_factor = 7
init_sample = jnp.array((init_sample - init_sample.mean()) / init_sample.std() * scaling_factor)
print("Generated data shape: ", init_sample.shape)

"""
Build training structure. Inputs are points over the S curve.
"""
numtr = nsamples - 1000
train_ds = {"image": init_sample[:numtr]}
test_ds = {"image": init_sample[numtr:]}


"""
Define configuration dictionary for model and training loop.

Parameters have been selected for demonstration purposes.
"""
# training configuration
train_conf: sflax.ConfigDict = {
    "seed": 4444,
    "opt_type": "ADAM",
    "batch_size": 256,
    "num_epochs": 1000,
    "base_learning_rate": 1e-4,
    "warmup_epochs": 0,
    "log_every_steps": 200,
    "log": True,
    #    "checkpointing": True,
}


"""
Print configuration of distributed run.
"""
print(f"\nJAX process: {jax.process_index()}{' / '}{jax.process_count()}")
print(f"JAX local devices: {jax.local_devices()}\n")


"""
Construct score model.
"""
dim = train_ds["image"].shape[1]  # signal dimension
pos_dim = 16  # positional embedding dimension

model = MLPScore(
    in_dim=dim,
    pos_dim=pos_dim,
)

print("Model defined")
print(model)

stddev_prior = 15.6

workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "diff_S_out")
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
sample_batch_size = 1024
num_steps = 1000
sample, sample_path = sampler(
    key, model, modvar, stddev_prior, xshape, num_steps, sample_batch_size
)
print("sample shape: ", sample.shape)

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
sample_r = sample.reshape((-1, 2))
sample_path_r = sample_path.reshape((-1, num_steps, 2))
fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(7, 7))
ax.scatter(sample_r[:, 0], sample_r[:, 1])
ax.scatter(sample_path_r[:, 0, 0], sample_path_r[:, 0, 1])
ax.scatter(init_sample[:, 0], init_sample[:, 1])
fig.legend(["Generated", "Gen. Initial", "True"])
fig.show()

"""
Plot sample paths.
"""
import numpy as np

N_part = np.random.choice(sample_batch_size, size=5)
fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(7, 7))
ax.plot(sample_r[:, 0], sample_r[:, 1], "*")
ax.plot(sample_path_r[:, 0, 0], sample_path_r[:, 0, 1], "*")
for j in N_part:
    sj = sample_path_r[j, :, :]
    ax.plot(sj[:, 0], sj[:, 1], "c", linewidth=2)
    ax.plot(sj[0, 0], sj[0, 1], "rx")
    ax.plot(sj[-1, 0], sj[-1, 1], "rx")
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
    fig.show()

input("\nWaiting for input to close figures and exit")
