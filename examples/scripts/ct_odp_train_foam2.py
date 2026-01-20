#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
CT Training and Reconstruction with ODP
=======================================

This example demonstrates the training of the unrolled optimization with
deep priors (ODP) gradient descent architecture described in
:cite:`diamond-2018-odp` applied to a CT reconstruction problem.

The source images are foam phantoms generated with xdesign.

A class
[scico.flax.ODPNet](../_autosummary/scico.flax.rst#scico.flax.ODPNet)
implements the ODP architecture, which solves the optimization problem

$$\mathrm{argmin}_{\mathbf{x}} \; \| A \mathbf{x} - \mathbf{y} \|_2^2
+ r(\mathbf{x}) \;,$$

where $A$ is a tomographic projector, $\mathbf{y}$ is a set of sinograms,
$r$ is a regularizer and $\mathbf{x}$ is the set of reconstructed images.
The ODP, gradient descent architecture, abstracts the iterative solution
by an unrolled network where each iteration corresponds to a different
stage in the ODP network and updates the prediction by solving

$$\mathbf{x}^{k+1} = \mathrm{argmin}_{\mathbf{x}} \; \alpha_k \| A
\mathbf{x} - \mathbf{y} \|_2^2 + \frac{1}{2} \| \mathbf{x} -
\mathbf{x}^k - \mathbf{x}^{k+1/2} \|_2^2 \;,$$

which for the CT problem, using gradient descent, corresponds to

$$\mathbf{x}^{k+1} = \mathbf{x}^k + \mathbf{x}^{k+1/2} - \alpha_k \,
A^T \, (A \mathbf{x}^k - \mathbf{y}) \;,$$

where $k$ is the index of the stage (iteration), $\mathbf{x}^k +
\mathbf{x}^{k+1/2} = \mathrm{ResNet}(\mathbf{x}^{k})$ is the
regularization (implemented as a residual convolutional neural network),
$\mathbf{x}^k$ is the output of the previous stage and $\alpha_k > 0$ is
a learned stage-wise parameter weighting the contribution of the fidelity
term. The output of the final stage is the set of reconstructed images.
"""

# isort: off
import os
from functools import partial
from time import time

import numpy as np

import logging
import ray

ray.init(logging_level=logging.ERROR)  # need to call init before jax import: ray-project/ray#44087

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
from scico.flax.examples import load_ct_data
from scico.flax.train.traversals import clip_positive, construct_traversal
from scico.linop.xray import XRayTransform2D

platform = get_backend().platform
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
Build CT projection operator. Parameters are chosen so that the operator
is equivalent to the one used to generate the training data.
"""
angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
A = XRayTransform2D(
    input_shape=(N, N),
    angles=angles,
    det_count=int(N * 1.05 / np.sqrt(2.0)),
    dx=1.0 / np.sqrt(2),
)
A = (1.0 / N) * A  # normalize projection operator


"""
Build training and testing structures. Inputs are the sinograms and
outputs are the original generated foams. Keep training and testing
partitions.
"""
numtr = 320
numtt = 32
train_ds = {"image": trdt["sino"][:numtr], "label": trdt["img"][:numtr]}
test_ds = {"image": ttdt["sino"][:numtt], "label": ttdt["img"][:numtt]}


"""
Define configuration dictionary for model and training loop.

Parameters have been selected for demonstration purposes and relatively
short training. The model depth is akin to the number of unrolled
iterations in the MoDL model. The block depth controls the number of
layers at each unrolled iteration. The number of filters is uniform
throughout the iterations. The iterations used for the conjugate gradient
(CG) solver can also be specified. Better performance may be obtained by
increasing depth, block depth, number of filters, CG iterations, or
training epochs, but may require longer training times.
"""
# model configuration
model_conf = {
    "depth": 8,
    "num_filters": 64,
    "block_depth": 6,
}
# training configuration
train_conf: sflax.ConfigDict = {
    "seed": 1234,
    "opt_type": "ADAM",
    "batch_size": 16,
    "num_epochs": 200,
    "base_learning_rate": 1e-3,
    "warmup_epochs": 0,
    "log_every_steps": 160,
    "log": True,
    "checkpointing": True,
}


"""
Construct functionality for ensuring that the learned fidelity weight
parameter is always positive.
"""
alphatrav = construct_traversal("alpha")  # select alpha parameters in model
alphapost = partial(
    clip_positive,  # apply this function
    traversal=alphatrav,  # to alpha parameters in model
    minval=1e-3,
)


"""
Print configuration of distributed run.
"""
print(f"\nJAX process: {jax.process_index()}{' / '}{jax.process_count()}")
print(f"JAX local devices: {jax.local_devices()}\n")


"""
Construct ODPNet model.
"""
channels = train_ds["image"].shape[-1]
model = sflax.ODPNet(
    operator=A,
    depth=model_conf["depth"],
    channels=channels,
    num_filters=model_conf["num_filters"],
    block_depth=model_conf["block_depth"],
    odp_block=sflax.inverse.ODPGrDescBlock,
    alpha_ini=1e-2,
)


"""
Run training loop.
"""
workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "odp_ct_out")

train_conf["workdir"] = workdir
train_conf["post_lst"] = [alphapost]
# Construct training object
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
del train_ds["image"]
del train_ds["label"]

fmap = sflax.FlaxMap(model, modvar)
del model, modvar

maxn = numtt
start_time = time()
output = fmap(test_ds["image"][:maxn])
time_eval = time() - start_time
output = np.clip(output, a_min=0, a_max=1.0)
epochs = train_conf["num_epochs"]


"""
Evaluate trained model in terms of reconstruction time and data fidelity.
"""
snr_eval = metric.snr(test_ds["label"][:maxn], output)
psnr_eval = metric.psnr(test_ds["label"][:maxn], output)
print(
    f"{'ODPNet training':18s}{'epochs:':2s}{epochs:>5d}{'':21s}"
    f"{'time[s]:':10s}{trainer.train_time:>7.2f}"
)
print(
    f"{'ODPNet testing':18s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}"
    f"{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}{'':3s}{'time[s]:':10s}{time_eval:>7.2f}"
)


"""
Plot comparison.
"""
np.random.seed(123)
indx = np.random.randint(0, high=maxn)

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    test_ds["image"][indx, ..., 0],
    title="Sinogram",
    cbar=None,
    fig=fig,
    ax=ax[1],
)
plot.imview(
    output[indx, ..., 0],
    title="ODPNet Reconstruction\nSNR: %.2f (dB), PSNR: %.2f"
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
