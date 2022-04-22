#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Training of MoDL for Deconvolution
==================================

This example demonstrates the training and application of a model-based deep learning (MoDL) architecture described in :cite:`aggarwal-2019-modl` for a deconvolution (deblurring) problem.

The source images are part of the [BSDS500 dataset] (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/) provided by the Berkeley Segmentation Dataset and Benchmark project.

A class [MoDLNet] implements the MoDL architecture, which solves the optimization problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \| A \mathbf{x} - \mathbf{y} \|_2^2 + \lambda \, \| \mathbf{x} - \mathrm{D}_w(\mathbf{x})\|_2^2 \;,$$

where $A$ is a circular convolution, $\mathbf{y}$ is a set of blurred images, $\mathrm{D}_w$ is the regularization (a denoiser), and $\mathbf{x}$ is the set of deblurred images. The MoDL abstracts the iterative solution by an unrolled network where each iteration corresponds to a different stage in the MoDL network and updates the prediction by solving

  $$\mathbf{x}^{k+1} = (A^T A + \lambda \, \mathbf{I})^{-1} (A^T \mathbf{y} + \lambda \, \mathbf{z}^k) \;,$$

via conjugate gradient. In the expression, $k$ is the index of the stage (iteration), $\mathbf{z}^k = \mathrm{ResNet}(\mathbf{x}^{k})$ is the regularization (a denoiser implemented as a residual convolutional neural network), $\mathbf{x}^k$ is the output of the previous stage, $\lambda > 0$ is a learned regularization parameter, and $\mathbf{I}$ is the identity operator. The output of the final stage is the set of deblurred images.
"""

import os
from functools import partial
from time import time

import jax
import jax.numpy as jnp

from scico import flax as sflax
from scico import plot
from scico.flax.examples import construct_blur_operator, load_image_data
from scico.flax.train.train import clip_positive, construct_traversal, train_step_post
from scico.metric import psnr, snr

"""
Prepare parallel processing. Set an arbitrary processor
count (only applies if GPU is not available).
"""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
platform = jax.lib.xla_bridge.get_backend().platform
print("Platform: ", platform)

"""
Define blur operator.
"""
output_size = 256  # patch size
channels = 1  # gray scale problem
blur_shape = (5, 5)  # shape of blur kernel
blur_sigma = 5  # Gaussian blur kernel parameter

opBlur = construct_blur_operator(output_size, channels, blur_shape, blur_sigma)

opBlur_vmap = jax.vmap(opBlur)  # for batch processing in data generation

"""
Read data from cache or generate if not available.
"""
train_nimg = 400  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg
gray = True  # use gray scale images
data_mode = "dcnv"  # deconvolution problem
noise_level = 0.01  # standard deviation of noise
noise_range = False  # use fixed noise level
stride = 100  # stride to sample multiple patches from each image
augment = True  # augment data via rotations and flips


train_ds, test_ds = load_image_data(
    train_nimg,
    test_nimg,
    output_size,
    gray,
    data_mode,
    verbose=True,
    noise_level=noise_level,
    noise_range=noise_range,
    transf=opBlur_vmap,
    stride=stride,
    augment=augment,
)


"""
Define configuration dictionary for model and training loop.
"""
batch_size = 16
epochs = 10
dconf: sflax.ConfigDict = {
    "seed": 0,
    "depth": 1,
    "num_filters": 64,
    "block_depth": 4,
    "opt_type": "ADAM",
    "batch_size": batch_size,
    "num_epochs": epochs,
    "base_learning_rate": 1e-3,
    "warmup_epochs": 0,
    "num_train_steps": -1,
    "steps_per_eval": -1,
    "log_every_steps": 500,
}

"""
Construct MoDLNet model. Use only one iteration in the model for
faster intialization and few CG iterations.
"""
model = sflax.MoDLNet(
    operator=opBlur,
    depth=1,
    channels=channels,
    num_filters=dconf["num_filters"],
    block_depth=dconf["block_depth"],
    cg_iter=3,
)


"""
Construct functionality for making sure that
the learned regularization parameter is always
positive.
"""
lmbdatrav = construct_traversal("lmbda")
lmbdapos = partial(
    clip_positive,
    traversal=lmbdatrav,
    minval=5e-4,
)
train_step = partial(train_step_post, post_fn=lmbdapos)


"""
Run first stage (initialization) training loop.
"""
workdir = os.path.join(
    os.path.expanduser("~"), ".cache", "scico", "examples", "img", "modl_dcnv_out"
)
print(f"{'JAX process: '}{jax.process_index()}{' / '}{jax.process_count()}")
print(f"{'JAX local devices: '}{jax.local_devices()}")

start_time = time()
modvar = sflax.train_and_evaluate(
    dconf,
    workdir,
    model,
    train_ds,
    test_ds,
    training_step_fn=train_step,
    checkpointing=True,
    log=True,
)
time_init = time() - start_time

print(f"{'MoDLNet Init':8s}{'epochs:':2s}{epochs:>5d}{'':3s}{'time[s]:':2s}{time_init:>5.2f}")

"""
Run second stage (depth iterations) training loop.
"""
model = sflax.MoDLNet(
    operator=opBlur,
    depth=dconf["depth"],
    channels=channels,
    num_filters=dconf["num_filters"],
    block_depth=dconf["block_depth"],
)

dconf["num_epochs"] = 70
workdir2 = os.path.join(
    os.path.expanduser("~"), ".cache", "scico", "examples", "img", "modl_dcnv_out", "iterated"
)

start_time = time()
modvar = sflax.train_and_evaluate(
    dconf,
    workdir2,
    model,
    train_ds,
    test_ds,
    training_step_fn=train_step,
    variables0=modvar,
    checkpointing=True,
    log=True,
)
time_train = time() - start_time

"""
Evaluate on testing data.
"""
del train_ds["image"]
del train_ds["label"]
start_time = time()
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"][:test_nimg])
time_eval = time() - start_time
output = jnp.clip(output, a_min=0, a_max=1.0)

"""
Compare trained model in terms of reconstruction time
and data fidelity.
"""
snr_eval = snr(test_ds["label"][:test_nimg], output)
psnr_eval = psnr(test_ds["label"][:test_nimg], output)
print(
    f"{'MoDLNet':14s}{'epochs:':2s}{dconf['num_epochs']:>5d}{'':3s}{'time[s]:':10s}{time_train:>5.2f}{'':3s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}"
)


# Plot comparison
key = jax.random.PRNGKey(54321)
indx_te = jax.random.randint(key, (1,), 0, test_nimg)[0]
fig, axes = plot.subplots(nrows=1, ncols=3, figsize=(12, 4.5))
plot.imview(test_ds["label"][indx_te, ..., 0], title="Ground truth", fig=fig, ax=axes[0])
plot.imview(test_ds["image"][indx_te, ..., 0], title=r"Blurred", fig=fig, ax=axes[1])
plot.imview(
    output[indx_te, ..., 0],
    title=r"MoDLNet Prediction",
    fig=fig,
    ax=axes[2],
)
fig.suptitle(r"Compare MoDLNet Deconvolution")
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
