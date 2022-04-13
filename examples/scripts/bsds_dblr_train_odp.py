#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Training of ODP for Deblurring
==============================

This example demonstrates the training and application of the unrolled optimization with deep priors (ODP) with proximal map architecture described in :cite:`diamond-2018-odp` for a deblurring problem.

The source images are part of the [BSDS500 dataset] (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/) provided by the Berkeley Segmentation Dataset and Benchmark project.

A class [ODPNet] implements the ODP architecture,
which solves the optimization problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \| A \mathbf{x} - \mathbf{y} \|_2^2 + r(\mathbf{x}) \;,$$

where $A$ is a circular convolution, $\mathbf{y}$ is a set of blurred images, $r$ is a regularizer and $\mathbf{x}$ is the set of deblurred images. The ODP, proximal map architecture, abstracts the iterative solution by an unrolled network where each iteration corresponds to a different stage in the ODP network and updates the prediction by solving

  $$\mathbf{x}^{k+1} = \mathrm{argmin}_{\mathbf{x}} \; \alpha_k \| A \mathbf{x} - \mathbf{y} \|_2^2 + \frac{1}{2} \| \mathbf{x} - \mathbf{x}^k - \mathbf{x}^{k+1/2} \|_2^2 \;,$$

which for the deblurring problem corresponds to

  $$\mathbf{x}^{k+1} = \mathcal{F}^{-1} \mathrm{diag} (\alpha_k | \mathcal{K}|^2 + 1 )^{-1} \mathcal{F} \, (\alpha_k K^T * \mathbf{y} + \mathbf{x}^k + \mathbf{x}^{k+1/2}) \;,$$

where $k$ is the index of the stage (iteration), $\mathbf{x}^k + \mathbf{x}^{k+1/2} = \mathrm{ResNet}(\mathbf{x}^{k})$ is the regularization (implemented as a residual convolutional neural network), $\mathbf{x}^k$ is the output of the previous stage, $\alpha_k > 0$ is a learned stage-wise parameter weighting the contribution of the fidelity term, $\mathcal{F}$ is the DFT, $K$ is the blurring kernel, and $\mathcal{K}$ is the DFT of $K$. The output of the final stage is the set of deblurred images.
"""

import os
from functools import partial
from time import time

import jax
import jax.numpy as jnp

from scico import flax as sflax
from scico import plot
from scico.examples_flax import construct_blurring_operator, load_image_data
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
Define blurring operator.
"""
output_size = 256  # patch size
channels = 1  # Gray scale problem
blur_ksize = (5, 5)  # Size of blurring kernel
blur_sigma = 5  # STD of Gaussian blurring

opBlur = construct_blurring_operator(output_size, channels, blur_ksize, blur_sigma)

opBlur_vmap = jax.vmap(opBlur)  # For batch processing in data generation

"""
Read data from cache or generate if not available.
"""
train_nimg = 400  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg
gray = True  # use gray scale images
data_mode = "dblr"  # Denoising problem
noise_level = 0.01  # Standard deviation of noise
noise_range = False  # Use fixed noise level
stride = 100  # Stride to sample multiple patches from each image
augment = True  # Augment data via rotations and flips


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
epochs = 100
dconf: sflax.ConfigDict = {
    "seed": 0,
    "depth": 1,
    "num_filters": 64,
    "block_depth": 5,
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
Construct ODPNet model.
"""
model = sflax.ODPNet(
    operator=opBlur,
    depth=dconf["depth"],
    channels=channels,
    num_filters=dconf["num_filters"],
    block_depth=dconf["block_depth"],
    odp_block=sflax.ODPProxDblrBlock,
    alpha_ini=10,
)


"""
Construct functionality for making sure that
the learned fidelity weight parameter is always
positive.
"""
alphatrav = construct_traversal("alpha")
alphapos = partial(
    clip_positive,
    traversal=alphatrav,
    minval=1e-3,
)
train_step = partial(train_step_post, post_fn=alphapos)


"""
Run training loop.
"""
workdir = os.path.join(
    os.path.expanduser("~"), ".cache", "scico", "examples", "img", "odp_dblr_out"
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
    # create_lr_schedule=create_exp_lr_schedule,
    training_step_fn=train_step,
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
    f"{'ODPNet':14s}{'epochs:':2s}{epochs:>5d}{'':3s}{'time[s]:':10s}{time_train:>5.2f}{'':3s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}"
)


# Plot comparison
key = jax.random.PRNGKey(54321)
indx_te = jax.random.randint(key, (1,), 0, test_nimg)[0]
fig, axes = plot.subplots(nrows=1, ncols=3, figsize=(12, 4.5))
plot.imview(test_ds["label"][indx_te, ..., 0], title="Ground truth", fig=fig, ax=axes[0])
plot.imview(test_ds["image"][indx_te, ..., 0], title=r"Blurred", fig=fig, ax=axes[1])
plot.imview(
    output[indx_te, ..., 0],
    title=r"ODPNet Prediction",
    fig=fig,
    ax=axes[2],
)
fig.suptitle(r"Compare ODPNet Deblurring")
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