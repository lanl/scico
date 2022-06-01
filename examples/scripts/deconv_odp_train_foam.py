#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Training of ODP for Deconvolution
=================================

This example demonstrates the training and application of the unrolled optimization with deep priors
(ODP) with proximal map architecture described in :cite:`diamond-2018-odp` for
 a deconvolution (deblurring) problem.

The source images are foam phantoms generated with xdesign.

A class [flax.ODPNet](../_autosummary/scico.learning.rst#scico.learning.ODP)
implements the ODP architecture, which solves the optimization problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \| A \mathbf{x} - \mathbf{y} \|_2^2 + r(\mathbf{x}) \;,$$

where $A$ is a circular convolution, $\mathbf{y}$ is a set of blurred images, $r$ is a regularizer
and $\mathbf{x}$ is the set of deblurred images. The ODP, proximal map architecture,
abstracts the iterative solution by an unrolled network where each iteration corresponds
to a different stage in the ODP network and updates the prediction by solving

  $$\mathbf{x}^{k+1} = \mathrm{argmin}_{\mathbf{x}} \; \alpha_k \| A \mathbf{x} - \mathbf{y} \|_2^2 + \frac{1}{2} \| \mathbf{x} - \mathbf{x}^k - \mathbf{x}^{k+1/2} \|_2^2 \;,$$

which for the deconvolution problem corresponds to

  $$\mathbf{x}^{k+1} = \mathcal{F}^{-1} \mathrm{diag} (\alpha_k | \mathcal{K}|^2 + 1 )^{-1} \mathcal{F} \, (\alpha_k K^T * \mathbf{y} + \mathbf{x}^k + \mathbf{x}^{k+1/2}) \;,$$

where $k$ is the index of the stage (iteration),
$\mathbf{x}^k + \mathbf{x}^{k+1/2} = \mathrm{ResNet}(\mathbf{x}^{k})$
is the regularization (implemented as a residual convolutional neural network),
 $\mathbf{x}^k$ is the output of the previous stage,
 $\alpha_k > 0$ is a learned stage-wise parameter weighting the contribution of the fidelity term,
 $\mathcal{F}$ is the DFT, $K$ is the blur kernel, and $\mathcal{K}$ is the DFT of $K$.
 The output of the final stage is the set of deblurred images.
"""

import os
from functools import partial
from time import time

import jax
import jax.numpy as jnp

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scico import flax as sflax
from scico import metric, plot
from scico.flax.examples import load_foam_blur_data
from scico.flax.train.train import clip_positive, construct_traversal, train_step_post
from scico.linop import CircularConvolve

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

n = 3  # convolution kernel size
σ = 20.0 / 255  # noise level
psf = jnp.ones((n, n)) / (n * n)  # blur kernel

ishape = (output_size, output_size)
opBlur = CircularConvolve(h=psf, input_shape=ishape)

opBlur_vmap = jax.vmap(opBlur)  # for batch processing in data generation

"""
Read data from cache or generate if not available.
"""
train_nimg = 416  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg

train_ds, test_ds = load_foam_blur_data(
    train_nimg,
    test_nimg,
    output_size,
    psf,
    σ,
    verbose=True,
)

"""
Define configuration dictionary for model and training loop.
"""
batch_size = 16
epochs = 50
dconf: sflax.ConfigDict = {
    "seed": 0,
    "depth": 2,
    "num_filters": 64,
    "block_depth": 3,
    "opt_type": "SGD",
    "momentum": 0.9,
    "batch_size": batch_size,
    "num_epochs": epochs,
    "base_learning_rate": 1e-2,
    "warmup_epochs": 0,
    "num_train_steps": -1,
    "steps_per_eval": -1,
    "log_every_steps": 500,
}

"""
Construct ODPNet model.
"""
channels = train_ds["image"].shape[-1]
model = sflax.ODPNet(
    operator=opBlur,
    depth=dconf["depth"],
    channels=channels,
    num_filters=dconf["num_filters"],
    block_depth=dconf["block_depth"],
    odp_block=sflax.ODPProxDcnvBlock,
    # alpha_ini=10,
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
workdir = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples", "odp_dcnv_out")
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
time_train = time() - start_time

"""
Evaluate on testing data.
"""
del train_ds["image"]
del train_ds["label"]
start_time = time()
fmap = sflax.FlaxMap(model, modvar)
output = fmap(test_ds["image"])
time_eval = time() - start_time
output = jnp.clip(output, a_min=0, a_max=1.0)

"""
Compare trained model in terms of reconstruction time
and data fidelity.
"""
snr_eval = metric.snr(test_ds["label"], output)
psnr_eval = metric.psnr(test_ds["label"], output)
print(
    f"{'ODPNet training':18s}{'epochs:':2s}{epochs:>5d}{'':21s}{'time[s]:':10s}{time_train:>5.2f}{'':3s}"
)
print(
    f"{'ODPNet testing':18s}{'SNR:':5s}{snr_eval:>5.2f}{' dB'}{'':3s}{'PSNR:':6s}{psnr_eval:>5.2f}{' dB'}{'':3s}{'time[s]:':10s}{time_eval:>5.2f}"
)

# Plot comparison
key = jax.random.PRNGKey(54321)
indx = jax.random.randint(key, (1,), 0, test_nimg)[0]

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(test_ds["label"][indx, ..., 0], title="Ground truth", cbar=None, fig=fig, ax=ax[0])
plot.imview(
    test_ds["image"][indx, ..., 0],
    title="Blurred: \nSNR: %.2f (dB), MAE: %.3f"
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
    title="ODPNet Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
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