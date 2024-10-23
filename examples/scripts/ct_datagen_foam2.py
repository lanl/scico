#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
CT Data Generation for NN Training
==================================

This example demonstrates how to generate synthetic CT data for training
neural network models. If desired, a basic reconstruction can be
generated using filtered back projection (FBP).
"""

# isort: off
import os
import numpy as np

import logging
import ray

ray.init(logging_level=logging.ERROR)  # need to call init before jax import: ray-project/ray#44087

# Set an arbitrary processor count (only applies if GPU is not available).
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from scico import plot
from scico.flax.examples import load_ct_data

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
Plot randomly selected sample.
"""
indx_tr = np.random.randint(0, train_nimg)
indx_te = np.random.randint(0, test_nimg)
fig, axes = plot.subplots(nrows=2, ncols=3, figsize=(9, 9))
plot.imview(
    trdt["img"][indx_tr, ..., 0], title="Ground truth - Training Sample", fig=fig, ax=axes[0, 0]
)
plot.imview(
    trdt["sino"][indx_tr, ..., 0], title="Sinogram - Training Sample", fig=fig, ax=axes[0, 1]
)
plot.imview(
    trdt["fbp"][indx_tr, ..., 0],
    title="FBP - Training Sample",
    fig=fig,
    ax=axes[0, 2],
)
plot.imview(
    ttdt["img"][indx_te, ..., 0],
    title="Ground truth - Testing Sample",
    fig=fig,
    ax=axes[1, 0],
)
plot.imview(
    ttdt["sino"][indx_te, ..., 0], title="Sinogram - Testing Sample", fig=fig, ax=axes[1, 1]
)
plot.imview(
    ttdt["fbp"][indx_te, ..., 0],
    title="FBP - Testing Sample",
    fig=fig,
    ax=axes[1, 2],
)
fig.suptitle(r"Training and Testing samples")
fig.tight_layout()
fig.colorbar(
    axes[0, 2].get_images()[0],
    ax=axes,
    shrink=0.5,
    pad=0.05,
    label="Arbitrary Units",
)
fig.show()


input("\nWaiting for input to close figures and exit")
