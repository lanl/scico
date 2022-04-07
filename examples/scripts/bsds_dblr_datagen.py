#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Generation of Blurred Data for NN Training
==========================================

This example demonstrates how to generate blurred image data for training neural network models for deblurring. The original images are part of the [BSDS500 dataset] (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/) provided by the Berkeley Segmentation Dataset and Benchmark project.
"""
import numpy as np

from jax import vmap

from scico import plot
from scico.examples_flax import construct_blurring_operator, load_image_data

"""
Define blurring operator.
"""
output_size = 256  # patch size
channels = 1  # Gray scale problem
blur_ksize = (5, 5)  # Size of blurring kernel
blur_sigma = 5  # STD of Gaussian blurring

opBlur = construct_blurring_operator(output_size, channels, blur_ksize, blur_sigma)

opBlur_vmap = vmap(opBlur)  # For batch processing

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
Plot randomly selected sample.
"""
indx_tr = np.random.randint(0, train_nimg)
indx_te = np.random.randint(0, test_nimg)
fig, axes = plot.subplots(nrows=2, ncols=2, figsize=(7, 7))
plot.imview(
    train_ds["label"][indx_tr, ..., 0],
    title="Ground truth - Training Sample",
    fig=fig,
    ax=axes[0, 0],
)
plot.imview(
    train_ds["image"][indx_tr, ..., 0],
    title="Blured Image - Training Sample",
    fig=fig,
    ax=axes[0, 1],
)
plot.imview(
    test_ds["label"][indx_te, ..., 0],
    title="Ground truth - Testing Sample",
    fig=fig,
    ax=axes[1, 0],
)
plot.imview(
    test_ds["image"][indx_te, ..., 0], title="Blured Image - Testing Sample", fig=fig, ax=axes[1, 1]
)
fig.suptitle(r"Training and Testing samples")
fig.tight_layout()
fig.colorbar(
    axes[1, 0].get_images()[0],
    ax=axes,
    location="bottom",
    shrink=0.5,
    pad=0.05,
    label="Arbitrary Units",
)
fig.show()

input("\nWaiting for input to close figures and exit")
