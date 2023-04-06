#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Blurred Data Generation (Foams) for NN Training
===============================================

This example demonstrates how to generate blurred image data for
training neural network models for deconvolution (deblurring). Foam
phantoms from xdesign are used to generate the clean images.
"""
import numpy as np

from scico import plot
from scico.flax.examples import load_foam1_blur_data

"""
Read data from cache or generate if not available.
"""
n = 3  # convolution kernel size
σ = 20.0 / 255  # noise level
psf = np.ones((n, n)) / (n * n)  # kernel

train_nimg = 416  # number of training images
test_nimg = 64  # number of testing images
nimg = train_nimg + test_nimg
output_size = 256  # image size

train_ds, test_ds = load_foam1_blur_data(
    train_nimg,
    test_nimg,
    output_size,
    psf,
    σ,
    verbose=True,
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
    title="Blurred Image - Training Sample",
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
    test_ds["image"][indx_te, ..., 0],
    title="Blurred Image - Testing Sample",
    fig=fig,
    ax=axes[1, 1],
)
fig.suptitle(r"Training and Testing samples")
fig.tight_layout()
fig.colorbar(
    axes[0, 1].get_images()[0],
    ax=axes,
    shrink=0.5,
    pad=0.05,
    label="Arbitrary Units",
)
fig.show()


input("\nWaiting for input to close figures and exit")
