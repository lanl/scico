#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Comparison of DnCNN Variants for Image Denoising
================================================

This example demonstrates the solution of an image denoising problem
using DnCNN :cite:`zhang-2017-dncnn` networks trained for different noise
levels, as well as custom variants with fewer network layers, and  with a
noise level input.

The networks trained for specific noise levels are labeled 6L, 6M, 6H,
17L, 17M, and 17H, where {6, 17} denote the number of layers, and {L, M,
H} represent noise standard deviation of the training images (0.06, 0.10,
and 0.20 respectively). The networks with a noise standard deviation
input are labeled 6N and 17N, where {6, 17} again denote the number of
layers.
"""

import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.random
from scico import metric, plot
from scico.denoiser import DnCNN

"""
Create a ground truth image.
"""
np.random.seed(1234)
N = 512  # image size
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU


"""
Test different DnCNN variants on images with different noise levels.
"""
print("  σ   | variant | noisy image PSNR (dB)   | denoised image PSNR (dB)")
for σ in [0.06, 0.10, 0.20]:
    print("------+---------+-------------------------+-------------------------")
    for variant in ["17L", "17M", "17H", "17N", "6L", "6M", "6H", "6N"]:

        # Instantiate a DnCNN.
        denoiser = DnCNN(variant=variant)

        # Generate a noisy image.
        noise, key = scico.random.randn(x_gt.shape, seed=0)
        y = x_gt + σ * noise

        if variant in ["6N", "17N"]:
            x_hat = denoiser(y, sigma=σ)
        else:
            x_hat = denoiser(y)

        x_hat = np.clip(x_hat, a_min=0, a_max=1.0)

        if variant[0] == "6":
            variant += " "  # add spaces to maintain alignment

        print(
            " %.2f | %s     |          %.2f          |          %.2f          "
            % (σ, variant, metric.psnr(x_gt, y), metric.psnr(x_gt, x_hat))
        )


"""
Show reference and denoised images for σ=0.2 and variant=6N.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(x_gt, title="Reference", fig=fig, ax=ax[0])
plot.imview(y, title="Noisy image: %.2f (dB)" % metric.psnr(x_gt, y), fig=fig, ax=ax[1])
plot.imview(x_hat, title="Denoised image: %.2f (dB)" % metric.psnr(x_gt, x_hat), fig=fig, ax=ax[2])
fig.show()


input("\nWaiting for input to close figures and exit")
