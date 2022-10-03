#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Variants of DnCNNs for Image Denoising
====================================

This example demonstrates the solution of an image denoising problem
using different DnCNN variants.
:cite:`zhang-2017-dncnn` DnCNN,
:cite:`zhang-2021-plug` training denoiser with noise level as input.

Overview of different DnCNNs used in this script:
[6L, 6M, 6H, 17L, 17M, 17H] non-blind DnCNNs, where [6, 17] denote the
number of layers, and [L, M, H] represent noise level of the training
samples.

[6V, 17V] DnCNNs with addtional noise levels as inputs, where [6, 17]
denote the number of layers
"""

import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import metric, plot
from scico._flax import DnCNNNet, load_weights
from scico.data import _flax_data_path


def denoise_(x_gt, σ, variant):
    """Generate noisy image and denoise it via a DnCNN

    Args:
        x_gt: Reference image.
        σ: Noise level used for degrading the reference image.
        variant: Identify the DnCNN model to be used

    Returns:
        Denoised ouput, generated noisy image
    """

    non_blind_denoiser = ["6L", "6M", "6H", "17L", "17M", "17H"]
    blind_denoiser_with_noise_level_input = ["6V", "17V"]

    if variant not in non_blind_denoiser + blind_denoiser_with_noise_level_input:
        raise ValueError(f"Invalid value of parameter variant: {variant}")
    if variant[0] == "6":
        nlayer = 6
    else:
        nlayer = 17

    """
    Generate an noisy image
    """
    noise, key = scico.random.randn(x_gt.shape, seed=0)
    y = x_gt + σ * noise

    """
    Instanize a DnCNN and load the weights defined by variant
    """
    y_input = (
        snp.stack([y, snp.ones_like(y) * σ], -1)
        if variant in blind_denoiser_with_noise_level_input
        else snp.expand_dims(y, -1)
    )

    channels = 2 if variant in blind_denoiser_with_noise_level_input else 1

    model = DnCNNNet(depth=nlayer, channels=channels, num_filters=64, dtype=np.float32)
    variables = load_weights(_flax_data_path("dncnn%s.npz" % variant))

    x_hat = model.apply(variables, y_input, train=False, mutable=False)

    x_hat = np.clip(x_hat, a_min=0, a_max=1.0)

    x_hat = x_hat[..., 0]

    return x_hat, y


"""
Create a ground truth image.
"""
np.random.seed(1234)
N = 512  # image size
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU

"""
Test different DnCNN on images with different noise levels
"""
print("  σ   | variant | noisy image PSNR (dB)   | denoised image PSNR (dB)")
for σ in [0.06, 0.1, 0.2]:
    for variant in ["6L", "6M", "6H", "6V", "17L", "17M", "17H", "17V"]:
        x_hat, y = denoise_(x_gt, σ, variant)

        if variant[0] == "6":
            variant += " "

        print(
            " %.2f | %s     |          %.2f          |          %.2f          "
            % (σ, variant, metric.psnr(x_gt, y), metric.psnr(x_gt, x_hat))
        )

    print("")

"""
Show reference and denoised images.
"""
x_hat, y = denoise_(x_gt, 0.1, "6V")

fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(x_gt, title="Reference", fig=fig, ax=ax[0])
plot.imview(y, title="Noisy image: %.2f (dB)" % metric.psnr(x_gt, y), fig=fig, ax=ax[1])
plot.imview(x_hat, title="Denoised image: %.2f (dB)" % metric.psnr(x_gt, x_hat), fig=fig, ax=ax[2])
fig.show()

input("\nWaiting for input to close figures and exit")
