#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Image Deconvolution (PGM Plug-and-Play Priors w/ BM3D)
======================================================

This example demonstrates the use of class
[pgm.AcceleratedPGM](../_autosummary/scico.optimize.rst#scico.optimize.AcceleratedPGM)
to solve an image deconvolution problem using the Plug-and-Play Priors
framework :cite:`venkatakrishnan-2013-plugandplay2`
:cite:`kamilov-2017-plugandplay`, using BM3D :cite:`dabov-2008-image`
as a denoiser.
"""

import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, plot
from scico.optimize.pgm import AcceleratedPGM
from scico.util import device_info

"""
Create a ground truth image.
"""
np.random.seed(1234)
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=512)
x_gt = jax.device_put(x_gt)  # convert to jax type, push to GPU


"""
Set up forward operator and test signal consisting of blurred signal
with additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / n ** 2
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = scico.random.randn(Ax.shape)
y = Ax + σ * noise


"""
Set up and run a PGM solver.
"""
f = loss.SquaredL2Loss(y=y, A=A)

L0 = 15  # APGM inverse step size parameter
λ = L0 * 2.0 / 255  # BM3D regularization strength
g = λ * functional.BM3D()

maxiter = 50  # number of APGM iterations

solver = AcceleratedPGM(
    f=f, g=g, L0=L0, x0=A.T @ y, maxiter=maxiter, itstat_options={"display": True, "period": 10}
)

print(f"Solving on {device_info()}\n")
x = solver.solve()
x = snp.clip(x, 0, 1)
hist = solver.itstat_object.history(transpose=True)


"""
Show the recovered image.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0])
nc = n // 2
y = snp.clip(y, 0, 1)
yc = y[nc:-nc, nc:-nc]
plot.imview(y, title="Blurred, noisy image: %.2f (dB)" % metric.psnr(x_gt, yc), fig=fig, ax=ax[1])
plot.imview(x, title="Deconvolved image: %.2f (dB)" % metric.psnr(x_gt, x), fig=fig, ax=ax[2])
fig.show()


"""
Plot convergence statistics.
"""
plot.plot(hist.Residual, ptyp="semilogy", title="PGM Residual", xlbl="Iteration", ylbl="Residual")


input("\nWaiting for input to close figures and exit")
