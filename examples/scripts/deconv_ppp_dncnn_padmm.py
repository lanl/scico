#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
PPP (with BM3D) Image Deconvolution (PADMM Solver)
==================================================

This example demonstrates the solution of an image deconvolution problem
using the ADMM Plug-and-Play Priors (PPP) algorithm
:cite:`venkatakrishnan-2013-plugandplay2`, with a non-blind variant of
the DnCNN :cite:`zhang-2017-dncnn` denoiser.
"""

import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot, random
from scico.optimize import ProximalADMM
from scico.util import device_info

"""
Create a ground truth image.
"""
np.random.seed(1234)
N = 512  # image size
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU


"""
Set up forward operator and test signal consisting of blurred signal with
additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / (n * n)
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = random.randn(Ax.shape)
y = Ax + σ * noise


"""
Set up PADMM solver.
"""
f = functional.ZeroFunctional()

g0 = loss.SquaredL2Loss(y=y)
λ = 15.0 / 255  # DnCNN denoiser sigma
g1 = λ * functional.DnCNN(variant="6N")
g = functional.SeparableFunctional((g0, g1))

D = linop.VerticalStack((A, linop.Identity(input_shape=A.input_shape)))


ρ = 0.4  # ADMM penalty parameter
maxiter = 20  # number of PADMM iterations

mu, nu = ProximalADMM.estimate_parameters(D)
solver = ProximalADMM(
    f=f,
    g=g,
    A=D,
    B=None,
    rho=ρ,
    mu=mu,
    nu=nu,
    x0=A.T @ y,
    maxiter=maxiter,
    itstat_options={"display": True},
)


"""
Run the solver.
"""
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
yc = snp.clip(y[nc:-nc, nc:-nc], 0, 1)
plot.imview(y, title="Blurred, noisy image: %.2f (dB)" % metric.psnr(x_gt, yc), fig=fig, ax=ax[1])
plot.imview(x, title="Deconvolved image: %.2f (dB)" % metric.psnr(x_gt, x), fig=fig, ax=ax[2])
fig.show()


"""
Plot convergence statistics.
"""
plot.plot(
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
)


input("\nWaiting for input to close figures and exit")
