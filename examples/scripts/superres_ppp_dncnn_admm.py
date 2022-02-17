#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Image Superresolution (ADMM Plug-and-Play Priors w/ DnCNN)
==========================================================

This example demonstrates the use of the ADMM Plug and Play Priors
(PPP) algorithm :cite:`venkatakrishnan-2013-plugandplay2` for solving
a simple image superresolution problem.
"""

import jax

import scico
import scico.numpy as snp
import scico.random
from scico import denoiser, functional, linop, loss, metric, plot
from scico.data import kodim23
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.solver import cg
from scico.util import device_info

"""
Define downsampling function.
"""


def downsample_image(img, rate):
    img = snp.mean(snp.reshape(img, (-1, rate, img.shape[1], img.shape[2])), axis=1)
    img = snp.mean(snp.reshape(img, (img.shape[0], -1, rate, img.shape[2])), axis=2)
    return img


"""
Read a ground truth image.
"""
img = kodim23(asfloat=True)[160:416, 60:316]
img = jax.device_put(img)


"""
Create a test image by downsampling and adding Gaussian white noise.
"""
rate = 4  # downsampling rate
σ = 2e-2  # noise standard deviation

Afn = lambda x: downsample_image(x, rate=rate)
s = Afn(img)
input_shape = img.shape
output_shape = s.shape
noise, key = scico.random.randn(s.shape, seed=0)
sn = s + σ * noise


"""
Set up the PPP problem pseudo-functional. The DnCNN denoiser
:cite:`zhang-2017-dncnn` is used as a regularizer.
"""
A = linop.LinearOperator(input_shape=input_shape, output_shape=output_shape, eval_fn=Afn)
f = loss.SquaredL2Loss(y=sn, A=A)
C = linop.Identity(input_shape=input_shape)
g = functional.DnCNN("17M")


"""
Compute a baseline solution via denoising of the pseudo-inverse of the
forward operator. This baseline solution is also used to initialize the
PPP solver.
"""
xpinv, info = cg(A.T @ A, A.T @ sn, snp.zeros(input_shape))
dncnn = denoiser.DnCNN("17M")
xden = dncnn(xpinv)


"""
Set up an ADMM solver and solve.
"""
ρ = 3.4e-2  # ADMM penalty parameter
maxiter = 12  # number of ADMM iterations
solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=xden,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 10}),
    itstat_options={"display": True},
)

print(f"Solving on {device_info()}\n")
xppp = solver.solve()
hist = solver.itstat_object.history(transpose=True)


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


"""
Show reference and test images.
"""
fig = plot.figure(figsize=(8, 6))
ax0 = plot.plt.subplot2grid((1, rate + 1), (0, 0), colspan=rate)
plot.imview(img, title="Reference", fig=fig, ax=ax0)
ax1 = plot.plt.subplot2grid((1, rate + 1), (0, rate))
plot.imview(sn, title="Downsampled", fig=fig, ax=ax1)
fig.show()


"""
Show recovered full-resolution images.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(xpinv, title="Pseudo-inverse: %.2f (dB)" % metric.psnr(img, xpinv), fig=fig, ax=ax[0])
plot.imview(
    xden, title="Denoised pseudo-inverse: %.2f (dB)" % metric.psnr(img, xden), fig=fig, ax=ax[1]
)
plot.imview(xppp, title="PPP solution: %.2f (dB)" % metric.psnr(img, xppp), fig=fig, ax=ax[2])
fig.show()


input("\nWaiting for input to close figures and exit")
