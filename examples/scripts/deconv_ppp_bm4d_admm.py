#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
PPP (with BM4D) Volume Deconvolution
====================================

This example demonstrates the solution of a 3D image deconvolution problem
(involving recovering a 3D volume that has been convolved with a 3D kernel
and corrupted by noise) using the ADMM Plug-and-Play Priors (PPP)
algorithm :cite:`venkatakrishnan-2013-plugandplay2`, with the BM4D
:cite:`maggioni-2012-nonlocal` denoiser.
"""

import numpy as np

import jax

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot, random
from scico.examples import create_3D_foam_phantom, downsample_volume, tile_volume_slices
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Create a ground truth image.
"""
np.random.seed(1234)
N = 128  # phantom size
Nx, Ny, Nz = N, N, N // 4
upsamp = 2
x_gt_hires = create_3D_foam_phantom((upsamp * Nz, upsamp * Ny, upsamp * Nx), N_sphere=100)
x_gt = downsample_volume(x_gt_hires, upsamp)
x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU


"""
Set up forward operator and test signal consisting of blurred signal with
additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n, n)) / (n**3)
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = random.randn(Ax.shape)
y = Ax + σ * noise


"""
Set up ADMM solver.
"""
f = loss.SquaredL2Loss(y=y, A=A)
C = linop.Identity(x_gt.shape)

λ = 40.0 / 255  # BM4D regularization strength
g = λ * functional.BM4D()

ρ = 1.0  # ADMM penalty parameter
maxiter = 10  # number of ADMM iterations

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=A.T @ y,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 100}),
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
Show slices of the recovered 3D volume.
"""
show_id = Nz // 2
fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(tile_volume_slices(x_gt), title="Ground truth", fig=fig, ax=ax[0])
nc = n // 2
yc = y[nc:-nc, nc:-nc, nc:-nc]
yc = snp.clip(yc, 0, 1)
plot.imview(
    tile_volume_slices(yc),
    title="Slices of blurred, noisy volume: %.2f (dB)" % metric.psnr(x_gt, yc),
    fig=fig,
    ax=ax[1],
)
plot.imview(
    tile_volume_slices(x),
    title="Slices of deconvolved volume: %.2f (dB)" % metric.psnr(x_gt, x),
    fig=fig,
    ax=ax[2],
)
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
