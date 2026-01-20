#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Polar Total Variation Denoising (PDHG)
======================================

This example compares denoising via standard isotropic total
variation (TV) regularization :cite:`rudin-1992-nonlinear`
:cite:`goldstein-2009-split` and a variant based on local polar
coordinates, as described in :cite:`hossein-2024-total`. It solves the
denoising problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x}
  \|_2^2 + \lambda R(\mathbf{x}) \;,$$

where $R$ is either the isotropic or polar TV regularizer, via the
primal–dual hybrid gradient (PDHG) algorithm.
"""

from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, plot
from scico.optimize import PDHG
from scico.util import device_info

"""
Create a ground truth image.
"""
N = 256  # image size
phantom = SiemensStar(16)
x_gt = snp.pad(discrete_phantom(phantom, N - 16), 8)
x_gt = x_gt / x_gt.max()


"""
Add noise to create a noisy test image.
"""
σ = 0.75  # noise standard deviation
noise, key = scico.random.randn(x_gt.shape, seed=0)
y = x_gt + σ * noise


"""
Denoise with standard isotropic total variation.
"""
λ_std = 0.8e0
f = loss.SquaredL2Loss(y=y)
g_std = λ_std * functional.L21Norm()

# The append=0 option makes the results of horizontal and vertical finite
# differences the same shape, which is required for the L21Norm.
C = linop.FiniteDifference(input_shape=x_gt.shape, append=0)
tau, sigma = PDHG.estimate_parameters(C, ratio=20.0)
solver = PDHG(
    f=f,
    g=g_std,
    C=C,
    tau=tau,
    sigma=sigma,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
print(f"Solving on {device_info()}\n")
solver.solve()
hist_std = solver.itstat_object.history(transpose=True)
x_std = solver.x
print()


"""
Denoise with polar total variation for comparison.
"""
# Tune the weight to give the same data fidelty as the isotropic case.
λ_plr = 1.2e0
g_plr = λ_plr * functional.L1Norm()

G = linop.PolarGradient(input_shape=x_gt.shape)
D = linop.Diagonal(snp.array([0.3, 1.0]).reshape((2, 1, 1)), input_shape=G.shape[0])
C = D @ G

tau, sigma = PDHG.estimate_parameters(C, ratio=20.0)
solver = PDHG(
    f=f,
    g=g_plr,
    C=C,
    tau=tau,
    sigma=sigma,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
solver.solve()
hist_plr = solver.itstat_object.history(transpose=True)
x_plr = solver.x
print()


"""
Compute and print the data fidelity.
"""
for x, name in zip((x_std, x_plr), ("Isotropic", "Polar")):
    df = f(x)
    print(f"Data fidelity for {(name + ' TV'):12}: {df:.2e}   SNR: {metric.snr(x_gt, x):5.2f} dB")


"""
Plot results.
"""
plt_args = dict(norm=plot.matplotlib.colors.Normalize(vmin=0, vmax=1.5))
fig, ax = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11, 10))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0], **plt_args)
plot.imview(y, title="Noisy version", fig=fig, ax=ax[0, 1], **plt_args)
plot.imview(x_std, title="Isotropic TV denoising", fig=fig, ax=ax[1, 0], **plt_args)
plot.imview(x_plr, title="Polar TV denoising", fig=fig, ax=ax[1, 1], **plt_args)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="Arbitrary Units"
)
fig.suptitle("Denoising comparison")
fig.show()

# zoomed version
fig, ax = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11, 10))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0], **plt_args)
plot.imview(y, title="Noisy version", fig=fig, ax=ax[0, 1], **plt_args)
plot.imview(x_std, title="Isotropic TV denoising", fig=fig, ax=ax[1, 0], **plt_args)
plot.imview(x_plr, title="Polar TV denoising", fig=fig, ax=ax[1, 1], **plt_args)
ax[0, 0].set_xlim(N // 4, N // 4 + N // 2)
ax[0, 0].set_ylim(N // 4, N // 4 + N // 2)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="Arbitrary Units"
)
fig.suptitle("Denoising comparison (zoomed)")
fig.show()


fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(20, 5))
plot.plot(
    snp.vstack((hist_std.Objective, hist_plr.Objective)).T,
    ptyp="semilogy",
    title="Objective function",
    xlbl="Iteration",
    lgnd=("Standard", "Polar"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack((hist_std.Prml_Rsdl, hist_plr.Prml_Rsdl)).T,
    ptyp="semilogy",
    title="Primal residual",
    xlbl="Iteration",
    lgnd=("Standard", "Polar"),
    fig=fig,
    ax=ax[1],
)
plot.plot(
    snp.vstack((hist_std.Dual_Rsdl, hist_plr.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Dual residual",
    xlbl="Iteration",
    lgnd=("Standard", "Polar"),
    fig=fig,
    ax=ax[2],
)
fig.show()


input("\nWaiting for input to close figures and exit")
