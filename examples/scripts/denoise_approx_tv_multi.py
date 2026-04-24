#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Denoising with Approximate Total Variation Proximal Operator
============================================================

This example demonstrates use of approximations to the proximal
operators of isotropic :cite:`kamilov-2016-minimizing` and anisotropic
:cite:`kamilov-2016-parallel` total variation norms for solving
denoising problems using proximal algorithms.
"""

import matplotlib
from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, plot
from scico.optimize import AcceleratedPGM
from scico.optimize.admm import ADMM, LinearSubproblemSolver
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
σ = 0.5  # noise standard deviation
noise, key = scico.random.randn(x_gt.shape, seed=0)
y = x_gt + σ * noise


"""
Denoise with isotropic total variation, solved via ADMM.
"""
λ_iso = 1.0e0
f = loss.SquaredL2Loss(y=y)
g_iso = λ_iso * functional.L21Norm()
C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)

solver = ADMM(
    f=f,
    g_list=[g_iso],
    C_list=[C],
    rho_list=[1e1],
    x0=y,
    maxiter=200,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
    itstat_options={"display": True, "period": 25},
)
print(f"Solving on {device_info()}\n")
x_iso = solver.solve()
print()


"""
Denoise with anisotropic total variation, solved via ADMM.
"""
# Tune the weight to give the same data fidelity as the isotropic case.
λ_aniso = 8.68e-1
g_aniso = λ_aniso * functional.L1Norm()

solver = ADMM(
    f=f,
    g_list=[g_aniso],
    C_list=[C],
    rho_list=[1e1],
    x0=y,
    maxiter=200,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 25}),
    itstat_options={"display": True, "period": 25},
)
x_aniso = solver.solve()
print()


"""
Denoise with isotropic total variation, solved using an approximation of
the TV norm proximal operator.
"""
h = λ_iso * functional.IsotropicTVNorm(circular=True, input_shape=y.shape)
solver = AcceleratedPGM(
    f=f, g=h, L0=1e3, x0=y, maxiter=500, itstat_options={"display": True, "period": 50}
)
x_iso_aprx = solver.solve()
print()


"""
Denoise with anisotropic total variation, solved using an approximation
of the TV norm proximal operator.
"""
h = λ_aniso * functional.AnisotropicTVNorm(circular=True, input_shape=y.shape)
solver = AcceleratedPGM(
    f=f, g=h, L0=1e3, x0=y, maxiter=500, itstat_options={"display": True, "period": 50}
)
x_aniso_aprx = solver.solve()
print()


"""
Compute and print the data fidelity.
"""
for x, name in zip(
    (x_iso, x_aniso, x_iso_aprx, x_aniso_aprx),
    ("Isotropic", "Anisotropic", "Approx. Isotropic", "Approx. Anisotropic"),
):
    df = f(x)
    print(f"Data fidelity for {name} TV: {' ' * (20 - len(name))} {df:.2e}")


"""
Plot results.
"""
matplotlib.rc("font", size=9)
plt_args = dict(norm=plot.matplotlib.colors.Normalize(vmin=0, vmax=1.5))
fig, ax = plot.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 8))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0], **plt_args)
plot.imview(
    y, title=f"Noisy version SNR: {metric.snr(x_gt, y):.2f} dB", fig=fig, ax=ax[1, 0], **plt_args
)
plot.imview(
    x_iso,
    title=f"Iso. TV denoising SNR: {metric.snr(x_gt, x_iso):.2f} dB",
    fig=fig,
    ax=ax[0, 1],
    **plt_args,
)
plot.imview(
    x_aniso,
    title=f"Aniso. TV denoising SNR: {metric.snr(x_gt, x_aniso):.2f} dB",
    fig=fig,
    ax=ax[1, 1],
    **plt_args,
)
plot.imview(
    x_iso_aprx,
    title=f"Approx. Iso. TV denoising SNR: {metric.snr(x_gt, x_iso_aprx):.2f} dB",
    fig=fig,
    ax=ax[0, 2],
    **plt_args,
)
plot.imview(
    x_aniso_aprx,
    title=f"Approx. Aniso. TV denoising SNR: {metric.snr(x_gt, x_aniso_aprx):.2f} dB",
    fig=fig,
    ax=ax[1, 2],
    **plt_args,
)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="Arbitrary Units"
)
fig.suptitle("Denoising comparison")
fig.show()


input("\nWaiting for input to close figures and exit")
