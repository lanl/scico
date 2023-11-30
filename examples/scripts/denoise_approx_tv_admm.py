#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Total Variation Denoising with Approximate Proximal Operator
============================================================

This example compares denoising via isotropic and anisotropic total
variation (TV) regularization :cite:`rudin-1992-nonlinear`
:cite:`goldstein-2009-split` via the optimization problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x}
  \|_2^2 + \lambda R(\mathbf{x}) \;,$$

where $R$ is either the isotropic or anisotropic TV regularizer.
Two different algorithms are used to solve these problems: the most
common approach, making use of variable splitting (e.g. see examples
using [ADMM](denoise_tv_admm.rst) and
[Proximal ADMM](denoise_tv_multi.rst)), and via use of an approximatation
to the proximal operator of the TV norm, as implemented in
[IsotropicTVNorm](../_autosummary/scico.functional.rst#scico.functional.IsotropicTVNorm)
and
[AnisotropicTVNorm](../_autosummary/scico.functional.rst#scico.functional.AnisotropicTVNorm),
allowing the use of the proximal gradient method, which does not support
variable splitting.
"""


import matplotlib
from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, plot
from scico.optimize import AcceleratedPGM, ProximalADMM
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
Denoise with isotropic total variation, solved via Proximal ADMM.
"""
λ_iso = 1.0e0
f = loss.SquaredL2Loss(y=y)
g_iso = λ_iso * functional.L21Norm()
C = linop.FiniteDifference(input_shape=x_gt.shape, circular=True)
mu, nu = ProximalADMM.estimate_parameters(C)

solver = ProximalADMM(
    f=f,
    g=g_iso,
    A=C,
    rho=1e0,
    mu=mu,
    nu=nu,
    x0=y,
    maxiter=200,
    itstat_options={"display": True, "period": 20},
)
print(f"Solving on {device_info()}\n")
x_iso = solver.solve()
print()


"""
Denoise with anisotropic total variation, solved via Proximal ADMM.
"""
# Tune the weight to give the same data fidelity as the isotropic case.
λ_aniso = 8.68e-1
g_aniso = λ_aniso * functional.L1Norm()

solver = ProximalADMM(
    f=f,
    g=g_aniso,
    A=C,
    rho=1e0,
    mu=mu,
    nu=nu,
    x0=y,
    maxiter=200,
    itstat_options={"display": True, "period": 20},
)
x_aniso = solver.solve()
print()


"""
Denoise with isotropic total variation, solved using an approximation of
the TV norm proximal operator.
"""
h = λ_iso * functional.IsotropicTVNorm()
solver = AcceleratedPGM(
    f=f, g=h, L0=3e2, x0=y, maxiter=250, itstat_options={"display": True, "period": 20}
)
x_iso_aprx = solver.solve()
print()


"""
Denoise with anisotropic total variation, solved using an approximation
of the TV norm proximal operator.
"""
h = λ_aniso * functional.AnisotropicTVNorm()
solver = AcceleratedPGM(
    f=f, g=h, L0=3e2, x0=y, maxiter=250, itstat_options={"display": True, "period": 20}
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
