#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
PPP (with BM3D) Fan-Beam CT Reconstruction
==========================================

This example demonstrates solution of a fan-beam tomographic reconstruction
problem using the Plug-and-Play Priors framework
:cite:`venkatakrishnan-2013-plugandplay2`, using BM3D
:cite:`dabov-2008-image` as a denoiser and SVMBIR
:cite:`svmbir-2020` for tomographic projection.

This example uses the data fidelity term as one of the ADMM $g$
functionals so that the optimization with respect to the data fidelity is
able to exploit the internal prox of the `SVMBIRExtendedLoss` functional.

We solve the problem in two different ways:
1. Approximating the fan-beam geometry using parallel-beam and using the
   parallel beam projector to compute the reconstruction.
2. Using the correct fan-beam geometry to perform a reconstruction.
"""

import numpy as np

import jax

import matplotlib.pyplot as plt
import svmbir
from matplotlib.ticker import MaxNLocator
from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import metric, plot
from scico.functional import BM3D
from scico.linop import Diagonal, Identity
from scico.linop.radon_svmbir import SVMBIRExtendedLoss, TomographicProjector
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Generate a ground truth image.
"""
N = 256  # image size
density = 0.025  # attenuation density of the image
np.random.seed(1234)
pad_len = 5
x_gt = discrete_phantom(Foam(size_range=[0.05, 0.02], gap=0.02, porosity=0.3), size=N - 2 * pad_len)
x_gt = x_gt / np.max(x_gt) * density
x_gt = np.pad(x_gt, pad_len)
x_gt[x_gt < 0] = 0


"""
Generate tomographic projector and sinogram for fan beam and parallel beam.
For fan beam, use view angles spanning 2π since unlike parallel beam, views
at 0 and π are not equivalent.
"""
num_angles = int(N / 2)
num_channels = N

# Use angles in the range [0, 2*pi] for fan beam
angles = snp.linspace(0, 2 * snp.pi, num_angles, endpoint=False, dtype=snp.float32)

dist_source_detector = 1500.0
magnification = 1.2
A_fan = TomographicProjector(
    x_gt.shape,
    angles,
    num_channels,
    geometry="fan-curved",
    dist_source_detector=dist_source_detector,
    magnification=magnification,
)
A_parallel = TomographicProjector(
    x_gt.shape,
    angles,
    num_channels,
    geometry="parallel",
)

sino_fan = A_fan @ x_gt


"""
Impose Poisson noise on sinograms. Higher max_intensity means less noise.
"""


def add_poisson_noise(sino, max_intensity):
    expected_counts = max_intensity * np.exp(-sino)
    noisy_counts = np.random.poisson(expected_counts).astype(np.float32)
    noisy_counts[noisy_counts == 0] = 1  # deal with 0s
    y = -np.log(noisy_counts / max_intensity)

    return y


y_fan = add_poisson_noise(sino_fan, max_intensity=500)


"""
Reconstruct using default prior of SVMBIR :cite:`svmbir-2020`.
"""
weights_fan = svmbir.calc_weights(y_fan, weight_type="transmission")

x_mrf_fan = svmbir.recon(
    np.array(y_fan[:, np.newaxis]),
    np.array(angles),
    weights=weights_fan[:, np.newaxis],
    num_rows=N,
    num_cols=N,
    positivity=True,
    verbose=0,
    stop_threshold=0.0,
    geometry="fan-curved",
    dist_source_detector=dist_source_detector,
    magnification=magnification,
    delta_channel=1.0,
    delta_pixel=1.0 / magnification,
)[0]

x_mrf_parallel = svmbir.recon(
    np.array(y_fan[:, np.newaxis]),
    np.array(angles),
    weights=weights_fan[:, np.newaxis],
    num_rows=N,
    num_cols=N,
    positivity=True,
    verbose=0,
    stop_threshold=0.0,
    geometry="parallel",
)[0]


"""
Push arrays to device.
"""
y_fan, x0_fan, weights_fan = jax.device_put([y_fan, x_mrf_fan, weights_fan])
x0_parallel = jax.device_put(x_mrf_parallel)


"""
Set problem parameters and BM3D pseudo-functional.
"""
ρ = 10  # ADMM penalty parameter
σ = density * 0.6  # denoiser sigma
g0 = σ * ρ * BM3D()


"""
Set up problem using `SVMBIRExtendedLoss`.
"""
f_extloss_fan = SVMBIRExtendedLoss(
    y=y_fan,
    A=A_fan,
    W=Diagonal(weights_fan),
    scale=0.5,
    positivity=True,
    prox_kwargs={"maxiter": 5, "ctol": 0.0},
)
f_extloss_parallel = SVMBIRExtendedLoss(
    y=y_fan,
    A=A_parallel,
    W=Diagonal(weights_fan),
    scale=0.5,
    positivity=True,
    prox_kwargs={"maxiter": 5, "ctol": 0.0},
)

solver_extloss_fan = ADMM(
    f=None,
    g_list=[f_extloss_fan, g0],
    C_list=[Identity(x_mrf_fan.shape), Identity(x_mrf_fan.shape)],
    rho_list=[ρ, ρ],
    x0=x0_fan,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 100}),
    itstat_options={"display": True},
)
solver_extloss_parallel = ADMM(
    f=None,
    g_list=[f_extloss_parallel, g0],
    C_list=[Identity(x_mrf_parallel.shape), Identity(x_mrf_parallel.shape)],
    rho_list=[ρ, ρ],
    x0=x0_parallel,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 100}),
    itstat_options={"display": True},
)


"""
Run the ADMM solvers.
"""
print(f"Solving on {device_info()}\n")
x_extloss_fan = solver_extloss_fan.solve()
hist_extloss_fan = solver_extloss_fan.itstat_object.history(transpose=True)

print()
x_extloss_parallel = solver_extloss_parallel.solve()
hist_extloss_parallel = solver_extloss_parallel.itstat_object.history(transpose=True)


"""
Show the recovered images. The parallel beam reconstruction is poor because
the parallel beam is a poor approximation of the specific fan beam geometry
used here.
"""
norm = plot.matplotlib.colors.Normalize(vmin=-0.1 * density, vmax=1.2 * density)

fig, ax = plt.subplots(1, 3, figsize=(20, 7))
plot.imview(img=x_gt, title="Ground Truth Image", cbar=True, fig=fig, ax=ax[0], norm=norm)
plot.imview(
    img=x_mrf_parallel,
    title=f"Parallel-beam MRF (PSNR: {metric.psnr(x_gt, x_mrf_parallel):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[1],
    norm=norm,
)
plot.imview(
    img=x_extloss_parallel,
    title=f"Parallel-beam Extended Loss (PSNR: {metric.psnr(x_gt, x_extloss_parallel):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[2],
    norm=norm,
)
fig.show()


fig, ax = plt.subplots(1, 3, figsize=(20, 7))
plot.imview(img=x_gt, title="Ground Truth Image", cbar=True, fig=fig, ax=ax[0], norm=norm)
plot.imview(
    img=x_mrf_fan,
    title=f"Fan-beam MRF (PSNR: {metric.psnr(x_gt, x_mrf_fan):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[1],
    norm=norm,
)
plot.imview(
    img=x_extloss_fan,
    title=f"Fan-beam Extended Loss (PSNR: {metric.psnr(x_gt, x_extloss_fan):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[2],
    norm=norm,
)
fig.show()


"""
Plot convergence statistics.
"""
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plot.plot(
    snp.vstack((hist_extloss_parallel.Prml_Rsdl, hist_extloss_parallel.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals for parallel-beam reconstruction",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[0],
)
ax[0].set_ylim([5e-3, 1e0])
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
plot.plot(
    snp.vstack((hist_extloss_fan.Prml_Rsdl, hist_extloss_fan.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals for fan-beam reconstruction",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[1],
)
ax[1].set_ylim([5e-3, 1e0])
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
fig.show()


input("\nWaiting for input to close figures and exit")
