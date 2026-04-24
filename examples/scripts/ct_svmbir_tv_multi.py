#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
TV-Regularized CT Reconstruction (Multiple Algorithms)
======================================================

This example demonstrates the use of different optimization algorithms to
solve the TV-regularized CT problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where $A$ is the X-ray transform (implemented using the SVMBIR
:cite:`svmbir-2020` tomographic projection), $\mathbf{y}$ is the sinogram,
$C$ is a 2D finite difference operator, and $\mathbf{x}$ is the
reconstructed image.
"""

import numpy as np

import komplot as kplt
import matplotlib
import svmbir
from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import functional, linop, metric
from scico.linop import Diagonal
from scico.linop.xray.svmbir import SVMBIRSquaredL2Loss, XRayTransform
from scico.optimize import PDHG, LinearizedADMM
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Generate a ground truth image.
"""
N = 256  # image size
density = 0.025  # attenuation density of the image
np.random.seed(1234)
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.005], gap=2e-3, porosity=1.0), size=N - 10)
x_gt = x_gt / np.max(x_gt) * density
x_gt = np.pad(x_gt, 5)
x_gt[x_gt < 0] = 0


"""
Generate tomographic projector and sinogram.
"""
num_angles = int(N / 2)
num_channels = N
angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
A = XRayTransform(x_gt.shape, angles, num_channels)
sino = A @ x_gt


"""
Impose Poisson noise on sinogram. Higher max_intensity means less noise.
"""
max_intensity = 2000
expected_counts = max_intensity * np.exp(-sino)
noisy_counts = np.random.poisson(expected_counts).astype(np.float32)
noisy_counts[noisy_counts == 0] = 1  # deal with 0s
y = -snp.log(noisy_counts / max_intensity)


"""
Reconstruct using default prior of SVMBIR :cite:`svmbir-2020`.
"""
weights = svmbir.calc_weights(y, weight_type="transmission")

x_mrf = svmbir.recon(
    np.array(y[:, np.newaxis]),
    np.array(angles),
    weights=weights[:, np.newaxis],
    num_rows=N,
    num_cols=N,
    positivity=True,
    verbose=0,
)[0]


"""
Set up problem.
"""
x0 = snp.array(x_mrf)
weights = snp.array(weights)
λ = 1e-1  # ℓ1 norm regularization parameter
f = SVMBIRSquaredL2Loss(y=y, A=A, W=Diagonal(weights), scale=0.5)
g = λ * functional.L21Norm()  # regularization functional
# The append=0 option makes the results of horizontal and vertical finite
# differences the same shape, which is required for the L21Norm.
C = linop.FiniteDifference(input_shape=x_gt.shape, append=0)


"""
Solve via ADMM.
"""
solve_admm = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[2e1],
    x0=x0,
    maxiter=50,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 10}),
    itstat_options={"display": True, "period": 10},
)
print(f"Solving on {device_info()}\n")
print("ADMM:")
x_admm = solve_admm.solve()
hist_admm = solve_admm.itstat_object.history(transpose=True)
print(f"PSNR: {metric.psnr(x_gt, x_admm):.2f} dB\n")


"""
Solve via Linearized ADMM.
"""
solver_ladmm = LinearizedADMM(
    f=f,
    g=g,
    C=C,
    mu=3e-2,
    nu=2e-1,
    x0=x0,
    maxiter=50,
    itstat_options={"display": True, "period": 10},
)
print("Linearized ADMM:")
x_ladmm = solver_ladmm.solve()
hist_ladmm = solver_ladmm.itstat_object.history(transpose=True)
print(f"PSNR: {metric.psnr(x_gt, x_ladmm):.2f} dB\n")


"""
Solve via PDHG.
"""
solver_pdhg = PDHG(
    f=f,
    g=g,
    C=C,
    tau=2e-2,
    sigma=8e0,
    x0=x0,
    maxiter=50,
    itstat_options={"display": True, "period": 10},
)
print("PDHG:")
x_pdhg = solver_pdhg.solve()
hist_pdhg = solver_pdhg.itstat_object.history(transpose=True)
print(f"PSNR: {metric.psnr(x_gt, x_pdhg):.2f} dB\n")


"""
Show the recovered images.
"""
norm = matplotlib.colors.Normalize(vmin=-0.1 * density, vmax=1.2 * density)
fig, ax = kplt.subplots(1, 2, figsize=[10, 5])
kplt.imview(img=x_gt, title="Ground Truth Image", show_cbar=True, ax=ax[0], norm=norm)
kplt.imview(
    img=x_mrf,
    title=f"MRF (PSNR: {metric.psnr(x_gt, x_mrf):.2f} dB)",
    show_cbar=True,
    ax=ax[1],
    norm=norm,
)
fig.show()

fig, ax = kplt.subplots(1, 3, figsize=[15, 5])
kplt.imview(
    img=x_admm,
    title=f"TV ADMM (PSNR: {metric.psnr(x_gt, x_admm):.2f} dB)",
    show_cbar=True,
    ax=ax[0],
    norm=norm,
)
kplt.imview(
    img=x_ladmm,
    title=f"TV LinADMM (PSNR: {metric.psnr(x_gt, x_ladmm):.2f} dB)",
    show_cbar=True,
    ax=ax[1],
    norm=norm,
)
kplt.imview(
    img=x_pdhg,
    title=f"TV PDHG (PSNR: {metric.psnr(x_gt, x_pdhg):.2f} dB)",
    show_cbar=True,
    ax=ax[2],
    norm=norm,
)
fig.show()


"""
Plot convergence statistics.
"""
fig, ax = kplt.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(27, 6))
kplt.plot(
    snp.array((hist_admm.Objective, hist_ladmm.Objective, hist_pdhg.Objective)).T,
    ylog=True,
    title="Objective function",
    xlabel="Iteration",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[0],
)
kplt.plot(
    snp.array((hist_admm.Prml_Rsdl, hist_ladmm.Prml_Rsdl, hist_pdhg.Prml_Rsdl)).T,
    ylog=True,
    title="Primal residual",
    xlabel="Iteration",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[1],
)
kplt.plot(
    snp.array((hist_admm.Dual_Rsdl, hist_ladmm.Dual_Rsdl, hist_pdhg.Dual_Rsdl)).T,
    ylog=True,
    title="Dual residual",
    xlabel="Iteration",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[2],
)
fig.show()

fig, ax = kplt.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(27, 6))
kplt.plot(
    snp.array((hist_admm.Objective, hist_ladmm.Objective, hist_pdhg.Objective)).T,
    snp.array((hist_admm.Time, hist_ladmm.Time, hist_pdhg.Time)).T,
    ylog=True,
    title="Objective function",
    xlabel="Time (s)",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[0],
)
kplt.plot(
    snp.array((hist_admm.Prml_Rsdl, hist_ladmm.Prml_Rsdl, hist_pdhg.Prml_Rsdl)).T,
    snp.array((hist_admm.Time, hist_ladmm.Time, hist_pdhg.Time)).T,
    ylog=True,
    title="Primal residual",
    xlabel="Time (s)",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[1],
)
kplt.plot(
    snp.array((hist_admm.Dual_Rsdl, hist_ladmm.Dual_Rsdl, hist_pdhg.Dual_Rsdl)).T,
    snp.array((hist_admm.Time, hist_ladmm.Time, hist_pdhg.Time)).T,
    ylog=True,
    title="Dual residual",
    xlabel="Time (s)",
    legend=("ADMM", "LinADMM", "PDHG"),
    ax=ax[2],
)
fig.show()


input("\nWaiting for input to close figures and exit")
