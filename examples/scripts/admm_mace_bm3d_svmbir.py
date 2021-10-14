#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D and SVMBIR)
================================================================

This example demonstrates the use of class
[admm.ADMM](../_autosummary/scico.admm.rst#scico.admm.ADMM) to solve a
tomographic reconstruction problem using the Plug-and-Play Priors framework
:cite:`venkatakrishnan-2013-plugandplay2`, using BM3D :cite:`dabov-2008-image`
as a denoiser and SVMBIR :cite:`svmbir-2020` for tomographic projection.

"""
import numpy as np

import jax

import matplotlib.pyplot as plt
import svmbir
from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import metric, plot
from scico.admm import ADMM, LinearSubproblemSolver
from scico.functional import BM3D, NonNegativeIndicator
from scico.linop import Diagonal, Identity
from scico.linop.radon_svmbir import ParallelBeamProjector, SVMBIRWeightedSquaredL2Loss


def gen_phantom(N, density):
    np.random.seed(1234)
    phantom = discrete_phantom(Foam(size_range=[0.05, 0.02], gap=0.02, porosity=0.3), size=N - 10)
    phantom = phantom / np.max(phantom) * density
    phantom = np.pad(phantom, 5)
    phantom[phantom < 0] = 0
    return phantom


def poisson_sino(sino, max_intensity=2000):
    """Create sinogram with Poisson noise. Higher max_intensity means
    less noise.
    """
    expected_counts = max_intensity * np.exp(-sino)
    noisy_counts = np.random.poisson(expected_counts).astype(np.float32)
    noisy_counts[noisy_counts == 0] = 1  # deal with 0s
    noisy_sino = -np.log(noisy_counts / max_intensity)
    return noisy_sino


"""
Generate a ground truth image.
"""
N = 256  # image size
density = 0.025  # attenuation density of the image
x_gt = gen_phantom(N, density)

"""
Generate tomographic projector and sinogram.
"""
num_angles = int(N / 2)
num_channels = N
angles = snp.linspace(0, snp.pi, num_angles, dtype=snp.float32)
A = ParallelBeamProjector(x_gt.shape, angles, num_channels)
sino = A @ x_gt

"""
Add noise to sinogram.
"""
y = poisson_sino(sino, max_intensity=2000)

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
Set up an ADMM solver.
"""
y, x0, weights = jax.device_put([y, x_mrf, weights])

ρ = 100  # denoiser weight (inverse data fidelity weight)
σ = density * 0.2  # denoiser sigma

weight_op = Diagonal(weights ** 0.5)

f = SVMBIRWeightedSquaredL2Loss(y=y, A=A, weight_op=weight_op, scale=0.5)
g0 = σ * ρ * BM3D()
g1 = NonNegativeIndicator()

solver = ADMM(
    f=None,
    g_list=[f, g0, g1],
    C_list=[Identity(x_mrf.shape), Identity(x_mrf.shape), Identity(x_mrf.shape)],
    rho_list=[ρ, ρ, ρ],
    x0=x0,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 100}),
    verbose=True,
)


x_bm3d = solver.solve()
hist = solver.itstat_object.history(transpose=True)

"""
Show the recovered image.
"""
norm = plot.matplotlib.colors.Normalize(vmin=-0.1 * density, vmax=1.2 * density)
fig, ax = plt.subplots(1, 3, figsize=[15, 5])
plot.imview(img=x_gt, title="Ground Truth Image", cbar=True, fig=fig, ax=ax[0], norm=norm)
plot.imview(
    img=x_mrf,
    title=f"MRF (PSNR: {metric.psnr(x_gt, x_mrf):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[1],
    norm=norm,
)
plot.imview(
    img=x_bm3d,
    title=f"BM3D (PSNR: {metric.psnr(x_gt, x_bm3d):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[2],
    norm=norm,
)
fig.show()

"""
Plot convergence statistics.
"""
plot.plot(
    snp.vstack((hist.Primal_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
)


input("\nWaiting for input to close figures and exit")
