#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
TV-Regularized Abel Inversion
=============================

This example demonstrates a total variation (TV) regularized Abel
inversion by solving the problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_1 \;,$$

where $A$ is the Abel projector (with an implementation based on a
projector from PyAbel :cite:`pyabel-2022`), $\mathbf{y}$ is the measured
data, $C$ is a 2D finite difference operator, and $\mathbf{x}$ is the
solution.
"""

import numpy as np

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_circular_phantom
from scico.linop.xray.abel import AbelTransform
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Create a ground truth image.
"""
N = 256  # image size
x_gt = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])


"""
Set up the forward operator and create a test measurement.
"""
A = AbelTransform(x_gt.shape)
y = A @ x_gt
np.random.seed(12345)
y = y + np.random.normal(size=y.shape).astype(np.float32)


"""
Compute inverse Abel transform solution.
"""
x_inv = A.inverse(y)


"""
Set up the problem to be solved. Anisotropic TV, which gives slightly
better performance than isotropic TV for this problem, is used here.
"""
f = loss.SquaredL2Loss(y=y, A=A)
λ = 2.35e1  # ℓ1 norm regularization parameter
g = λ * functional.L1Norm()  # Note the use of anisotropic TV
C = linop.FiniteDifference(input_shape=x_gt.shape)


"""
Set up ADMM solver object.
"""
ρ = 1.03e2  # ADMM penalty parameter
maxiter = 100  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=snp.clip(x_inv, 0.0, 1.0),
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 10},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
solver.solve()
x_tv = snp.clip(solver.x, 0.0, 1.0)


"""
Show results.
"""
norm = plot.matplotlib.colors.Normalize(vmin=-0.1, vmax=1.2)
fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(12, 12))
plot.imview(x_gt, title="Ground Truth", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 0], norm=norm)
plot.imview(y, title="Measurement", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 1])
plot.imview(
    x_inv,
    title="Inverse Abel: %.2f (dB)" % metric.psnr(x_gt, x_inv),
    cmap=plot.cm.Blues,
    fig=fig,
    ax=ax[1, 0],
    norm=norm,
)
plot.imview(
    x_tv,
    title="TV-Regularized Inversion: %.2f (dB)" % metric.psnr(x_gt, x_tv),
    cmap=plot.cm.Blues,
    fig=fig,
    ax=ax[1, 1],
    norm=norm,
)
fig.show()


input("\nWaiting for input to close figures and exit")
