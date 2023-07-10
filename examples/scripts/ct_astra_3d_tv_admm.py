#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction
===============================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where $A$ is the Radon transform, $\mathbf{y}$ is the sinogram, $C$ is
a 3D finite difference operator, and $\mathbf{x}$ is the desired
image.
"""


import numpy as np

import jax

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.radon_astra import TomographicProjector
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Create a ground truth image and projector.
"""

Nx = 128
Ny = 256
Nz = 64

tangle = create_tangle_phantom(Nx, Ny, Nz)
tangle = jax.device_put(tangle)

n_projection = 10  # number of projections
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
A = TomographicProjector(
    tangle.shape, [1.0, 1.0], [Nz, max(Nx, Ny)], angles
)  # Radon transform operator
y = A @ tangle  # sinogram


"""
Set up ADMM solver object.
"""
λ = 2e0  # L1 norm regularization parameter
ρ = 5e0  # ADMM penalty parameter
maxiter = 25  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

# The append=0 option makes the results of horizontal and vertical
# finite differences the same shape, which is required for the L21Norm,
# which is used so that g(Cx) corresponds to isotropic TV.
C = linop.FiniteDifference(input_shape=tangle.shape, append=0)
g = λ * functional.L21Norm()

f = loss.SquaredL2Loss(y=y, A=A)

x0 = A.T(y)

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=x0,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)

"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
solver.solve()
hist = solver.itstat_object.history(transpose=True)
tangle_recon = solver.x

print(
    "TV Restruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon))
)

"""
Show the recovered image.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 5))
plot.imview(tangle[32], title="Ground truth (central slice)", cbar=None, fig=fig, ax=ax[0])

plot.imview(
    tangle_recon[32],
    title="TV Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon)),
    fig=fig,
    ax=ax[1],
)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
fig.show()

input("\nWaiting for input to close figures and exit")
