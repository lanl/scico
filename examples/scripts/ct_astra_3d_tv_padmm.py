#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction (PADMM)
=======================================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where $A$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $C$ is a 3D finite difference operator,
and $\mathbf{x}$ is the desired image.
"""


import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.xray.astra import FlexibleXRayTransform
from scico.optimize import ProximalADMM
from scico.util import device_info

"""
Create a ground truth image and projector.
"""
Nx = 128
Ny = 256
Nz = 64

tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

n_projection = 10  # number of projections
angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles
det_spacing = [1.0, 1.0]
det_count = [Nz, max(Nx, Ny)]
vectors = np.zeros((n_projection, 12))
vectors[:, 0] = np.sin(angles)
vectors[:, 1] = -np.cos(angles)
vectors[:, 6] = np.cos(angles) * det_spacing[0]
vectors[:, 7] = np.sin(angles) * det_spacing[0]
vectors[:, 11] = det_spacing[1]

C = FlexibleXRayTransform(tangle.shape, det_count, vectors)  # CT projection operator
y = C @ tangle  # sinogram


"""
Set up ADMM solver object.
"""
𝛼 = 1e2
λ = 2e0 / 𝛼  # ℓ2,1 norm regularization parameter
ρ = 1e-2  # ADMM penalty parameter
maxiter = 1000  # number of ADMM iterations

f = functional.ZeroFunctional()
g0 = loss.SquaredL2Loss(y=y)
g1 = λ * functional.L21Norm()
g = functional.SeparableFunctional((g0, g1))
D = linop.FiniteDifference(input_shape=tangle.shape, append=0)

A = linop.VerticalStack((C, 𝛼 * D))
mu, nu = ProximalADMM.estimate_parameters(A)
# print(mu, nu)

solver = ProximalADMM(
    f=f,
    g=g,
    A=A,
    B=None,
    rho=ρ,
    mu=mu,
    nu=nu,
    # x0=C.adj(y),
    maxiter=maxiter,
    itstat_options={"display": True, "period": 50},
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
