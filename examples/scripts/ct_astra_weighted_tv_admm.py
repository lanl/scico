#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
TV-Regularized Low-Dose CT Reconstruction
=========================================

This example demonstrates solution of a low-dose CT reconstruction
problem with isotropic total variation (TV) regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_W^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where $A$ is the X-ray transform (the CT forward projection),
$\mathbf{y}$ is the sinogram, the norm weighting $W$ is chosen so that
the weighted norm is an approximation to the Poisson negative log
likelihood :cite:`sauer-1993-local`, $C$ is a 2D finite difference
operator, and $\mathbf{x}$ is the reconstructed image.
"""

import numpy as np

from xdesign import Soil, discrete_phantom

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.linop.xray.astra import XRayTransform2D
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Create a ground truth image.
"""
N = 512  # phantom size
np.random.seed(0)
x_gt = discrete_phantom(Soil(porosity=0.80), size=384)
x_gt = np.ascontiguousarray(np.pad(x_gt, (64, 64)))
x_gt = np.clip(x_gt, 0, np.inf)  # clip to positive values
x_gt = snp.array(x_gt)  # convert to jax type


"""
Configure CT projection operator and generate synthetic measurements.
"""
n_projection = 360  # number of projections
Io = 1e3  # source flux
𝛼 = 1e-2  # attenuation coefficient
angles = np.linspace(0, 2 * np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
A = XRayTransform2D(x_gt.shape, N, 1.0, angles)  # CT projection operator
y_c = A @ x_gt  # sinogram


r"""
Add Poisson noise to projections according to

$$\mathrm{counts} \sim \mathrm{Poi}\left(I_0 \exp (- \alpha A
\mathbf{x} ) \right)$$

$$\mathbf{y} = - \frac{1}{\alpha} \log\left(\mathrm{counts} /
I_0\right) \;.$$

We use the NumPy random functionality so we can generate using 64-bit
numbers.
"""
counts = np.random.poisson(Io * snp.exp(-𝛼 * A @ x_gt))
counts = np.clip(counts, a_min=1, a_max=np.inf)  # replace any 0s count with 1
y = -1 / 𝛼 * np.log(counts / Io)
y = snp.array(y)  # convert back to float32 as a jax array


"""
Set up post processing. For this example, we clip all reconstructions
to the range of the ground truth.
"""


def postprocess(x):
    return snp.clip(x, 0, snp.max(x_gt))


"""
Compute an FBP reconstruction as an initial guess.
"""
x0 = postprocess(A.fbp(y))


r"""
Set up and solve the un-weighted reconstruction problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_{2,1} \;.$$
"""
# Note that rho and lambda were selected via a parameter sweep (not
# shown here).
ρ = 2.5e3  # ADMM penalty parameter
lambda_unweighted = 3e2  # regularization strength
maxiter = 100  # number of ADMM iterations
cg_tol = 1e-5  # CG relative tolerance
cg_maxiter = 10  # maximum CG iterations per ADMM iteration
f = loss.SquaredL2Loss(y=y, A=A)
admm_unweighted = ADMM(
    f=f,
    g_list=[lambda_unweighted * functional.L21Norm()],
    C_list=[linop.FiniteDifference(x_gt.shape, append=0)],
    rho_list=[ρ],
    x0=x0,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 10},
)
print(f"Solving on {device_info()}\n")
admm_unweighted.solve()
x_unweighted = postprocess(admm_unweighted.x)


r"""
Set up and solve the weighted reconstruction problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_W^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where

  $$W = \mathrm{diag}( \mathrm{counts} / I_0 ) \;.$$

The data fidelity term in this formulation follows
:cite:`sauer-1993-local` (9) except for the scaling by $I_0$, which we
use to maintain balance between the data and regularization terms if
$I_0$ changes.
"""
lambda_weighted = 5e1
weights = snp.array(counts / Io)
f = loss.SquaredL2Loss(y=y, A=A, W=linop.Diagonal(weights))
admm_weighted = ADMM(
    f=f,
    g_list=[lambda_weighted * functional.L21Norm()],
    C_list=[linop.FiniteDifference(x_gt.shape, append=0)],
    rho_list=[ρ],
    maxiter=maxiter,
    x0=x0,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 10},
)
print()
admm_weighted.solve()
x_weighted = postprocess(admm_weighted.x)


"""
Show recovered images.
"""


def plot_recon(x, title, ax):
    """Plot an image with title indicating error metrics."""
    plot.imview(
        x,
        title=f"{title}\nSNR: {metric.snr(x_gt, x):.2f} (dB), MAE: {metric.mae(x_gt, x):.3f}",
        fig=fig,
        ax=ax,
    )


fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(11, 10))
plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0, 0])
plot_recon(x0, "FBP Reconstruction", ax=ax[0, 1])
plot_recon(x_unweighted, "Unweighted TV Reconstruction", ax=ax[1, 0])
plot_recon(x_weighted, "Weighted TV Reconstruction", ax=ax[1, 1])
for ax_ in ax.ravel():
    ax_.set_xlim(64, 448)
    ax_.set_ylim(64, 448)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05, wspace=0.2, hspace=0.01)
fig.colorbar(
    ax[0, 0].get_images()[0], ax=ax, location="right", shrink=0.9, pad=0.05, label="arbitrary units"
)
fig.show()


input("\nWaiting for input to close figures and exit")
