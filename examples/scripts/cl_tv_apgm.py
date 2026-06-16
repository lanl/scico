#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
TV-Regularized Computed Laminography Reconstruction
===================================================

This example demonstrates solution of a sparse-view CT reconstruction
problem with isotropic total variation (TV) regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - A \mathbf{x}
  \|_2^2 + \lambda \| C \mathbf{x} \|_{2,1} \;,$$

where $A$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $C$ is a 2D finite difference operator, and
$\mathbf{x}$ is the reconstructed image. This example uses the CT
projector integrated into scico, while the companion
[example script](ct_astra_tv_admm.rst) uses the projector provided by
the astra package.
"""

import numpy as np

import jax

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, optimize, plot
from scico.examples import create_laminar_phantom
from scico.linop.xray import XRayTransform3D as scicoXRayTransform3D
from scico.linop.xray import cl_angles_to_vecs, cl_fbp
from scico.util import device_info

try:
    import astra

    from scico.linop.xray.astra import XRayTransform3D as astraXRayTransform3D
except ImportError:
    have_astra = True
else:
    have_astra = False

have_gpu = True if jax.devices()[0].platform == "gpu" else False


"""
Create a ground truth image.
"""
x_gt = create_laminar_phantom()


"""
Configure projection operator and generate synthetic measurements.
"""
vol_shape = x_gt.shape
det_shape = (int(vol_shape[1] * 0.85), int(vol_shape[2] * 1.4))
num_views = 180
alpha = 61.0 * np.pi / 180.0
theta = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
vectors = cl_angles_to_vecs(theta, alpha)

if have_astra and have_gpu:
    X = astraXRayTransform3D(
        vol_shape,
        det_count=det_shape,
        vectors=vectors,
    )
else:
    matrices = astra.convert_to_scico_geometry(
        input_shape=vol_shape, det_count=det_shape, vectors=vectors
    )
    X = scicoXRayTransform3D(vol_shape, matrices, det_shape)


y = X @ x_gt
yn = y + 0.05 * np.random.rand(*y.shape).astype(np.float32)


x_fbp = snp.clip(cl_fbp(y, alpha, X), 0.0, 1.0)


"""
Set up problem functional and APGM solver object.
"""

f = loss.SquaredL2Loss(y=yn, A=X)
λ = 1e0
g = λ * functional.IsotropicTVNorm(circular=False, input_shape=vol_shape)
maxiter = 200
print("Estimating L0")
L0 = linop.operator_norm(X, maxiter=10) ** 2

solver = optimize.AcceleratedPGM(
    f=f,
    g=g,
    L0=L0,
    x0=x_fbp,
    maxiter=maxiter,
    itstat_options={"display": True, "period": 20},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x_rec = snp.clip(solver.solve(), 0.0, 1.0)
hist = solver.itstat_object.history(transpose=True)


"""
Show the recovered image.
"""

slice_index = 32
fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 5))
plot.imview(x_gt[slice_index], title="Ground truth", cbar=None, cmap="viridis", fig=fig, ax=ax[0])
plot.imview(
    x_fbp[slice_index],
    title="FBP Reconstruction: \nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt, x_fbp), metric.mae(x_gt, x_fbp)),
    cbar=None,
    cmap="viridis",
    fig=fig,
    ax=ax[1],
)
plot.imview(
    x_rec[slice_index],
    title="TV Reconstruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(x_gt, x_rec), metric.mae(x_gt, x_rec)),
    cmap="viridis",
    fig=fig,
    ax=ax[2],
)
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[2].get_images()[0], cax=cax, label="arbitrary units")
fig.show()


"""
Plot convergence statistics.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    hist.Objective,
    title="Objective function",
    xlbl="Iteration",
    ylbl="Functional value",
    fig=fig,
    ax=ax[0],
)
plot.plot(
    hist.Residual,
    ptyp="semilogy",
    title="Residual",
    xlbl="Iteration",
    fig=fig,
    ax=ax[1],
)
fig.show()


input("\nWaiting for input to close figures and exit")
