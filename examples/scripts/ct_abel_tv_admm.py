#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Abel Transform Demo
===================

This example demonstrates a TV-regularized Abel inversion using
an Abel projector based on PyAbel :cite:`pyabel-2022`
"""

import numpy as np

import jax

import scico.numpy as snp
from scico import functional, linop, loss, plot
from scico.linop.abel import AbelProjector
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info


def dist_map_2D(img_shape, center=None):
    """Computes a 2D map of the distance from a center pixel."""

    if center == None:
        center = [img_dim // 2 for img_dim in img_shape]

    coords = [np.arange(0, img_dim) for img_dim in img_shape]
    coord_mesh = np.meshgrid(*coords, sparse=True, indexing="ij")

    dist_map = sum([(coord_mesh[i] - center[i]) ** 2 for i in range(len(coord_mesh))])
    dist_map = np.sqrt(dist_map)

    return dist_map


def create_french_test_phantom(img_shape, radius_list, val_list, center=None):
    """Computes a french test object with given radii, and intensities."""

    dist_map = dist_map_2D(img_shape, center)

    img = np.zeros(img_shape)
    for r, val in zip(radius_list, val_list):
        img[dist_map < r] = val

    return img


x_gt = create_french_test_phantom((256, 254), [100, 50, 25], [1, 0, 0.5])
x_gt = jax.device_put(x_gt)

A = AbelProjector(x_gt.shape)
y = A @ x_gt
y = y + 1 * np.random.normal(size=y.shape).astype(np.float32)
ATy = A.T @ y


"""
Set up ADMM solver object.
"""
λ = 1.71e01  # L1 norm regularization parameter
ρ = 4.83e01  # ADMM penalty parameter
maxiter = 100  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

g = λ * functional.L1Norm()
C = linop.FiniteDifference(input_shape=x_gt.shape)

f = loss.SquaredL2Loss(y=y, A=A)

x_inv = A.inverse(y)
x0 = snp.clip(x_inv, 0, 1.0)

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
x_tv = snp.clip(solver.x, 0, 1.0)

norm = plot.matplotlib.colors.Normalize(vmin=-0.1, vmax=1.2)
fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(12, 12))
plot.imview(x_gt, title="Ground Truth", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 0], norm=norm)
plot.imview(y, title="Measurements", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 1])
plot.imview(x_inv, title="Inverse Abel", cmap=plot.cm.Blues, fig=fig, ax=ax[1, 0], norm=norm)
plot.imview(
    x_tv, title="TV Regularized Inversion", cmap=plot.cm.Blues, fig=fig, ax=ax[1, 1], norm=norm
)
fig.show()


input("\nWaiting for input to close figures and exit")
