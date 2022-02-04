#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Abel Transform Tuning Demo
===================

This example demonstrates the use of
[scico.ray.tune](../_autosummary/scico.ray.tune.rst) to tune
parameters for the companion [example script](ct_abel.rst).
"""

import numpy as np

import jax

import scico.ray as ray
from scico import functional, linop, loss, metric, plot
from scico.linop.abel import AbelProjector
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.ray import tune

"""
Create a ground truth image.
"""


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


x_gt = create_french_test_phantom((256, 256), [100, 50, 25], [1, 0, 0.5])


"""
Set up the forward operator and create a test measurement
"""
A = AbelProjector(x_gt.shape)
y = A @ x_gt
y = y + 1 * np.random.normal(size=y.shape)
ATy = A.T @ y

"""
Put main arrays into ray object store.
"""
ray_x_gt, ray_y = ray.put(np.array(x_gt)), ray.put(np.array(y))


"""
Define performance evaluation function.
"""


def eval_params(config):
    # Extract solver parameters from config dict.
    λ, ρ = config["lambda"], config["rho"]
    # Get main arrays from ray object store.
    x_gt, y = ray.get([ray_x_gt, ray_y])
    # Put main arrays on jax device.
    x_gt, y = jax.device_put([x_gt, y])
    # Set up problem to be solved.
    A = AbelProjector(x_gt.shape)
    f = loss.SquaredL2Loss(y=y, A=A)
    g = λ * functional.L1Norm()
    C = linop.FiniteDifference(input_shape=x_gt.shape)
    # Define solver.
    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[C],
        rho_list=[ρ],
        x0=A.inverse(y),
        maxiter=5,
        subproblem_solver=LinearSubproblemSolver(),
    )
    # Perform 50 iterations, reporting performance to ray.tune every 5 iterations.
    for step in range(10):
        x_admm = solver.solve()
        tune.report(psnr=float(metric.psnr(x_gt, x_admm)))


"""
Define parameter search space and resources per trial.
"""
config = {"lambda": tune.loguniform(1e-2, 1e3), "rho": tune.loguniform(1e-1, 1e3)}
resources = {"gpu": 0, "cpu": 1}  # gpus per trial, cpus per trial


"""
Run parameter search.
"""
analysis = tune.run(
    eval_params,
    metric="psnr",
    mode="max",
    num_samples=100,
    config=config,
    resources_per_trial=resources,
    hyperopt=True,
    verbose=True,
)

"""
Display best parameters and corresponding performance.
"""
best_config = analysis.get_best_config(metric="psnr", mode="max")
print(f"Best PSNR: {analysis.get_best_trial().last_result['psnr']:.2f} dB")
print("Best config: " + ", ".join([f"{k}: {v:.2e}" for k, v in best_config.items()]))


"""
Plot parameter values visited during parameter search. Marker sizes are
proportional to number of iterations run at each parameter pair. The best
point in the parameter space is indicated in red.
"""
fig = plot.figure(figsize=(8, 8))
for t in analysis.trials:
    n = t.metric_analysis["training_iteration"]["max"]
    plot.plot(
        t.config["lambda"],
        t.config["rho"],
        ptyp="loglog",
        lw=0,
        ms=(0.5 + 1.5 * n),
        marker="o",
        mfc="blue",
        mec="blue",
        fig=fig,
    )
_, ax = plot.plot(
    best_config["lambda"],
    best_config["rho"],
    ptyp="loglog",
    title="Parameter search sampling locations\n(marker size proportional to number of iterations)",
    xlbl=r"$\rho$",
    ylbl=r"$\lambda$",
    lw=0,
    ms=5.0,
    marker="o",
    mfc="red",
    mec="red",
    fig=fig,
)
ax.set_xlim([config["rho"].lower, config["rho"].upper])
ax.set_ylim([config["lambda"].lower, config["lambda"].upper])
fig.show()


input("\nWaiting for input to close figures and exit")
