#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Image Deconvolution Parameter Tuning
====================================

This example demonstrates the use of
[scico.ray.tune](../_autosummary/scico.ray.tune.rst) to tune parameters
for the companion [example script](deconv_tv_admm.rst).
"""


import numpy as np

import jax

from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
import scico.ray as ray
from scico import functional, linop, loss, metric, plot
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.ray import tune

"""
Create a ground truth image.
"""
phantom = SiemensStar(32)
x_gt = snp.pad(discrete_phantom(phantom, 240), 8)


"""
Set up the forward operator and create a test signal consisting of a
blurred signal with additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / (n * n)
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = scico.random.randn(Ax.shape, seed=0)
y = Ax + σ * noise

"""
Put main arrays into ray object store.
"""
ray_x_gt, ray_psf, ray_y = ray.put(np.array(x_gt)), ray.put(np.array(psf)), ray.put(np.array(y))


"""
Define performance evaluation function.
"""


def eval_params(config, reporter):
    # Extract solver parameters from config dict.
    λ, ρ = config["lambda"], config["rho"]
    # Get main arrays from ray object store.
    x_gt, psf, y = ray.get([ray_x_gt, ray_psf, ray_y])
    # Put main arrays on jax device.
    x_gt, psf, y = jax.device_put([x_gt, psf, y])
    # Set up problem to be solved.
    A = linop.Convolve(h=psf, input_shape=x_gt.shape)
    f = loss.SquaredL2Loss(y=y, A=A)
    g = λ * functional.L1Norm()
    C = linop.FiniteDifference(input_shape=x_gt.shape)
    # Define solver.
    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[C],
        rho_list=[ρ],
        x0=A.adj(y),
        maxiter=10,
        subproblem_solver=LinearSubproblemSolver(),
    )
    # Perform 50 iterations, reporting performance to ray.tune every 10 iterations.
    for step in range(5):
        x_admm = solver.solve()
        reporter(psnr=float(metric.psnr(x_gt, x_admm)))


"""
Define parameter search space and resources per trial.
"""
config = {"lambda": tune.loguniform(1e-2, 1e0), "rho": tune.loguniform(1e-1, 1e1)}
resources = {"cpu": 4, "gpu": 0}  # cpus per trial, gpus per trial


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
plot.plot(
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
ax = fig.axes[0]
ax.set_xlim([config["rho"].lower, config["rho"].upper])
ax.set_ylim([config["lambda"].lower, config["lambda"].upper])
fig.show()


input("\nWaiting for input to close figures and exit")
