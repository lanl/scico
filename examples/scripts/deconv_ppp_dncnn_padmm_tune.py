#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Image Deconvolution Parameter Tuning (PPP/DnCNN/Proximal ADMM)
==============================================================

This example demonstrates the use of
[scico.ray.tune](../_autosummary/scico.ray.tune.rst) to tune parameters
for the companion [example script](deconv_tv_admm.rst).

This script is hard-coded to run on CPU only to avoid the large number of
warnings that are emitted when GPU resources are requested but not available,
and due to the difficulty of supressing these warnings in a way that does
not force use of the CPU only. To enable GPU usage, comment out the
`os.environ` statements near the beginning of the script, and change the
value of the "gpu" entry in the `resources` dict from 0 to 1. Note that
two environment variables are set to suppress the warnings because
`JAX_PLATFORMS` was intended to replace `JAX_PLATFORM_NAME` but this change
has yet to be correctly implemented
(see [google/jax#6805](https://github.com/google/jax/issues/6805) and
[google/jax#10272](https://github.com/google/jax/pull/10272)).
"""

# isort: off
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.numpy as snp
import scico.ray as ray
from scico import functional, linop, loss, metric, plot, random
from scico.optimize import ProximalADMM
from scico.ray import tune

"""
Create a ground truth image.
"""
np.random.seed(1234)
N = 512  # image size
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.0025], gap=1e-3, porosity=1), size=N)
x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU


"""
Set up the forward operator and create a test signal consisting of a
blurred signal with additive Gaussian noise.
"""
n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / (n * n)
A = linop.Convolve(h=psf, input_shape=x_gt.shape)

mu, nu = ProximalADMM.estimate_parameters(A)

Ax = A(x_gt)  # blurred image
noise, key = random.randn(Ax.shape)
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
    f = λ * functional.DnCNN(variant="6N")
    g = loss.SquaredL2Loss(y=y)
    g.has_eval = False  # temporary scico bug workaround
    # Define solver.
    solver = ProximalADMM(
        f=f,
        g=g,
        A=A,
        B=None,
        rho=ρ,
        mu=mu,
        nu=nu,
        x0=A.T @ y,
        ##maxiter=10,
        # maxiter=5
        maxiter=12,
    )
    ## Perform 15 iterations, reporting performance to ray.tune every 5 iterations.
    # for step in range(3):
    #    x_padmm = solver.solve()
    #    reporter(psnr=float(metric.psnr(x_gt, x_padmm)))
    x_padmm = snp.clip(solver.solve(), 0, 1)
    reporter(psnr=float(metric.psnr(x_gt, x_padmm)))


"""
Define parameter search space and resources per trial.
"""
# config = {"lambda": tune.loguniform(1e-2, 1e0), "rho": tune.loguniform(1e-1, 1e1)}
config = {"lambda": tune.loguniform(1e-2, 1e-1), "rho": tune.loguniform(1e-1, 2e0)}
resources = {"cpu": 4, "gpu": 0}  # cpus per trial, gpus per trial


"""
Run parameter search.
"""
analysis = tune.run(
    eval_params,
    metric="psnr",
    mode="max",
    num_samples=200,
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


pltx = [t.config["rho"] for t in analysis.trials]
plty = [t.config["lambda"] for t in analysis.trials]
pltz = [t.metric_analysis["psnr"]["max"] for t in analysis.trials]
pltx, plty, pltz = zip(*filter(lambda x: x[2] >= 20.0, zip(pltx, plty, pltz)))

fig, ax = plot.subplots(figsize=(10, 8))
sc = ax.scatter(pltx, plty, c=pltz, cmap=plot.cm.plasma_r)
fig.colorbar(sc)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\lambda$")
fig.show()


input("\nWaiting for input to close figures and exit")
