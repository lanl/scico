#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Image Demosaicing (ADMM Plug-and-Play Priors w/ BM3D)
=====================================================

This example demonstrates the use of the ADMM Plug and Play Priors
(PPP) algorithm :cite:`venkatakrishnan-2013-plugandplay2` for solving
a raw image demosaicing problem.
"""

import numpy as np

import jax

from bm3d import bm3d_rgb
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

import scico
import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, plot
from scico.data import kodim23
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Read a ground truth image.
"""
img = kodim23(asfloat=True)[160:416, 60:316]
img = jax.device_put(img)  # convert to jax type, push to GPU


"""
Define demosaicing forward operator and its transpose.
"""


def Afn(x):
    """Map an RGB image to a single channel image with each pixel
    representing a single colour according to the colour filter array.
    """

    y = snp.zeros(x.shape[0:2])
    y = y.at[1::2, 1::2].set(x[1::2, 1::2, 0])
    y = y.at[0::2, 1::2].set(x[0::2, 1::2, 1])
    y = y.at[1::2, 0::2].set(x[1::2, 0::2, 1])
    y = y.at[0::2, 0::2].set(x[0::2, 0::2, 2])
    return y


def ATfn(x):
    """Back project a single channel raw image to an RGB image with zeros
    at the locations of undefined samples.
    """

    y = snp.zeros(x.shape + (3,))
    y = y.at[1::2, 1::2, 0].set(x[1::2, 1::2])
    y = y.at[0::2, 1::2, 1].set(x[0::2, 1::2])
    y = y.at[1::2, 0::2, 1].set(x[1::2, 0::2])
    y = y.at[0::2, 0::2, 2].set(x[0::2, 0::2])
    return y


"""
Define a baseline demosaicing function based on the demosaicing
algorithm of :cite:`menon-2007-demosaicing` from package
[colour_demosaicing](https://github.com/colour-science/colour-demosaicing).
"""


def demosaic(cfaimg):
    """Apply baseline demosaicing."""
    return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern="BGGR").astype(np.float32)


"""
Create a test image by color filter array sampling and adding Gaussian
white noise.
"""
s = Afn(img)
rgbshp = s.shape + (3,)  # shape of reconstructed RGB image
σ = 2e-2  # noise standard deviation
noise, key = scico.random.randn(s.shape, seed=0)
sn = s + σ * noise


"""
Compute a baseline demosaicing solution.
"""
imgb = jax.device_put(bm3d_rgb(demosaic(sn), 3 * σ).astype(np.float32))


"""
Set up an ADMM solver object. Note the use of the baseline solution
as an initializer. We use BM3D :cite:`dabov-2008-image` as the
denoiser, using the [code](https://pypi.org/project/bm3d) released
with :cite:`makinen-2019-exact`.
"""
A = linop.LinearOperator(input_shape=rgbshp, output_shape=s.shape, eval_fn=Afn, adj_fn=ATfn)
f = loss.SquaredL2Loss(y=sn, A=A)
C = linop.Identity(input_shape=rgbshp)
g = 1.8e-1 * 6.1e-2 * functional.BM3D(is_rgb=True)
ρ = 1.8e-1  # ADMM penalty parameter
maxiter = 12  # number of ADMM iterations

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=imgb,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-3, "maxiter": 100}),
    itstat_options={"display": True},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x = solver.solve()
hist = solver.itstat_object.history(transpose=True)


"""
Show reference and demosaiced images.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(img, title="Reference", fig=fig, ax=ax[0])
plot.imview(imgb, title="Baseline demoisac: %.2f (dB)" % metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(x, title="PPP demoisac: %.2f (dB)" % metric.psnr(img, x), fig=fig, ax=ax[2])
fig.show()


"""
Plot convergence statistics.
"""
plot.plot(
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
)


input("\nWaiting for input to close figures and exit")
