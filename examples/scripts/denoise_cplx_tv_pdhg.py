#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Complex Total Variation Denoising
=================================

This example demonstrates solution of a problem of the form

  $$\argmin_{\mathbf{x}} \; f(\mathbf{x}) + g(C(\mathbf{x})) \;,$$

where $C$ is a nonlinear operator, via non-linear PDHG
:cite:`valkonen-2014-primal`. The example problem represents total
variation (TV) denoising applied to a complex image with piece-wise
smooth magnitude and non-smooth phase. The appropriate TV denoising
formulation for this problem is

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x}
  \|_2^2 + \lambda \| C(\mathbf{x}) \|_{2,1} \;,$$

where $\mathbf{y}$ is the measurement, $\|\cdot\|_{2,1}$ is the
$\ell_{2,1}$ mixed norm, and $C$ is a non-linear operator that applies a
linear difference operator to the magnitude of a complex array. The
standard TV solution, which is also computed for comparison purposes,
gives very poor results since the difference is applied independently to
real and imaginary components of the complex image.
"""


from mpl_toolkits.axes_grid1 import make_axes_locatable
from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, metric, operator, plot
from scico.examples import phase_diff
from scico.optimize import PDHG
from scico.util import device_info

"""
Create a ground truth image.
"""
N = 256  # image size
phantom = SiemensStar(16)
x_mag = snp.pad(discrete_phantom(phantom, N - 16), 8) + 1.0
x_mag /= x_mag.max()
# Create reference image with structured magnitude and random phase
x_gt = x_mag * snp.exp(-1j * scico.random.randn(x_mag.shape, seed=0)[0])


"""
Add noise to create a noisy test image.
"""
σ = 0.25  # noise standard deviation
noise, key = scico.random.randn(x_gt.shape, seed=1, dtype=snp.complex64)
y = x_gt + σ * noise


"""
Denoise with standard total variation.
"""
λ_tv = 6e-2
f = loss.SquaredL2Loss(y=y)
g = λ_tv * functional.L21Norm()
# The append=0 option makes the results of horizontal and vertical finite
# differences the same shape, which is required for the L21Norm.
C = linop.FiniteDifference(input_shape=x_gt.shape, input_dtype=snp.complex64, append=0)
solver_tv = PDHG(
    f=f,
    g=g,
    C=C,
    tau=4e-1,
    sigma=4e-1,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
print(f"Solving on {device_info()}\n")
x_tv = solver_tv.solve()
hist_tv = solver_tv.itstat_object.history(transpose=True)


"""
Denoise with non-linear total variation.
"""
λ_nltv = 2e-1
g = λ_nltv * functional.L21Norm()
# Redefine C for real input (now applied to magnitude of a complex array)
C = linop.FiniteDifference(input_shape=x_gt.shape, input_dtype=snp.float32, append=0)
# Operator computing differences of absolute values
D = C @ operator.Abs(input_shape=x_gt.shape, input_dtype=snp.complex64)
solver_nltv = PDHG(
    f=f,
    g=g,
    C=D,
    tau=4e-1,
    sigma=4e-1,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
x_nltv = solver_nltv.solve()
hist_nltv = solver_nltv.itstat_object.history(transpose=True)


"""
Plot results.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(27, 6))
plot.plot(
    snp.vstack((hist_tv.Objective, hist_nltv.Objective)).T,
    ptyp="semilogy",
    title="Objective function",
    xlbl="Iteration",
    lgnd=("PDHG", "NL-PDHG"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack((hist_tv.Prml_Rsdl, hist_nltv.Prml_Rsdl)).T,
    ptyp="semilogy",
    title="Primal residual",
    xlbl="Iteration",
    lgnd=("PDHG", "NL-PDHG"),
    fig=fig,
    ax=ax[1],
)
plot.plot(
    snp.vstack((hist_tv.Dual_Rsdl, hist_nltv.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Dual residual",
    xlbl="Iteration",
    lgnd=("PDHG", "NL-PDHG"),
    fig=fig,
    ax=ax[2],
)
fig.show()


fig, ax = plot.subplots(nrows=2, ncols=4, figsize=(20, 10))
norm = plot.matplotlib.colors.Normalize(
    vmin=min(snp.abs(x_gt).min(), snp.abs(y).min(), snp.abs(x_tv).min(), snp.abs(x_nltv).min()),
    vmax=max(snp.abs(x_gt).max(), snp.abs(y).max(), snp.abs(x_tv).max(), snp.abs(x_nltv).max()),
)
plot.imview(snp.abs(x_gt), title="Ground truth", cbar=None, fig=fig, ax=ax[0, 0], norm=norm)
plot.imview(
    snp.abs(y),
    title="Measured: PSNR %.2f (dB)" % metric.psnr(snp.abs(x_gt), snp.abs(y)),
    cbar=None,
    fig=fig,
    ax=ax[0, 1],
    norm=norm,
)
plot.imview(
    snp.abs(x_tv),
    title="TV: PSNR %.2f (dB)" % metric.psnr(snp.abs(x_gt), snp.abs(x_tv)),
    cbar=None,
    fig=fig,
    ax=ax[0, 2],
    norm=norm,
)
plot.imview(
    snp.abs(x_nltv),
    title="NL-TV: PSNR %.2f (dB)" % metric.psnr(snp.abs(x_gt), snp.abs(x_nltv)),
    cbar=None,
    fig=fig,
    ax=ax[0, 3],
    norm=norm,
)
divider = make_axes_locatable(ax[0, 3])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[0, 3].get_images()[0], cax=cax)
norm = plot.matplotlib.colors.Normalize(
    vmin=min(snp.angle(x_gt).min(), snp.angle(x_tv).min(), snp.angle(x_nltv).min()),
    vmax=max(snp.angle(x_gt).max(), snp.angle(x_tv).max(), snp.angle(x_nltv).max()),
)
plot.imview(
    snp.angle(x_gt),
    title="Ground truth",
    cbar=None,
    fig=fig,
    ax=ax[1, 0],
    norm=norm,
)
plot.imview(
    snp.angle(y),
    title="Measured: Mean phase diff. %.2f" % phase_diff(snp.angle(x_gt), snp.angle(y)).mean(),
    cbar=None,
    fig=fig,
    ax=ax[1, 1],
    norm=norm,
)
plot.imview(
    snp.angle(x_tv),
    title="TV: Mean phase diff. %.2f" % phase_diff(snp.angle(x_gt), snp.angle(x_tv)).mean(),
    cbar=None,
    fig=fig,
    ax=ax[1, 2],
    norm=norm,
)
plot.imview(
    snp.angle(x_nltv),
    title="NL-TV: Mean phase diff. %.2f" % phase_diff(snp.angle(x_gt), snp.angle(x_nltv)).mean(),
    cbar=None,
    fig=fig,
    ax=ax[1, 3],
    norm=norm,
)
divider = make_axes_locatable(ax[1, 3])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1, 3].get_images()[0], cax=cax)
ax[0, 0].set_ylabel("Magnitude")
ax[1, 0].set_ylabel("Phase")
fig.tight_layout()
fig.show()


input("\nWaiting for input to close figures and exit")
