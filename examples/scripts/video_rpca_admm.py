#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Video Decomposition via Robust PCA
==================================

This example demonstrates video foreground/background separation via a
variant of the Robust PCA problem

  $$\mathrm{argmin}_{\mathbf{x}_0, \mathbf{x}_1} \; \| \mathbf{x}_0 +
      \mathbf{x}_1 - \mathbf{y} \|_2^2 + \lambda_0 \| \mathbf{x}_0 \|_*
      + \lambda_1 \| \mathbf{x}_1 \|_1 \;,$$

where $\mathbf{x}_0$ and $\mathbf{x}_1$ are respectively low-rank and
sparse components, $\| \cdot \|_*$ denotes the nuclear norm, and
$\| \cdot \|_1$ denotes the $\ell_1$ norm.

Note: while video foreground/background separation is not an example of
the scientific and computational imaging problems that are the focus of
SCICO, it provides a convenient demonstration of Robust PCA, which does
have potential application in scientific imaging problems.
"""

import imageio

import scico.numpy as snp
from scico import functional, linop, loss, plot
from scico.examples import rgb2gray
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Load example video.
"""
reader = imageio.get_reader("imageio:newtonscradle.gif")
nfrm = reader.get_length()
frmlst = []
for i, frm in enumerate(reader):
    frmlst.append(rgb2gray(frm[..., 0:3].astype(snp.float32) / 255.0))
vid = snp.stack(frmlst, axis=2)


"""
Construct matrix with each column consisting of a vectorised video frame.
"""
y = vid.reshape((-1, vid.shape[-1]))


"""
Define functional for Robust PCA problem.
"""
A = linop.Sum(sum_axis=0, input_shape=(2,) + y.shape)
f = loss.SquaredL2Loss(y=y, A=A)
C0 = linop.Slice(idx=0, input_shape=(2,) + y.shape)
g0 = functional.NuclearNorm()
C1 = linop.Slice(idx=1, input_shape=(2,) + y.shape)
g1 = functional.L1Norm()


"""
Set up an ADMM solver object.
"""
λ0 = 1e1  # nuclear norm regularization parameter
λ1 = 3e1  # l1 norm regularization parameter
ρ0 = 2e1  # ADMM penalty parameter
ρ1 = 2e1  # ADMM penalty parameter
maxiter = 50  # number of ADMM iterations

solver = ADMM(
    f=f,
    g_list=[λ0 * g0, λ1 * g1],
    C_list=[C0, C1],
    rho_list=[ρ0, ρ1],
    x0=A.adj(y),
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(),
    itstat_options={"display": True, "period": 10},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x = solver.solve()
hist = solver.itstat_object.history(transpose=True)


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
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[1],
)
fig.show()


"""
Reshape low-rank component as background video sequence and sparse component
as foreground video sequence.
"""
xlr = C0(x)
xsp = C1(x)
vbg = xlr.reshape(vid.shape)
vfg = xsp.reshape(vid.shape)


"""
Display original video frames and corresponding background and foreground frames.
"""
fig, ax = plot.subplots(nrows=4, ncols=3, figsize=(10, 10))
ax[0][0].set_title("Original")
ax[0][1].set_title("Background")
ax[0][2].set_title("Foreground")
for n, fn in enumerate(range(1, 9, 2)):
    plot.imview(vid[..., fn], fig=fig, ax=ax[n][0])
    plot.imview(vbg[..., fn], fig=fig, ax=ax[n][1])
    plot.imview(vfg[..., fn], fig=fig, ax=ax[n][2])
    ax[n][0].set_ylabel("Frame %d" % fn, labelpad=5, rotation=90, size="large")
fig.tight_layout()
fig.show()


input("\nWaiting for input to close figures and exit")
