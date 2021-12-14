#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Basis Pursuit DeNoising (Accelerated PGM)
=========================================

This example demonstrates the use of class
[pgm.AcceleratedPGM](../_autosummary/scico.optimize.rst#scico.optimize.AcceleratedPGM)
to solve the sparse coding problem problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \| \mathbf{y} - D \mathbf{x} \|_2^2
  + \lambda \| \mathbf{x} \|_1\;,$$

where $D$ the dictionary, $\mathbf{y}$ the signal to be represented,
and $\mathbf{x}$ is the sparse representation.
"""

import numpy as np

import jax

from scico import functional, linop, loss, plot
from scico.optimize.pgm import AcceleratedPGM
from scico.util import device_info

"""
Construct a random dictionary, a reference random sparse
representation, and a test signal consisting of the synthesis of the
reference sparse representation.
"""
m = 512  # Signal size
n = 4 * m  # Dictionary size
s = 32  # Sparsity level (number of non-zeros)
σ = 0.5  # Noise level

np.random.seed(12345)
D = np.random.randn(m, n)
L0 = np.linalg.norm(D, 2) ** 2

x_gt = np.zeros(n)  # true signal
idx = np.random.permutation(list(range(0, n - 1)))
x_gt[idx[0:s]] = np.random.randn(s)
y = D @ x_gt + σ * np.random.randn(m)  # synthetic signal

x_gt = jax.device_put(x_gt)  # convert to jax array, push to GPU
y = jax.device_put(y)  # convert to jax array, push to GPU


"""
Set up the forward operator and AcceleratedPGM solver object.
"""
maxiter = 100
λ = 2.98e1
A = linop.MatrixOperator(D)
f = loss.SquaredL2Loss(y=y, A=A)
g = λ * functional.L1Norm()
solver = AcceleratedPGM(
    f=f, g=g, L0=L0, x0=A.adj(y), maxiter=maxiter, itstat_options={"display": True, "period": 10}
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x = solver.solve()
hist = solver.itstat_object.history(transpose=True)


"""
Plot the recovered coefficients and convergence statistics.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    np.vstack((x_gt, x)).T,
    title="Coefficients",
    lgnd=("Ground Truth", "Recovered"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    np.vstack((hist.Objective, hist.Residual)).T,
    ptyp="semilogy",
    title="Convergence",
    xlbl="Iteration",
    lgnd=("Objective", "Residual"),
    fig=fig,
    ax=ax[1],
)
fig.show()


input("\nWaiting for input to close figures and exit")
