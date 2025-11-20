#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Non-Negative Basis Pursuit DeNoising (APGM)
===========================================

This example demonstrates the solution of a non-negative sparse coding
problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - D \mathbf{x} \|_2^2
  + \lambda \| \mathbf{x} \|_1 + \iota_{\mathrm{NN}}(\mathbf{x}) \;,$$

where $D$ the dictionary, $\mathbf{y}$ the signal to be represented,
$\mathbf{x}$ is the sparse representation, and $\iota_{\mathrm{NN}}$ is
the indicator function of the non-negativity constraint.

In this example the problem is solved via Accelerated PGM, using the
proximal averaging method :cite:`yu-2013-better` to approximate the
proximal operator of the sum of the $\ell_1$ norm and an indicator
function, while ADMM is used in a
[companion example](sparsecode_nn_admm.rst).
"""

import numpy as np

import scico.numpy as snp
from scico import functional, linop, loss, plot
from scico.optimize.pgm import AcceleratedPGM
from scico.util import device_info

"""
Create random dictionary, reference random sparse representation, and
test signal consisting of the synthesis of the reference sparse
representation.
"""
m = 32  # signal size
n = 128  # dictionary size
s = 10  # sparsity level

np.random.seed(1)
D = np.random.randn(m, n).astype(np.float32)
D = D / np.linalg.norm(D, axis=0, keepdims=True)  # normalize dictionary
L0 = max(np.linalg.norm(D, 2) ** 2, 5e1)

xt = np.zeros(n, dtype=np.float32)  # true signal
idx = np.random.randint(low=0, high=n, size=s)  # support of xt
xt[idx] = np.random.rand(s)
y = D @ xt + 5e-2 * np.random.randn(m)  # synthetic signal

xt = snp.array(xt)  # convert to jax array
y = snp.array(y)  # convert to jax array


"""
Set up the forward operator and APGM solver object.
"""
lmbda = 2e-1
A = linop.MatrixOperator(D)
f = loss.SquaredL2Loss(y=y, A=A)
g = functional.ProximalAverage([lmbda * functional.L1Norm(), functional.NonNegativeIndicator()])
maxiter = 250  # number of APGM iterations
solver = AcceleratedPGM(
    f=f, g=g, L0=L0, x0=A.adj(y), maxiter=maxiter, itstat_options={"display": True, "period": 20}
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x = solver.solve()


"""
Plot the recovered coefficients and signal.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    np.vstack((xt, solver.x)).T,
    title="Coefficients",
    lgnd=("Ground Truth", "Recovered"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    np.vstack((D @ xt, y, D @ solver.x)).T,
    title="Signal",
    lgnd=("Ground Truth", "Noisy", "Recovered"),
    fig=fig,
    ax=ax[1],
)
fig.show()


input("\nWaiting for input to close figures and exit")
