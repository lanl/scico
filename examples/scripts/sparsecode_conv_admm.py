#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Convolutional Sparse Coding (ADMM)
==================================

This example demonstrates the solution of a simple convolutional sparse
coding problem

  $$\mathrm{argmin}_{\mathbf{x}} \; \frac{1}{2} \Big\| \mathbf{y} -
  \sum_k \mathbf{h}_k \ast \mathbf{x}_k \Big\|_2^2 + \lambda \sum_k
  ( \| \mathbf{x}_k \|_1 - \| \mathbf{x}_k \|_2 ) \;,$$

where the $\mathbf{h}$_k is a set of filters comprising the dictionary,
the $\mathbf{x}$_k is a corrresponding set of coefficient maps, and
$\mathbf{y}$ is the signal to be represented. The problem is solved via
an ADMM algorithm using the frequency-domain approach proposed in
:cite:`wohlberg-2014-efficient`.
"""

import numpy as np

import scico.numpy as snp
from scico import plot
from scico.examples import create_conv_sparse_phantom
from scico.functional import L1MinusL2Norm
from scico.linop import CircularConvolve, Identity, Sum
from scico.loss import SquaredL2Loss
from scico.optimize.admm import ADMM, FBlockCircularConvolveSolver
from scico.util import device_info

"""
Set problem size and create random convolutional dictionary (a set of
filters) and a corresponding sparse random set of coefficient maps.
"""
N = 128  # image size
Nnz = 128  # number of non-zeros in coefficient maps
h, x0 = create_conv_sparse_phantom(N, Nnz)


"""
Normalize dictionary filters and scale coefficient maps accordingly.
"""
hnorm = np.sqrt(np.sum(h**2, axis=(1, 2), keepdims=True))
h /= hnorm
x0 *= hnorm


"""
Convert numpy arrays to jax arrays.
"""
h = snp.array(h)
x0 = snp.array(x0)


"""
Set up sum-of-convolutions forward operator.
"""
C = CircularConvolve(h, input_shape=x0.shape, ndims=2)
S = Sum(input_shape=C.output_shape, axis=0)
A = S @ C


"""
Construct test image from dictionary $\mathbf{h}$ and coefficient maps
$\mathbf{x}_0$.
"""
y = A(x0)


"""
Set functional and solver parameters.
"""
λ = 1e0  # l1-l2 norm regularization parameter
ρ = 2e0  # ADMM penalty parameter
maxiter = 200  # number of ADMM iterations


"""
Define loss function and regularization. Note the use of the
$\ell_1 - \ell_2$ norm, which has been found to provide slightly better
performance than the $\ell_1$ norm in this type of problem
:cite:`wohlberg-2021-psf`.
"""
f = SquaredL2Loss(y=y, A=A)
g0 = λ * L1MinusL2Norm()
C0 = Identity(input_shape=x0.shape)


"""
Initialize ADMM solver.
"""
solver = ADMM(
    f=f,
    g_list=[g0],
    C_list=[C0],
    rho_list=[ρ],
    alpha=1.8,
    maxiter=maxiter,
    subproblem_solver=FBlockCircularConvolveSolver(check_solve=True),
    itstat_options={"display": True, "period": 10},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x1 = solver.solve()
hist = solver.itstat_object.history(transpose=True)


"""
Show the recovered coefficient maps.
"""
fig, ax = plot.subplots(nrows=2, ncols=3, figsize=(12, 8.6))
plot.imview(x0[0], title="Coef. map 0", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 0])
ax[0, 0].set_ylabel("Ground truth")
plot.imview(x0[1], title="Coef. map 1", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 1])
plot.imview(x0[2], title="Coef. map 2", cmap=plot.cm.Blues, fig=fig, ax=ax[0, 2])
plot.imview(x1[0], cmap=plot.cm.Blues, fig=fig, ax=ax[1, 0])
ax[1, 0].set_ylabel("Recovered")
plot.imview(x1[1], cmap=plot.cm.Blues, fig=fig, ax=ax[1, 1])
plot.imview(x1[2], cmap=plot.cm.Blues, fig=fig, ax=ax[1, 2])
fig.tight_layout()
fig.show()


"""
Show test image and reconstruction from recovered coefficient maps.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 6))
plot.imview(y, title="Test image", cmap=plot.cm.gist_heat_r, fig=fig, ax=ax[0])
plot.imview(A(x1), title="Reconstructed image", cmap=plot.cm.gist_heat_r, fig=fig, ax=ax[1])
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
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[1],
)
fig.show()


input("\nWaiting for input to close figures and exit")
