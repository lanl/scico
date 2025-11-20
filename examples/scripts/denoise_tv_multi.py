#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Comparison of Optimization Algorithms for Total Variation Denoising
===================================================================

This example compares the performance of alternating direction method of
multipliers (ADMM), linearized ADMM, proximal ADMM, and primal–dual
hybrid gradient (PDHG) in solving the isotropic total variation (TV)
denoising problem

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - \mathbf{x}
  \|_2^2 + \lambda R(\mathbf{x}) \;,$$

where $R$ is the isotropic TV: the sum of the norms of the gradient
vectors at each point in the image $\mathbf{x}$.
"""


from xdesign import SiemensStar, discrete_phantom

import scico.numpy as snp
import scico.random
from scico import functional, linop, loss, plot
from scico.optimize import PDHG, LinearizedADMM, ProximalADMM
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Create a ground truth image.
"""
phantom = SiemensStar(32)
N = 256  # image size
x_gt = snp.pad(discrete_phantom(phantom, N - 16), 8)


"""
Add noise to create a noisy test image.
"""
σ = 1.0  # noise standard deviation
noise, key = scico.random.randn(x_gt.shape, seed=0)
y = x_gt + σ * noise


"""
Construct operators and functionals and set regularization parameter.
"""
# The append=0 option makes the results of horizontal and vertical
# finite differences the same shape, which is required for the L21Norm.
C = linop.FiniteDifference(input_shape=x_gt.shape, append=0)
f = loss.SquaredL2Loss(y=y)
λ = 1e0
g = λ * functional.L21Norm()


"""
The first step of the first-run solver is much slower than the
following steps, presumably due to just-in-time compilation of
relevant operators in first use. The code below performs a preliminary
solver step, the result of which is discarded, to reduce this bias in
the timing results. The precise cause of the remaining differences in
time required to compute the first step of each algorithm is unknown,
but it is worth noting that this difference becomes negligible when
just-in-time compilation is disabled (e.g. via the `JAX_DISABLE_JIT`
environment variable).
"""
solver_admm = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[1e1],
    x0=y,
    maxiter=1,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 1}),
)
solver_admm.solve();  # fmt: skip
# trailing semi-colon suppresses output in notebook


"""
Solve via ADMM with a maximum of 2 CG iterations.
"""
solver_admm = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[1e1],
    x0=y,
    maxiter=200,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 2}),
    itstat_options={"display": True, "period": 10},
)
print(f"Solving on {device_info()}\n")
print("ADMM solver")
solver_admm.solve()
hist_admm = solver_admm.itstat_object.history(transpose=True)


"""
Solve via Linearized ADMM.
"""
solver_ladmm = LinearizedADMM(
    f=f,
    g=g,
    C=C,
    mu=1e-2,
    nu=1e-1,
    x0=y,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
print("\nLinearized ADMM solver")
solver_ladmm.solve()
hist_ladmm = solver_ladmm.itstat_object.history(transpose=True)


"""
Solve via Proximal ADMM.
"""
mu, nu = ProximalADMM.estimate_parameters(C)
solver_padmm = ProximalADMM(
    f=f,
    g=g,
    A=C,
    rho=1e0,
    mu=mu,
    nu=nu,
    x0=y,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
print("\nProximal ADMM solver")
solver_padmm.solve()
hist_padmm = solver_padmm.itstat_object.history(transpose=True)


"""
Solve via PDHG.
"""
tau, sigma = PDHG.estimate_parameters(C, factor=1.5)
solver_pdhg = PDHG(
    f=f,
    g=g,
    C=C,
    tau=tau,
    sigma=sigma,
    maxiter=200,
    itstat_options={"display": True, "period": 10},
)
print("\nPDHG solver")
solver_pdhg.solve()
hist_pdhg = solver_pdhg.itstat_object.history(transpose=True)


"""
Plot results. It is worth noting that:

1. PDHG outperforms ADMM both with respect to iterations and time.
2. Proximal ADMM has similar performance to PDHG with respect to iterations,
   but is slightly inferior with respect to time.
3. ADMM greatly outperforms Linearized ADMM with respect to iterations.
4. ADMM slightly outperforms Linearized ADMM with respect to time. This is
   possible because the ADMM $\mathbf{x}$-update can be solved relatively
   cheaply, with only 2 CG iterations. If more CG iterations were required,
   the time comparison would be favorable to Linearized ADMM.
"""
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(27, 6))
plot.plot(
    snp.vstack(
        (hist_admm.Objective, hist_ladmm.Objective, hist_padmm.Objective, hist_pdhg.Objective)
    ).T,
    ptyp="semilogy",
    title="Objective function",
    xlbl="Iteration",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack(
        (hist_admm.Prml_Rsdl, hist_ladmm.Prml_Rsdl, hist_padmm.Prml_Rsdl, hist_pdhg.Prml_Rsdl)
    ).T,
    ptyp="semilogy",
    title="Primal residual",
    xlbl="Iteration",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[1],
)
plot.plot(
    snp.vstack(
        (hist_admm.Dual_Rsdl, hist_ladmm.Dual_Rsdl, hist_padmm.Dual_Rsdl, hist_pdhg.Dual_Rsdl)
    ).T,
    ptyp="semilogy",
    title="Dual residual",
    xlbl="Iteration",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[2],
)
fig.show()

fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(27, 6))
plot.plot(
    snp.vstack(
        (hist_admm.Objective, hist_ladmm.Objective, hist_padmm.Objective, hist_pdhg.Objective)
    ).T,
    snp.vstack((hist_admm.Time, hist_ladmm.Time, hist_padmm.Time, hist_pdhg.Time)).T,
    ptyp="semilogy",
    title="Objective function",
    xlbl="Time (s)",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack(
        (hist_admm.Prml_Rsdl, hist_ladmm.Prml_Rsdl, hist_padmm.Prml_Rsdl, hist_pdhg.Prml_Rsdl)
    ).T,
    snp.vstack((hist_admm.Time, hist_ladmm.Time, hist_padmm.Time, hist_pdhg.Time)).T,
    ptyp="semilogy",
    title="Primal residual",
    xlbl="Time (s)",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[1],
)
plot.plot(
    snp.vstack(
        (hist_admm.Dual_Rsdl, hist_ladmm.Dual_Rsdl, hist_padmm.Dual_Rsdl, hist_pdhg.Dual_Rsdl)
    ).T,
    snp.vstack((hist_admm.Time, hist_ladmm.Time, hist_padmm.Time, hist_pdhg.Time)).T,
    ptyp="semilogy",
    title="Dual residual",
    xlbl="Time (s)",
    lgnd=("ADMM", "LinADMM", "ProxADMM", "PDHG"),
    fig=fig,
    ax=ax[2],
)
fig.show()


input("\nWaiting for input to close figures and exit")
