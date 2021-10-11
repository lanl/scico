import numpy as np

import pytest

import scico.numpy as snp

try:
    from scico.linop.radon_svmbir import (
        ParallelBeamProjector,
        SVMBIRWeightedSquaredL2Loss,
    )
except ImportError as e:
    pytest.skip("svmbir not installed", allow_module_level=True)


import matplotlib.pyplot as plt
import svmbir
from xdesign import Foam, discrete_phantom

from scico.admm import ADMM, LinearSubproblemSolver
from scico.functional import BM3D, NonNegativeIndicator
from scico.linop import Diagonal, Identity


def poisson_sino(sino, max_intensity=2000):
    # y = -log(c/b)
    # c = b exp(-y)
    counts = np.random.poisson(max_intensity * np.exp(-sino)).astype(np.float32)
    counts[counts == 0] = 1  # deal with 0s
    return -np.log(counts / max_intensity)


def gen_phantom(N, density):
    phantom = discrete_phantom(Foam(size_range=[0.06, 0.03], gap=0.02, porosity=1), size=N - 10)
    phantom = phantom / np.max(phantom) * density
    phantom = np.pad(phantom, 5)
    phantom[phantom < 0] = 0
    return phantom


N = 200
num_angles = int(N / 2)
num_channels = N


density = 0.04

x_gt = gen_phantom(N, density)
plt.imshow(x_gt)
plt.show()


angles = snp.linspace(0, snp.pi, num_angles, dtype=snp.float32)

A = ParallelBeamProjector(x_gt.shape, angles, num_channels)

sino = A @ x_gt

y = poisson_sino(sino, max_intensity=2000)


plt.imshow(sino)
plt.show()

plt.imshow(y)
plt.show()


weights = svmbir.calc_weights(y, weight_type="transmission")

xs = svmbir.recon(
    np.array(y[:, np.newaxis]),
    np.array(angles),
    weights=weights[:, np.newaxis],
    num_rows=N,
    num_cols=N,
    positivity=True,
    verbose=0,
)
xs = xs[0]

plt.imshow(xs)
plt.show()


weight_op = Diagonal(weights ** 0.5)

# scale = 0.5 # produces correct scale
scale = 1  # produces incorrect scale
f = SVMBIRWeightedSquaredL2Loss(y=y, A=A, weight_op=weight_op, scale=scale)


ρ = 100  # denoiser weight (inverse data weight)
σ = density * 0.2  # denoiser sigma

g0 = σ * ρ * BM3D()  # this way denoising is constant w.r.t. ρ
g1 = NonNegativeIndicator()

maxiter = 10
solver = ADMM(
    f=f,
    g_list=[g0, g1],
    C_list=[Identity(xs.shape), Identity(xs.shape)],
    rho_list=[ρ, 1],
    x0=xs,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 100}),
    verbose=True,
)

xa1 = solver.solve()


fig, ax = plt.subplots(1, 3, figsize=[14, 5])
ax = np.atleast_1d(ax)
vmin = -density * 0.1
vmax = density * 1.2

im = ax[0].imshow(x_gt, vmin=vmin, vmax=vmax)
ax[0].set_title("Ground Truth")
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(xs, vmin=vmin, vmax=vmax)
ax[1].set_title("MRF")
plt.colorbar(im, ax=ax[1])

im = ax[2].imshow(xa1, vmin=vmin, vmax=vmax)
ax[2].set_title("BM3D")
plt.colorbar(im, ax=ax[2])


fig.suptitle(f"scale = {scale}")

plt.show()


# weight_op = Diagonal(weights**0.5)

# f = SVMBIRWeightedSquaredL2Loss(y=y, A=A, weight_op=weight_op, scale=0.5)
# x0 = A.T @ y


# ρ = 50  # ADMM penalty parameter


# σ = density*0.1
# # g = ZeroFunctional()
# g0 = σ*ρ * BM3D()
# g1 = NonNegativeIndicator()
# C = Identity(x0.shape)

# maxiter = 15
# solver = ADMM(
#     f=f,
#     g_list=[g0, g1],
#     C_list=[C, C],
#     rho_list=[ρ, 1],
#     x0=x0,
#     maxiter=maxiter,
#     subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 100}),
#     verbose=True,
# )

# xa1 = solver.solve()

# plt.imshow(xa1)
# plt.show()
