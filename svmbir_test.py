import matplotlib.pyplot as plt

import scico
import scico.numpy as snp
from scico.linop import Diagonal, Identity
from scico.linop.radon_svmbir import ParallelBeamProjector, SVMBIRWeightedSquaredL2Loss
from scico.solver import cg

# from scico.test.linop.test_linop import adjoint_AAt_test, adjoint_AtA_test


def make_im(Nx, Ny, is_3d=True):
    x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny))

    im = snp.where((x - 0.25) ** 2 / 3 + y ** 2 < 0.1, 1.0, 0.0)
    if is_3d:
        im = im[snp.newaxis, :, :]
    im = im.astype(snp.float32)

    return im


def make_A(im, num_angles, num_channels):
    angles = snp.linspace(0, snp.pi, num_angles, dtype=snp.float32)
    A = ParallelBeamProjector(im.shape, angles, num_channels)

    return A


def cg_prox(f, v, λ):  # Question: Should this be part of some other module?
    # prox:
    #   arg min  1/2 || x - v ||^2 + λ α || A x - y ||^2_W
    #      x
    #
    # solution at:
    #   (I + λ 2α A^T W A) x = v + λ 2α A^T W y
    W = f.W
    A = f.A
    scale = f.scale
    y = f.y
    hessian = f.hessian  # = (2α A^T W A)
    lhs = Identity(v.shape) + λ * hessian
    rhs = v + 2 * λ * scale * A.adj(W(y))
    x, _ = cg(lhs, rhs, x0=v)
    return x


# Nx = 128
# Ny = 129
# num_angles = 200
# num_channels = 201


Nx = 64
Ny = 65
num_angles = 100
num_channels = 101


im = make_im(Nx, Ny, is_3d=True) / Nx * 10
A = make_A(im, num_angles, num_channels)
y = A @ im


W = snp.exp(-y) * 20
# W = snp.ones_like(y)
λ = 0.01  # needs to be chosen small enough so that solution is not unstable
λ = 1

f = SVMBIRWeightedSquaredL2Loss(y=y, A=A, W=Diagonal(W))
v, _ = scico.random.normal(im.shape, dtype=im.dtype)
v *= im.max() * 0.5

xprox_svmbir = f.prox(v, λ)


xprox_cg = cg_prox(f, v, λ)


print(snp.linalg.norm(xprox_svmbir - xprox_cg) / snp.linalg.norm(xprox_svmbir))


fig, ax = plt.subplots(2, 2, figsize=[8, 8])

r = 0.2
hand = ax[0, 0].imshow(im.squeeze(), vmin=-r, vmax=r)
ax[0, 0].set_title("im")
plt.colorbar(hand, ax=ax[0, 0])
hand = ax[0, 1].imshow(v.squeeze(), vmin=-r, vmax=r)
ax[0, 1].set_title("v")
plt.colorbar(hand, ax=ax[0, 1])
hand = ax[1, 0].imshow(xprox_svmbir.squeeze(), vmin=-r, vmax=r)
ax[1, 0].set_title("xprox_svmbir")
plt.colorbar(hand, ax=ax[1, 0])
hand = ax[1, 1].imshow(xprox_cg.squeeze(), vmin=-r, vmax=r)
ax[1, 1].set_title("xprox_cg")
plt.colorbar(hand, ax=ax[1, 1])

plt.show()
