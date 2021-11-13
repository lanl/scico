import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.linop import Diagonal, Identity
from scico.solver import cg
from scico.test.linop.test_linop import adjoint_test
from scico.test.test_functional import prox_test

try:
    from scico.linop.radon_svmbir import (
        ParallelBeamProjector,
        SVMBIRWeightedSquaredL2Loss,
    )
except ImportError as e:
    pytest.skip("svmbir not installed", allow_module_level=True)


BIG_INPUT = (128, 129, 200, 201)
SMALL_INPUT = (4, 5, 7, 8)


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


def cg_prox(f, v, λ):
    # prox:
    #   arg min  1/2 || x - v ||^2 + λ α || A x - y ||^2_W
    #      x
    #
    # solution at:
    #   (I + λ 2α A^T W A) x = v + λ 2α A^T W y
    W = f.W
    A = f.A
    α = f.scale
    y = f.y
    hessian = f.hessian  # = (2α A^T W A)
    lhs = Identity(v.shape) + λ * hessian
    rhs = v + 2 * λ * α * A.adj(W(y))
    x, _ = cg(lhs, rhs, x0=v)
    return x


@pytest.mark.parametrize("Nx, Ny, num_angles, num_channels", (BIG_INPUT,))
@pytest.mark.parametrize("is_3d", (True, False))
def test_grad(Nx, Ny, num_angles, num_channels, is_3d):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(im, num_angles, num_channels)

    def f(im):
        return snp.sum(A._eval(im) ** 2)

    val_1 = jax.grad(f)(im)
    val_2 = 2 * A.adj(A(im))

    np.testing.assert_allclose(val_1, val_2)


@pytest.mark.parametrize("Nx, Ny, num_angles, num_channels", (BIG_INPUT,))
@pytest.mark.parametrize("is_3d", (True, False))
def test_adjoint(Nx, Ny, num_angles, num_channels, is_3d):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(im, num_angles, num_channels)

    adjoint_test(A)


@pytest.mark.parametrize("Nx, Ny, num_angles, num_channels", (SMALL_INPUT,))
@pytest.mark.parametrize("is_3d", (True, False))
def test_prox(Nx, Ny, num_angles, num_channels, is_3d):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(im, num_angles, num_channels)

    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)
    f = SVMBIRWeightedSquaredL2Loss(y=sino, A=A)
    prox_test(v, f, f.prox, alpha=0.25)


@pytest.mark.parametrize("Nx, Ny, num_angles, num_channels", (SMALL_INPUT,))
@pytest.mark.parametrize("is_3d", (True, False))
def test_prox_weights(Nx, Ny, num_angles, num_channels, is_3d):
    im = make_im(Nx, Ny, is_3d)
    A = make_A(im, num_angles, num_channels)

    sino = A @ im

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)

    # test with weights
    weights, _ = scico.random.uniform(sino.shape, dtype=im.dtype)
    W = scico.linop.Diagonal(weights)
    f = SVMBIRWeightedSquaredL2Loss(y=sino, A=A, W=W)
    prox_test(v, f, f.prox, alpha=0.25)


@pytest.mark.parametrize("Nx, Ny, num_angles, num_channels", (SMALL_INPUT,))
@pytest.mark.parametrize("is_3d", (True, False))
@pytest.mark.parametrize("is_weighted", (True, False))
def test_prox_cg(Nx, Ny, num_angles, num_channels, is_3d, is_weighted):
    im = make_im(Nx, Ny, is_3d=is_3d) / Nx * 10
    A = make_A(im, num_angles, num_channels)
    y = A @ im

    if is_weighted:
        W = snp.exp(-y) * 20
    else:
        W = snp.ones_like(y)

    λ = 0.01  # needs to be chosen small
    # enough so that solution is not unstable.

    f = SVMBIRWeightedSquaredL2Loss(y=y, A=A, W=Diagonal(W))
    v, _ = scico.random.normal(im.shape, dtype=im.dtype)
    v *= im.max() * 0.5

    xprox_svmbir = f.prox(v, λ)
    xprox_cg = cg_prox(f, v, λ)

    assert snp.linalg.norm(xprox_svmbir - xprox_cg) / snp.linalg.norm(xprox_svmbir) < 0.01
