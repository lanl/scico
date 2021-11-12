import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
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
