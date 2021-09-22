import numpy as np

import jax

import pytest

import scico
import scico.numpy as snp
from scico.test.linop.test_linop import adjoint_AAt_test, adjoint_AtA_test
from scico.test.test_functional import prox_test

try:
    from scico.linop.radon_svmbir import (
        ParallelBeamProjector,
        SvmbirWeightedSquaredL2Loss,
    )
except ImportError as e:
    pytest.skip("svmbir not installed", allow_module_level=True)

# make everything different sizes to catch dimension swap bugs
Nx, Ny = 128, 129
num_angles = 200
num_channels = 201

x, y = snp.meshgrid(snp.linspace(-1, 1, Nx), snp.linspace(-1, 1, Ny))

im = snp.where((x - 0.25) ** 2 / 3 + y ** 2 < 0.1, 1.0, 0.0)
im = im[snp.newaxis, :, :]

angles = snp.linspace(0, snp.pi, num_angles)

A = ParallelBeamProjector(im.shape, angles, num_channels)


def test_grad():
    def f(im):
        return snp.sum(A._eval(im) ** 2)

    val_1 = jax.grad(f)(im)
    val_2 = 2 * A.adj(A(im))

    np.testing.assert_allclose(val_1, val_2)


def test_adjoint():
    adjoint_AtA_test(A)
    adjoint_AAt_test(A)


def test_prox():
    sino = A @ im

    f = SvmbirWeightedSquaredL2Loss(y=sino, A=A)

    v, _ = scico.random.normal(im.shape, dtype=im.dtype)

    prox_test(v, f, f.prox, alpha=0.25)
