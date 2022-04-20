import numpy as np

from jax.config import config

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import warnings

import pytest

from scico import functional
from scico.numpy import blockarray
from scico.numpy.testing import assert_allclose
from scico.random import randn


class SeparableTestObject:
    def __init__(self, dtype):
        self.f = functional.L1Norm()
        self.g = functional.SquaredL2Norm()
        self.fg = functional.SeparableFunctional([self.f, self.g])

        n = 4
        m = 6
        key = None

        self.v1, key = randn((n,), key=key, dtype=dtype)  # point for prox eval
        self.v2, key = randn((m,), key=key, dtype=dtype)  # point for prox eval
        self.vb = blockarray([self.v1, self.v2])


@pytest.fixture(params=[np.float32, np.complex64, np.float64, np.complex128])
def test_separable_obj(request):
    return SeparableTestObject(request.param)


def test_separable_eval(test_separable_obj):
    fv1 = test_separable_obj.f(test_separable_obj.v1)
    gv2 = test_separable_obj.g(test_separable_obj.v2)
    fgv = test_separable_obj.fg(test_separable_obj.vb)
    assert_allclose(fv1 + gv2, fgv, rtol=5e-2)


def test_separable_prox(test_separable_obj):
    alpha = 0.1
    fv1 = test_separable_obj.f.prox(test_separable_obj.v1, alpha)
    gv2 = test_separable_obj.g.prox(test_separable_obj.v2, alpha)
    fgv = test_separable_obj.fg.prox(test_separable_obj.vb, alpha)
    out = blockarray((fv1, gv2)).ravel()
    assert_allclose(out, fgv.ravel(), rtol=5e-2)


def test_separable_grad(test_separable_obj):
    # Used to restore the warnings after the context is used
    with warnings.catch_warnings():
        # Ignores warning raised by ensure_on_device
        warnings.filterwarnings(action="ignore", category=UserWarning)

        # Verifies that there is a warning on f.grad and fg.grad
        np.testing.assert_warns(test_separable_obj.f.grad(test_separable_obj.v1))
        np.testing.assert_warns(test_separable_obj.fg.grad(test_separable_obj.vb))

        # Tests the separable grad with warnings being supressed
        fv1 = test_separable_obj.f.grad(test_separable_obj.v1)
        gv2 = test_separable_obj.g.grad(test_separable_obj.v2)
        fgv = test_separable_obj.fg.grad(test_separable_obj.vb)
        out = blockarray((fv1, gv2)).ravel()
        assert_allclose(out, fgv.ravel(), rtol=5e-2)
