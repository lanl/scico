import numpy as np

import pytest
from prox import prox_test

from scico import functional, linop
from scico.random import randn

# Enable 64-bit mode for output dtype checks, scoped to this module by a
# fixture in scico/test/conftest.py.
pytestmark = pytest.mark.x64


class TestComposed:
    def setup_method(self):
        key = None
        self.shape = (2, 3, 4)
        self.dtype = np.float32
        self.x, key = randn(self.shape, key=key, dtype=self.dtype)
        self.composed = functional.ComposedFunctional(
            functional.L2Norm(),
            linop.Reshape(self.x.shape, (2, -1), input_dtype=self.dtype),
        )

    def test_eval(self):
        np.testing.assert_allclose(self.composed(self.x), self.composed.functional(self.x))

    def test_prox(self):
        prox_test(self.x, self.composed.__call__, self.composed.prox, 1.0)
