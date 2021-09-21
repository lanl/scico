import numpy as np

import jax
import jax.scipy.signal as signal

import pytest

from scico.blockarray import BlockArray
from scico.linop import Convolve, ConvolveByX
from scico.operator.biconvolve import BiConvolve
from scico.random import randn


class TestBiConvolve:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_eval(self, input_dtype, mode, jit):
        x, key = randn((32, 32), dtype=input_dtype, key=self.key)
        h, key = randn((4, 4), dtype=input_dtype, key=self.key)

        x_h = BlockArray.array([x, h])

        A = BiConvolve(input_shape=x_h.shape, mode=mode, jit=jit)
        signal_out = signal.convolve(x, h, mode=mode)
        np.testing.assert_allclose(A(x_h), signal_out, rtol=1e-4)

        # Test freezing
        A_x = A.freeze(0, x)
        assert isinstance(A_x, ConvolveByX)
        np.testing.assert_allclose(A_x(h), signal_out, rtol=1e-4)

        A_h = A.freeze(1, h)
        assert isinstance(A_h, Convolve)
        np.testing.assert_allclose(A_h(x), signal_out, rtol=1e-4)

        with pytest.raises(ValueError):
            A.freeze(2, x)

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            A = BiConvolve(input_shape=(2, 2))

        with pytest.raises(ValueError):
            shape = ((2, 2), (3, 3), (4, 4))  # 3 blocks
            A = BiConvolve(input_shape=shape)

        with pytest.raises(ValueError):
            shape = ((2, 2), (3,))  # 3 blocks
            A = BiConvolve(input_shape=shape)
