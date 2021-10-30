import numpy as np

import jax

import pytest

from scico.linop import DFT
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


class TestDFT:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("pad_output", [True, False])
    @pytest.mark.parametrize("jit", [False, True])
    def test_eval(self, input_shape, pad_output, jit):
        if pad_output:
            output_shape = (48, 64)[: len(input_shape)]
        else:
            output_shape = None

        x, key = randn(input_shape, dtype=np.complex64, key=self.key)
        F = DFT(input_shape=input_shape, output_shape=output_shape, jit=jit)
        Fx = F @ x

        # In the future we can compare against snp.fft.fftn,
        # but at present it does not support the "s" argument
        if output_shape is None:
            s = None
        else:
            s = np.array(output_shape)

        np_result = np.fft.fftn(x.copy(), s=s).astype(np.complex64)

        np.testing.assert_allclose(Fx, np_result, rtol=1e-4)

    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("pad_output", [True, False])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, input_shape, pad_output, jit):
        if pad_output:
            output_shape = (48, 64)[: len(input_shape)]
        else:
            output_shape = None

        F = DFT(input_shape=input_shape, output_shape=output_shape, jit=jit)
        adjoint_test(F, self.key)

    @pytest.mark.parametrize("input_shape", [(32,), (32, 48)])
    @pytest.mark.parametrize("pad_output", [True, False])
    @pytest.mark.parametrize("truncate", [True, False])
    def test_inv(self, input_shape, pad_output, truncate):
        if pad_output:
            output_shape = (48, 64)[: len(input_shape)]
        else:
            output_shape = None

        F = DFT(input_shape=input_shape, output_shape=output_shape)

        y, key = randn(F.output_shape, dtype=np.complex64, key=self.key)

        Fi_y = F.inv(y, truncate=truncate)
        if truncate:
            assert Fi_y.shape == input_shape
            np_result = np.fft.ifftn(y.copy())

            for i, s in enumerate(input_shape):
                np_result = np.take(np_result, indices=np.r_[:s], axis=i)
            np.testing.assert_allclose(Fi_y.copy(), np_result, rtol=1e-4)
        else:
            np_result = np.fft.ifftn(y.copy())
            np.testing.assert_allclose(Fi_y.copy(), np_result, rtol=1e-4)

    def test_length_mismatch(self):
        input_shape = (32, 48)
        output_shape = (32,)
        with pytest.raises(ValueError):
            F = DFT(input_shape=input_shape, output_shape=output_shape)
