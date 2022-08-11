import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop import DFT
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


class TestDFT:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_shape", [(16,), (16, 4), (16, 4, 7)])
    @pytest.mark.parametrize(
        "axes_and_shape",
        [
            (None, None),
            ((0,), None),
            ((0,), (20,)),
            ((0, 2), None),
            ((0, 2), (20, 8)),
            (None, (6, 8)),
        ],
    )
    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    @pytest.mark.parametrize("jit", [False, True])
    def test_dft(self, input_shape, axes_and_shape, norm, jit):
        axes = axes_and_shape[0]
        axes_shape = axes_and_shape[1]

        # Skip bad parameter permutations
        if axes is not None and len(axes) >= len(input_shape):
            return
        if axes is not None and max(axes) >= len(input_shape):
            return
        if axes_shape is not None and len(axes_shape) > len(input_shape):
            return

        x, self.key = randn(input_shape, dtype=np.complex64, key=self.key)
        F = DFT(input_shape=input_shape, axes=axes, axes_shape=axes_shape, norm=norm, jit=jit)
        Fx = F @ x

        # Test eval
        snp_result = snp.fft.fftn(x, s=axes_shape, axes=axes, norm=norm).astype(np.complex64)
        np.testing.assert_allclose(Fx, snp_result, rtol=1e-6)

        # Test adjoint
        adjoint_test(F, self.key)

        # Test inverse
        y, self.key = randn(F.output_shape, dtype=np.complex64, key=self.key)
        Fiy = F.inv(y)
        snp_result = snp.fft.ifftn(y, s=F.inv_axes_shape, axes=axes, norm=norm).astype(np.complex64)
        np.testing.assert_allclose(Fiy, snp_result, rtol=1e-6)

    def test_axes_check(self):
        input_shape = (32, 48)
        axes = (0,)
        axes_shape = (40, 50)
        with pytest.raises(ValueError):
            F = DFT(input_shape=input_shape, axes=axes, axes_shape=axes_shape)
