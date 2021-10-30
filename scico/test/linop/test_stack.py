import numpy as np

import jax

import pytest

from scico.linop import Convolve, Identity, LinearOperatorStack
from scico.test.linop.test_linop import adjoint_test


class TestLinearOperatorStack:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [False, True])
    def test_construct(self, jit):
        # requires a list of LinearOperators
        I = Identity((42,))
        with pytest.raises(ValueError):
            H = LinearOperatorStack(I, jit=jit)

        # checks input sizes
        A = Identity((3, 2))
        B = Identity((7, 2))
        with pytest.raises(ValueError):
            H = LinearOperatorStack([A, B], jit=jit)

        # in general, returns a BlockArray
        A = Convolve(jax.device_put(np.ones((3, 3))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], jit=jit)
        x = np.ones((9, 15))
        y = H @ x
        assert y.shape == ((11, 17), (10, 16))

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # by default, collapse to DeviceArray when possible
        A = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], jit=jit)
        x = np.ones((9, 15))
        y = H @ x
        assert y.shape == (2, 10, 16)

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # let user turn off collapsing
        A = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], collapse=False, jit=jit)
        x = np.ones((9, 15))
        y = H @ x
        assert y.shape == ((10, 16), (10, 16))

    @pytest.mark.parametrize("collapse", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, collapse, jit):
        # general case
        A = Convolve(jax.device_put(np.ones((3, 3))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], collapse=collapse, jit=jit)
        adjoint_test(H, self.key)

        # collapsable case
        A = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], collapse=collapse, jit=jit)
        adjoint_test(H, self.key)

    @pytest.mark.parametrize("collapse", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_algebra(self, collapse, jit):
        # adding
        A = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        B = Convolve(jax.device_put(np.ones((2, 2))), (9, 15))
        H = LinearOperatorStack([A, B], collapse=collapse, jit=jit)

        A = Convolve(jax.device_put(np.random.rand(2, 2)), (9, 15))
        B = Convolve(jax.device_put(np.random.rand(2, 2)), (9, 15))
        G = LinearOperatorStack([A, B], collapse=collapse, jit=jit)

        x = np.ones((9, 15))
        S = H + G

        # test correctness of adding
        assert S.output_shape == H.output_shape
        assert S.input_shape == H.input_shape
        np.testing.assert_allclose((S @ x)[0], (H @ x + G @ x)[0])
        np.testing.assert_allclose((S @ x)[1], (H @ x + G @ x)[1])

        # result of adding two conformable stacks should be a stack
        assert isinstance(S, LinearOperatorStack)
        assert isinstance(H - G, LinearOperatorStack)

        # scalar multiplication
        assert isinstance(1.0 * H, LinearOperatorStack)

        # op scaling
        scalars = [2.0, 3.0]
        y1 = S @ x
        S2 = S.scale_ops(scalars)
        y2 = S2 @ x

        np.testing.assert_allclose(scalars[0] * y1[0], y2[0])
