import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop import Convolve, DiagonalStack, Identity, Sum, VerticalStack
from scico.test.linop.test_linop import adjoint_test


class TestVerticalStack:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [False, True])
    def test_construct(self, jit):
        # requires a list of LinearOperators
        I = Identity((42,))
        with pytest.raises(ValueError):
            H = VerticalStack(I, jit=jit)

        # checks input sizes
        A = Identity((3, 2))
        B = Identity((7, 2))
        with pytest.raises(ValueError):
            H = VerticalStack([A, B], jit=jit)

        # in general, returns a BlockArray
        A = Convolve(jax.device_put(np.ones((3, 3))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == ((9, 13), (8, 12))

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # by default, collapse to DeviceArray when possible
        A = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == (2, 8, 12)

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # let user turn off collapsing
        A = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], collapse=False, jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == ((8, 12), (8, 12))

    @pytest.mark.parametrize("collapse", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, collapse, jit):
        # general case
        A = Convolve(jax.device_put(np.ones((3, 3))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], collapse=collapse, jit=jit)
        adjoint_test(H, self.key)

        # collapsable case
        A = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], collapse=collapse, jit=jit)
        adjoint_test(H, self.key)

    @pytest.mark.parametrize("collapse", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_algebra(self, collapse, jit):
        # adding
        A = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        B = Convolve(jax.device_put(np.ones((2, 2))), (7, 11))
        H = VerticalStack([A, B], collapse=collapse, jit=jit)

        A = Convolve(jax.device_put(np.random.rand(2, 2)), (7, 11))
        B = Convolve(jax.device_put(np.random.rand(2, 2)), (7, 11))
        G = VerticalStack([A, B], collapse=collapse, jit=jit)

        x = np.ones((7, 11))
        S = H + G

        # test correctness of adding
        assert S.output_shape == H.output_shape
        assert S.input_shape == H.input_shape
        np.testing.assert_allclose((S @ x)[0], (H @ x + G @ x)[0])
        np.testing.assert_allclose((S @ x)[1], (H @ x + G @ x)[1])

        # result of adding two conformable stacks should be a stack
        assert isinstance(S, VerticalStack)
        assert isinstance(H - G, VerticalStack)

        # scalar multiplication
        assert isinstance(1.0 * H, VerticalStack)

        # op scaling
        scalars = [2.0, 3.0]
        y1 = S @ x
        S2 = S.scale_ops(scalars)
        y2 = S2 @ x

        np.testing.assert_allclose(scalars[0] * y1[0], y2[0])


class TestBlockDiagonalLinearOperator:
    def test_apply(self):
        S1 = (3, 4)
        S2 = (3, 5)
        S3 = (2, 2)
        A1 = Identity(S1)
        A2 = 2 * Identity(S2)
        A3 = Sum(S3)
        H = DiagonalStack((A1, A2, A3))

        x = snp.ones((S1, S2, S3))
        y = H @ x
        y_expected = snp.blockarray((snp.ones(S1), 2 * snp.ones(S2), snp.sum(snp.ones(S3))))

        assert y == y_expected

    def test_adjoint(self):
        S1 = (3, 4)
        S2 = (3, 5)
        S3 = (2, 2)
        A1 = Identity(S1)
        A2 = 2 * Identity(S2)
        A3 = Sum(S3)
        H = DiagonalStack((A1, A2, A3))

        y = snp.ones((S1, S2, ()), dtype=snp.float32)
        x = H.T @ y
        x_expected = snp.blockarray(
            (
                snp.ones(S1),
                snp.ones(S2),
                snp.ones(S3),
            )
        )

        assert x == x_expected

    def test_input_collapse(self):
        S = (3, 4)
        A1 = Identity(S)
        A2 = Sum(S)

        H = DiagonalStack((A1, A2))
        assert H.input_shape == (2, *S)

        H = DiagonalStack((A1, A2), allow_input_collapse=False)
        assert H.input_shape == (S, S)

    def test_output_collapse(self):
        S1 = (3, 4)
        S2 = (5, 3, 4)
        A1 = Identity(S1)
        A2 = Sum(S2, axis=0)

        H = DiagonalStack((A1, A2))
        assert H.output_shape == (2, *S1)

        H = DiagonalStack((A1, A2), allow_output_collapse=False)
        assert H.output_shape == (S1, S1)
