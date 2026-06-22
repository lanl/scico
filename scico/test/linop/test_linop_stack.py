import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop import (
    Convolve,
    DiagonalReplicated,
    DiagonalStack,
    Identity,
    Sum,
    VerticalStack,
)
from scico.operator import Abs
from scico.random import randn
from scico.test.linop.test_linop import adjoint_test


class TestVerticalStack:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    @pytest.mark.parametrize("jit", [False, True])
    def test_construct(self, jit):
        # requires a list of LinearOperators
        Id = Identity((42,))
        with pytest.raises(TypeError):
            H = VerticalStack(Id, jit=jit)

        # requires all list elements to be LinearOperators
        A = Abs((42,))
        with pytest.raises(TypeError):
            H = VerticalStack((A, Id), jit=jit)

        # checks input sizes
        A = Identity((3, 2))
        B = Identity((7, 2))
        with pytest.raises(ValueError):
            H = VerticalStack([A, B], jit=jit)

        # in general, returns a BlockArray
        A = Convolve(snp.ones((3, 3)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == ((9, 13), (8, 12))

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # by default, collapse_output to jax array when possible
        A = Convolve(snp.ones((2, 2)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == (2, 8, 12)

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A @ x)
        assert np.allclose(y[1], B @ x)

        # let user turn off collapsing
        A = Convolve(snp.ones((2, 2)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], collapse_output=False, jit=jit)
        x = np.ones((7, 11))
        y = H @ x
        assert y.shape == ((8, 12), (8, 12))

    @pytest.mark.parametrize("collapse_output", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_adjoint(self, collapse_output, jit):
        # general case
        A = Convolve(snp.ones((3, 3)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)
        adjoint_test(H, self.key)

        # collapsable case
        A = Convolve(snp.ones((2, 2)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)
        adjoint_test(H, self.key)

    @pytest.mark.parametrize("collapse_output", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_algebra(self, collapse_output, jit):
        # adding
        A = Convolve(snp.ones((2, 2)), (7, 11))
        B = Convolve(snp.ones((2, 2)), (7, 11))
        H = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)

        A = Convolve(snp.array(np.random.rand(2, 2)), (7, 11))
        B = Convolve(snp.array(np.random.rand(2, 2)), (7, 11))
        G = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)

        x = np.ones((7, 11))
        S = H + G

        # test correctness of addition
        assert S.output_shape == H.output_shape
        assert S.input_shape == H.input_shape
        np.testing.assert_allclose((S @ x)[0], (H @ x + G @ x)[0])
        np.testing.assert_allclose((S @ x)[1], (H @ x + G @ x)[1])


class TestBlockDiagonalLinearOperator:
    def test_construct(self):
        Id = Identity((42,))
        A = Abs((42,))
        with pytest.raises(TypeError):
            H = DiagonalStack((A, Id))

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

        np.testing.assert_equal(y, y_expected)

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

        H = DiagonalStack((A1, A2), collapse_input=False)
        assert H.input_shape == (S, S)

    def test_output_collapse(self):
        S1 = (3, 4)
        S2 = (5, 3, 4)
        A1 = Identity(S1)
        A2 = Sum(S2, axis=0)

        H = DiagonalStack((A1, A2))
        assert H.output_shape == (2, *S1)

        H = DiagonalStack((A1, A2), collapse_output=False)
        assert H.output_shape == (S1, S1)


class TestDiagonalReplicated:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    def test_adjoint(self):
        x, key = randn((2, 3, 4), key=self.key)
        A = Sum(x.shape[1:], axis=-1)
        D = DiagonalReplicated(A, x.shape[0])
        y = D.T(D(x))
        np.testing.assert_allclose(y[0], A.T(A(x[0])))
        np.testing.assert_allclose(y[1], A.T(A(x[1])))
