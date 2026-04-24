import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.operator import (
    Abs,
    DiagonalReplicated,
    DiagonalStack,
    Operator,
    VerticalStack,
)
from scico.random import randn

TestOpA = Operator(input_shape=(3, 4), output_shape=(2, 3, 4), eval_fn=lambda x: snp.stack((x, x)))
TestOpB = Operator(
    input_shape=(3, 4), output_shape=(6, 4), eval_fn=lambda x: snp.concatenate((x, x))
)
TestOpC = Operator(
    input_shape=(3, 4), output_shape=(6, 4), eval_fn=lambda x: snp.concatenate((x, 2 * x))
)


class TestVerticalStack:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    @pytest.mark.parametrize("jit", [False, True])
    def test_construct(self, jit):
        # requires a list of Operators
        A = Abs((42,))
        with pytest.raises(TypeError):
            H = VerticalStack(A, jit=jit)

        # checks input sizes
        A = Abs((3, 2))
        B = Abs((7, 2))
        with pytest.raises(ValueError):
            H = VerticalStack([A, B], jit=jit)

        # in general, returns a BlockArray
        A = TestOpA
        B = TestOpB
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((3, 4))
        y = H(x)
        assert y.shape == ((2, 3, 4), (6, 4))

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A(x))
        assert np.allclose(y[1], B(x))

        # by default, collapse_output to jax array when possible
        A = TestOpB
        B = TestOpB
        H = VerticalStack([A, B], jit=jit)
        x = np.ones((3, 4))
        y = H(x)
        assert y.shape == (2, 6, 4)

        # ... result should be [A@x, B@x]
        assert np.allclose(y[0], A(x))
        assert np.allclose(y[1], B(x))

        # let user turn off collapsing
        A = TestOpA
        B = TestOpA
        H = VerticalStack([A, B], collapse_output=False, jit=jit)
        x = np.ones((3, 4))
        y = H(x)
        assert y.shape == ((2, 3, 4), (2, 3, 4))

    @pytest.mark.parametrize("collapse_output", [False, True])
    @pytest.mark.parametrize("jit", [False, True])
    def test_algebra(self, collapse_output, jit):
        # adding
        A = TestOpB
        B = TestOpB
        H = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)

        A = TestOpC
        B = TestOpC
        G = VerticalStack([A, B], collapse_output=collapse_output, jit=jit)

        x = np.ones((3, 4))
        S = H + G

        # test correctness of addition
        assert S.output_shape == H.output_shape
        assert S.input_shape == H.input_shape
        np.testing.assert_allclose((S(x))[0], (H(x) + G(x))[0])
        np.testing.assert_allclose((S(x))[1], (H(x) + G(x))[1])


class TestBlockDiagonalOperator:
    def test_construct(self):
        # requires a list of Operators
        A = Abs((8,))
        with pytest.raises(TypeError):
            H = VerticalStack(A)

        # no nested output shapes
        A = Abs(((8,), (10,)))
        with pytest.raises(ValueError):
            H = VerticalStack((A, A))

        # output dtypes must be the same
        A = Abs(input_shape=(8,), input_dtype=snp.float32)
        B = Abs(input_shape=(8,), input_dtype=snp.int32)
        with pytest.raises(ValueError):
            H = VerticalStack((A, B))

    def test_apply(self):
        S1 = (3, 4)
        S2 = (3, 5)
        S3 = (2, 2)
        A1 = Abs(S1)
        A2 = 2 * Abs(S2)
        A3 = Abs(S3)
        H = DiagonalStack((A1, A2, A3))

        x = snp.ones((S1, S2, S3))
        y = H(x)
        y_expected = snp.blockarray((snp.ones(S1), 2 * snp.ones(S2), snp.sum(snp.ones(S3))))

        np.testing.assert_equal(y, y_expected)

    def test_input_collapse(self):
        S = (3, 4)
        A1 = TestOpA
        A2 = TestOpB

        H = DiagonalStack((A1, A2))
        assert H.input_shape == (2, *S)

        H = DiagonalStack((A1, A2), collapse_input=False)
        assert H.input_shape == (S, S)

    def test_output_collapse(self):
        A1 = TestOpB
        A2 = TestOpC

        H = DiagonalStack((A1, A2))
        assert H.output_shape == (2, *A1.output_shape)

        H = DiagonalStack((A1, A2), collapse_output=False)
        assert H.output_shape == (A1.output_shape, A1.output_shape)


class TestDiagonalReplicated:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    @pytest.mark.parametrize("map_type", ["auto", "vmap"])
    @pytest.mark.parametrize("input_axis", [0, 1])
    def test_map_auto_vmap(self, input_axis, map_type):
        x, key = randn((2, 3, 4), key=self.key)
        mapshape = (3, 4) if input_axis == 0 else (2, 4)
        replicates = x.shape[input_axis]
        A = Abs(mapshape)
        D = DiagonalReplicated(A, replicates, input_axis=input_axis, map_type=map_type)
        y = D(x)
        assert y.shape[input_axis] == replicates

    @pytest.mark.skipif(jax.device_count() < 2, reason="multiple devices required for test")
    def test_map_auto_pmap(self):
        x, key = randn((2, 3, 4), key=self.key)
        A = Abs(x.shape[1:])
        replicates = x.shape[0]
        D = DiagonalReplicated(A, replicates, map_type="pmap")
        y = D(x)
        assert y.shape[0] == replicates

    def test_input_axis(self):
        # Ensure that operators can be stacked on final axis
        x, key = randn((2, 3, 4), key=self.key)
        A = Abs(x.shape[0:2])
        replicates = x.shape[2]
        D = DiagonalReplicated(A, replicates, input_axis=2)
        y = D(x)
        assert y.shape == (2, 3, 4)
        D = DiagonalReplicated(A, replicates, input_axis=-1)
        y = D(x)
        assert y.shape == (2, 3, 4)

    def test_output_axis(self):
        x, key = randn((2, 3, 4), key=self.key)
        A = Abs(x.shape[1:])
        replicates = x.shape[0]
        D = DiagonalReplicated(A, replicates, output_axis=1)
        y = D(x)
        assert y.shape == (3, 2, 4)
