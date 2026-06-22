import operator as op

import numpy as np

from jax import config

import pytest

# enable 64-bit mode for output dtype checks
config.update("jax_enable_x64", True)

import jax

from test_linop import adjoint_test

import scico.numpy as snp
from scico import linop
from scico.random import randn


class TestDiagonal:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    input_shapes = [(8,), (8, 12), ((3,), (4, 5))]

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_eval(self, input_shape, diagonal_dtype):
        diagonal, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)
        x, key = randn(input_shape, dtype=diagonal_dtype, key=key)

        D = linop.Diagonal(diagonal=diagonal)
        assert (D @ x).shape == D.output_shape
        snp.testing.assert_allclose((diagonal * x), (D @ x), rtol=1e-5)

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    def test_eval_broadcasting(self, diagonal_dtype):
        # array broadcast
        diagonal, key = randn((3, 1, 4), dtype=diagonal_dtype, key=self.key)
        x, key = randn((5, 1), dtype=diagonal_dtype, key=key)
        D = linop.Diagonal(diagonal, x.shape)
        assert (D @ x).shape == (3, 5, 4)
        np.testing.assert_allclose((diagonal * x).ravel(), (D @ x).ravel(), rtol=1e-5)

        # blockarray broadcast
        diagonal, key = randn(((3, 1, 4), (5, 5)), dtype=diagonal_dtype, key=self.key)
        x, key = randn(((5, 1), (1,)), dtype=diagonal_dtype, key=key)
        D = linop.Diagonal(diagonal, x.shape)
        assert (D @ x).shape == ((3, 5, 4), (5, 5))
        snp.testing.assert_allclose((diagonal * x), (D @ x), rtol=1e-5)

        # blockarray x array -> error
        diagonal, key = randn(((3, 1, 4), (5, 5)), dtype=diagonal_dtype, key=self.key)
        x, key = randn((5, 1), dtype=diagonal_dtype, key=key)
        with pytest.raises(ValueError):
            D = linop.Diagonal(diagonal, x.shape)

        # array x blockarray -> error
        diagonal, key = randn((3, 1, 4), dtype=diagonal_dtype, key=self.key)
        x, key = randn(((5, 1), (1,)), dtype=diagonal_dtype, key=key)
        with pytest.raises(ValueError):
            D = linop.Diagonal(diagonal, x.shape)

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_adjoint(self, input_shape, diagonal_dtype):
        diagonal, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)
        D = linop.Diagonal(diagonal=diagonal)

        adjoint_test(D)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape1", input_shapes)
    @pytest.mark.parametrize("input_shape2", input_shapes)
    def test_binary_op(self, input_shape1, input_shape2, diagonal_dtype, operator):
        diagonal1, key = randn(input_shape1, dtype=diagonal_dtype, key=self.key)
        diagonal2, key = randn(input_shape2, dtype=diagonal_dtype, key=key)
        x, key = randn(input_shape1, dtype=diagonal_dtype, key=key)

        D1 = linop.Diagonal(diagonal=diagonal1)
        D2 = linop.Diagonal(diagonal=diagonal2)

        if input_shape1 != input_shape2:
            with pytest.raises(ValueError):
                a = operator(D1, D2) @ x
        else:
            a = operator(D1, D2) @ x
            Dnew = linop.Diagonal(operator(diagonal1, diagonal2))
            b = Dnew @ x
            snp.testing.assert_allclose(a, b, rtol=1e-5)

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape1", input_shapes)
    @pytest.mark.parametrize("input_shape2", input_shapes)
    def test_matmul(self, input_shape1, input_shape2, diagonal_dtype):
        diagonal1, key = randn(input_shape1, dtype=diagonal_dtype, key=self.key)
        diagonal2, key = randn(input_shape2, dtype=diagonal_dtype, key=key)
        x, key = randn(input_shape1, dtype=diagonal_dtype, key=key)

        D1 = linop.Diagonal(diagonal=diagonal1)
        D2 = linop.Diagonal(diagonal=diagonal2)

        if input_shape1 != input_shape2:
            with pytest.raises(ValueError):
                D3 = D1 @ D2
        else:
            D3 = D1 @ D2
            assert isinstance(D3, linop.Diagonal)
            a = D3 @ x
            D4 = linop.Diagonal(diagonal1 * diagonal2)
            b = D4 @ x
            snp.testing.assert_allclose(a, b, rtol=1e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_binary_op_mismatch(self, operator):
        diagonal_dtype = np.float32
        input_shape1 = (8,)
        input_shape2 = (12,)
        diagonal1, key = randn(input_shape1, dtype=diagonal_dtype, key=self.key)
        diagonal2, key = randn(input_shape2, dtype=diagonal_dtype, key=key)

        D1 = linop.Diagonal(diagonal=diagonal1)
        D2 = linop.Diagonal(diagonal=diagonal2)
        with pytest.raises(ValueError):
            operator(D1, D2)

    @pytest.mark.parametrize("operator", [op.mul, op.truediv])
    def test_scalar_right(self, operator):
        if operator == op.truediv:
            pytest.xfail("scalar / LinearOperator is not supported")

        diagonal_dtype = np.float32
        input_shape = (8,)

        diagonal1, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)
        scalar = np.random.randn()
        x, key = randn(input_shape, dtype=diagonal_dtype, key=key)

        D = linop.Diagonal(diagonal=diagonal1)
        scaled_D = operator(scalar, D)

        np.testing.assert_allclose(scaled_D @ x, operator(scalar, D @ x), rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.mul, op.truediv])
    def test_scalar_left(self, operator):
        diagonal_dtype = np.float32
        input_shape = (8,)

        diagonal1, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)
        scalar = np.random.randn()
        x, key = randn(input_shape, dtype=diagonal_dtype, key=key)

        D = linop.Diagonal(diagonal=diagonal1)
        scaled_D = operator(D, scalar)

        np.testing.assert_allclose(scaled_D @ x, operator(D @ x, scalar), rtol=5e-5)

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    def test_gram_op(self, diagonal_dtype):
        input_shape = (7,)
        diagonal, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)

        D1 = linop.Diagonal(diagonal=diagonal)
        D2 = D1.gram_op
        D3 = D1.H @ D1
        assert isinstance(D3, linop.Diagonal)
        snp.testing.assert_allclose(D2.diagonal, D3.diagonal, rtol=1e-6)

    @pytest.mark.parametrize("diagonal_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("ord", [None, "fro", "nuc", -np.inf, np.inf, 1, -1, 2, -2])
    def test_norm(self, diagonal_dtype, ord):
        input_shape = (5,)
        diagonal, key = randn(input_shape, dtype=diagonal_dtype, key=self.key)

        D1 = linop.Diagonal(diagonal=diagonal)
        D2 = snp.diag(diagonal)
        n1 = D1.norm(ord=ord)
        n2 = snp.linalg.norm(D2, ord=ord)
        snp.testing.assert_allclose(n1, n2, rtol=1e-6)

    def test_norm_except(self):
        input_shape = (5,)
        diagonal, key = randn(input_shape, dtype=np.float32, key=self.key)

        D = linop.Diagonal(diagonal=diagonal)
        with pytest.raises(ValueError):
            n = D.norm(ord=3)


class TestScaledIdentity:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    input_shapes = [(8,), (8, 12), ((3,), (4, 5))]

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_eval(self, input_shape, input_dtype):
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        scalar, key = randn((), dtype=input_dtype, key=key)

        Id = linop.ScaledIdentity(scalar=scalar, input_shape=input_shape, input_dtype=input_dtype)
        assert (Id @ x).shape == Id.output_shape
        snp.testing.assert_allclose(scalar * x, Id @ x, rtol=1e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_binary_op(self, input_shape, operator):
        input_dtype = np.float32
        diagonal, key = randn(input_shape, dtype=input_dtype, key=self.key)
        x, key = randn(input_shape, dtype=input_dtype, key=key)
        scalar, key = randn((), dtype=input_dtype, key=key)

        Id = linop.ScaledIdentity(scalar, input_shape=input_shape)
        D = linop.Diagonal(diagonal=diagonal)

        IdD = operator(Id, D)
        assert isinstance(IdD, linop.Diagonal)
        snp.testing.assert_allclose(IdD @ x, operator(scalar, diagonal) * x, rtol=1e-6)

        DId = operator(D, Id)
        assert isinstance(DId, linop.Diagonal)
        snp.testing.assert_allclose(DId @ x, operator(diagonal, scalar) * x, rtol=1e-6)

    def test_scale(self):
        input_shape = (5,)
        input_dtype = np.float32
        scalar1, key = randn((), dtype=input_dtype, key=self.key)
        scalar2, key = randn((), dtype=input_dtype, key=key)

        x, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Id = linop.ScaledIdentity(scalar=scalar1, input_shape=input_shape, input_dtype=input_dtype)

        sId = scalar2 * Id
        assert isinstance(sId, linop.ScaledIdentity)
        snp.testing.assert_allclose(sId @ x, scalar1 * scalar2 * x, rtol=1e-6)

        Ids = Id * scalar2
        assert isinstance(Ids, linop.ScaledIdentity)
        snp.testing.assert_allclose(Ids @ x, scalar1 * scalar2 * x, rtol=1e-6)

        Idds = Id / scalar2
        assert isinstance(Idds, linop.ScaledIdentity)
        snp.testing.assert_allclose(Idds @ x, x * scalar1 / scalar2, rtol=1e-6)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("ord", [None, "fro", "nuc", -np.inf, np.inf, 1, -1, 2, -2])
    def test_norm(self, input_dtype, ord):
        input_shape = (5,)
        scalar, key = randn((), dtype=input_dtype, key=self.key)

        Id = linop.ScaledIdentity(scalar=scalar, input_shape=input_shape, input_dtype=input_dtype)
        D = linop.Diagonal(
            diagonal=scalar * snp.ones(input_shape),
            input_shape=input_shape,
            input_dtype=input_dtype,
        )
        n1 = Id.norm(ord=ord)
        n2 = D.norm(ord=ord)
        snp.testing.assert_allclose(n1, n2, rtol=1e-6)

    def test_norm_except(self):
        input_shape = (5,)

        Id = linop.Identity(input_shape=input_shape, input_dtype=np.float32)
        with pytest.raises(ValueError):
            n = Id.norm(ord=3)


class TestIdentity:
    def setup_method(self, method):
        self.key = jax.random.key(12345)

    input_shapes = [(8,), (8, 12), ((3,), (4, 5))]

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_eval(self, input_shape, input_dtype):
        x, key = randn(input_shape, dtype=input_dtype, key=self.key)

        Id = linop.Identity(input_shape=input_shape, input_dtype=input_dtype)
        assert (Id @ x).shape == Id.output_shape
        snp.testing.assert_allclose(x, Id @ x, rtol=1e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    @pytest.mark.parametrize("input_shape", input_shapes)
    def test_binary_op(self, input_shape, operator):
        input_dtype = np.float32
        diagonal, key = randn(input_shape, dtype=input_dtype, key=self.key)
        scalar, key = randn((), dtype=input_dtype, key=key)
        x, key = randn(input_shape, dtype=input_dtype, key=key)

        Id = linop.Identity(input_shape=input_shape)
        Ids = linop.ScaledIdentity(scalar=scalar, input_shape=input_shape)
        D = linop.Diagonal(diagonal=diagonal)

        IdD = operator(Id, D)
        assert isinstance(IdD, linop.Diagonal)
        snp.testing.assert_allclose(IdD @ x, operator(1.0, diagonal) * x, rtol=1e-6)

        DId = operator(D, Id)
        assert isinstance(DId, linop.Diagonal)
        snp.testing.assert_allclose(DId @ x, operator(diagonal, 1.0) * x, rtol=1e-6)

        IdIds = operator(Id, Ids)
        assert isinstance(IdIds, linop.ScaledIdentity)
        snp.testing.assert_allclose(IdIds @ x, operator(1.0, scalar) * x, rtol=1e-6)

        IdsId = operator(Ids, Id)
        assert isinstance(IdsId, linop.ScaledIdentity)
        snp.testing.assert_allclose(IdsId @ x, operator(scalar, 1.0) * x, rtol=1e-6)

    def test_scale(self):
        input_shape = (5,)
        input_dtype = np.float32
        scalar, key = randn((), dtype=input_dtype, key=self.key)
        x, key = randn(input_shape, dtype=input_dtype, key=key)
        Id = linop.Identity(input_shape=input_shape, input_dtype=input_dtype)

        sId = scalar * Id
        assert isinstance(sId, linop.ScaledIdentity)
        snp.testing.assert_allclose(sId @ x, scalar * x, rtol=1e-6)

        Ids = Id * scalar
        assert isinstance(Ids, linop.ScaledIdentity)
        snp.testing.assert_allclose(Ids @ x, scalar * x, rtol=1e-6)

        Idds = Id / scalar
        assert isinstance(Idds, linop.ScaledIdentity)
        snp.testing.assert_allclose(Idds @ x, x / scalar, rtol=1e-6)
