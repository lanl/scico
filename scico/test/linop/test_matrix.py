import operator as op

import numpy as np

import jax
from jax.interpreters.xla import DeviceArray

import pytest

import scico.numpy as snp
from scico import linop
from scico.linop import MatrixOperator
from scico.random import randn
from scico.test.linop.test_linop import AbsMatOp


class TestMatrix:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_eval(self, input_shape, input_dtype):

        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)

        x, key = randn(Ao.input_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(A @ x, Ao @ x)

        # Invalid shapes
        with pytest.raises(TypeError):
            y, key = randn((64,), dtype=Ao.input_dtype, key=key)
            _ = Ao @ y

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_adjoint(self, input_shape, input_dtype):

        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)

        x, key = randn(Ao.output_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(A.conj().T @ x, Ao.conj().T @ x)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_adjoint_method(self, input_shape, input_dtype):
        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)
        x, key = randn(Ao.output_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(Ao.adj(x), Ao.conj().T @ x)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_hermetian_method(self, input_shape, input_dtype):
        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)
        x, key = randn(Ao.output_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(Ao.H @ x, Ao.conj().T @ x)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_gram_method(self, input_shape, input_dtype):
        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)
        x, key = randn(Ao.input_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(Ao.gram(x), A.conj().T @ A @ x, rtol=5e-5)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("input_shape", [(3, 3), (3, 4)])
    def test_gram_op(self, input_shape, input_dtype):
        A, key = randn(input_shape, dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)
        G = Ao.gram_op
        x, key = randn(Ao.input_shape, dtype=Ao.input_dtype, key=key)
        np.testing.assert_allclose(G @ x, A.conj().T @ A @ x, rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_add_sub(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)
        C, key = randn((4, 4), key=key)
        x, key = randn((6,), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        Co = MatrixOperator(C)

        ABx = operator(Ao, Bo) @ x
        AxBx = operator(Ao @ x, Bo @ x)
        np.testing.assert_allclose(ABx, AxBx, rtol=5e-5)

        with pytest.raises(ValueError):
            operator(Ao, Co)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_scalar_left(self, operator):
        scalar = np.float32(np.random.randn())

        A, key = randn((4, 6), key=self.key)
        x, key = randn((6,), key=key)
        Ao = MatrixOperator(A)

        np.testing.assert_allclose(operator(scalar, Ao) @ x, operator(scalar, A) @ x, rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_scalar_right(self, operator):
        scalar = np.float32(np.random.randn())

        A, key = randn((4, 6), key=self.key)
        x, key = randn((6,), key=key)
        Ao = MatrixOperator(A)

        np.testing.assert_allclose(operator(Ao, scalar) @ x, operator(A, scalar) @ x, rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_elementwise_matops(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)

        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)

        np.testing.assert_allclose(operator(Ao, Bo).A, operator(A, B), rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_elementwise_array_left(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        np.testing.assert_allclose(operator(Ao, B).A, operator(A, B), rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_elementwise_array_right(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        np.testing.assert_allclose(operator(A, Bo).A, operator(A, B), rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_elementwise_matop_shape_mismatch(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 4), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        with pytest.raises(ValueError):
            operator(Ao, Bo)

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_elementwise_array_shape_mismatch(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 4), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        with pytest.raises(ValueError):
            operator(Ao, B)

        with pytest.raises(ValueError):
            operator(B, Ao)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_elementwise_linop(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)
        Ao = MatrixOperator(A)
        Bo = AbsMatOp(B)
        x, key = randn(Ao.input_shape, dtype=Ao.input_dtype, key=key)

        np.testing.assert_allclose(operator(Ao, Bo) @ x, operator(Ao @ x, Bo @ x), rtol=5e-5)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_elementwise_linop_mismatch(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 4), key=key)
        Ao = MatrixOperator(A)
        Bo = AbsMatOp(B)
        with pytest.raises(ValueError):
            operator(Ao, Bo)

    @pytest.mark.parametrize("operator", [op.mul, op.truediv])
    def test_elementwise_linop_invalid(self, operator):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((4, 6), key=key)
        Ao = MatrixOperator(A)
        Bo = AbsMatOp(B)
        with pytest.raises(TypeError):
            operator(Ao, Bo)

        with pytest.raises(TypeError):
            operator(Bo, Ao)

    def test_matmul(self):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((6, 3), key=key)
        Ao = MatrixOperator(A)
        Bo = MatrixOperator(B)
        x, key = randn(Bo.input_shape, dtype=Ao.input_dtype, key=key)

        AB = Ao @ Bo
        np.testing.assert_allclose((Ao @ Bo) @ x, Ao @ (Bo @ x), rtol=5e-5)

    def test_matmul_linop(self):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((6, 3), key=key)
        Ao = MatrixOperator(A)
        Bo = AbsMatOp(B)
        x, key = randn(Bo.input_shape, dtype=Ao.input_dtype, key=key)

        AB = Ao @ Bo
        np.testing.assert_allclose((Ao @ Bo) @ x, Ao @ (Bo @ x), rtol=5e-5)

    def test_matmul_linop_shape_mismatch(self):
        A, key = randn((4, 6), key=self.key)
        B, key = randn((5, 3), key=key)
        Ao = MatrixOperator(A)
        Bo = AbsMatOp(B)
        with pytest.raises(ValueError):
            _ = Ao @ Bo

    def test_matmul_identity(self):
        A, key = randn((4, 6), key=self.key)
        Ao = MatrixOperator(A)
        I = linop.Identity(input_shape=(6,))
        assert Ao == Ao @ I

    def test_init_devicearray(self):
        A = np.random.randn(4, 6)
        Ao = MatrixOperator(A)
        assert isinstance(Ao.A, DeviceArray)

        with pytest.raises(TypeError):
            MatrixOperator([1.0, 3.0])

    @pytest.mark.parametrize("input_shape", [(3,), (2, 3, 4)])
    def test_init_wrong_dims(self, input_shape):
        A = np.random.randn(*input_shape)
        with pytest.raises(TypeError):
            Ao = MatrixOperator(A)

    def test_to_array(self):
        A = np.random.randn(4, 6)
        Ao = MatrixOperator(A)
        A_array = Ao.to_array()
        assert isinstance(A_array, np.ndarray)
        np.testing.assert_allclose(A_array, A)

    @pytest.mark.parametrize("ord", ["fro", 2])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    def test_norm(self, ord, axis, keepdims, input_dtype):  # pylint: disable=W0622
        A, key = randn((4, 6), dtype=input_dtype, key=self.key)
        Ao = MatrixOperator(A)

        if ord == "fro" and axis is not None:
            # Not defined;
            pass
        else:
            x = Ao.norm(ord=ord, axis=axis, keepdims=keepdims)
            y = snp.linalg.norm(A, ord=ord, axis=axis, keepdims=keepdims)
            np.testing.assert_allclose(x, y, rtol=5e-5)
