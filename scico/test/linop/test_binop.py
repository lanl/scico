import operator as op

import pytest

import scico.numpy as snp
from scico import linop
from scico.operator import Abs, Operator


class TestBinaryOp:
    def setup_method(self, method):
        self.input_shape = (5,)
        self.input_dtype = snp.float32

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_case1(self, operator):
        A = linop.Convolve(
            snp.ones((2,)), input_shape=self.input_shape, input_dtype=self.input_dtype, mode="same"
        )
        B = Abs(input_shape=self.input_shape, input_dtype=self.input_dtype)

        assert type(operator(A, B)) == Operator
        assert type(operator(B, A)) == Operator
        assert type(operator(2.0 * A, 3.0 * B)) == Operator
        assert type(operator(2.0 * B, 3.0 * A)) == Operator

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_case2(self, operator):
        A = linop.Convolve(
            snp.ones((2,)), input_shape=self.input_shape, input_dtype=self.input_dtype, mode="same"
        )
        B = linop.Identity(input_shape=self.input_shape, input_dtype=self.input_dtype)

        assert type(operator(A, B)) == linop.LinearOperator
        assert type(operator(B, A)) == linop.LinearOperator
        assert type(operator(2.0 * A, 3.0 * B)) == linop.LinearOperator
        assert type(operator(2.0 * B, 3.0 * A)) == linop.LinearOperator

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_case3(self, operator):
        A = linop.SingleAxisFiniteDifference(
            input_shape=self.input_shape, input_dtype=self.input_dtype, circular=True
        )
        B = linop.Identity(input_shape=self.input_shape, input_dtype=self.input_dtype)

        assert type(operator(A, B)) == linop.LinearOperator
        assert type(operator(B, A)) == linop.LinearOperator
        assert type(operator(2.0 * A, 3.0 * B)) == linop.LinearOperator
        assert type(operator(2.0 * B, 3.0 * A)) == linop.LinearOperator

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_case4(self, operator):
        A = linop.ScaledIdentity(
            scalar=0.5, input_shape=self.input_shape, input_dtype=self.input_dtype
        )
        B = linop.Identity(input_shape=self.input_shape, input_dtype=self.input_dtype)

        assert type(operator(A, B)) == linop.ScaledIdentity
        assert type(operator(B, A)) == linop.ScaledIdentity
        assert type(operator(2.0 * A, 3.0 * B)) == linop.ScaledIdentity
        assert type(operator(2.0 * B, 3.0 * A)) == linop.ScaledIdentity
