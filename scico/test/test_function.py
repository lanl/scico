import numpy as np

import pytest

import scico.numpy as snp
from scico.function import Function
from scico.linop import jacobian
from scico.random import randn


class TestFunction:
    def setup_method(self):
        key = None
        self.shape = (7, 8)
        self.dtype = snp.float32
        self.x, key = randn(self.shape, key=key, dtype=self.dtype)
        self.y, key = randn(self.shape, key=key, dtype=self.dtype)
        self.func = lambda x, y: snp.abs(x) + snp.abs(y)

    def test_init(self):
        F = Function((self.shape, self.shape), input_dtypes=self.dtype, eval_fn=self.func)
        assert F.output_shape == self.shape
        assert len(F.input_dtypes) == 2
        assert F.output_dtype == self.dtype

    def test_eval(self):
        F = Function(
            (self.shape, self.shape),
            output_shape=self.shape,
            eval_fn=self.func,
            input_dtypes=(self.dtype, self.dtype),
            output_dtype=self.dtype,
        )
        np.testing.assert_allclose(self.func(self.x, self.y), F(self.x, self.y))

    def test_eval_jit(self):
        F = Function(
            (self.shape, self.shape),
            output_shape=self.shape,
            eval_fn=self.func,
            input_dtypes=(self.dtype, self.dtype),
            output_dtype=self.dtype,
            jit=True,
        )
        np.testing.assert_allclose(self.func(self.x, self.y), F(self.x, self.y))

    def test_slice(self):
        F = Function((self.shape, self.shape), input_dtypes=self.dtype, eval_fn=self.func)
        Op = F.slice(0, self.y)
        np.testing.assert_allclose(Op(self.x), F(self.x, self.y))

    def test_join(self):
        F = Function((self.shape, self.shape), input_dtypes=self.dtype, eval_fn=self.func)
        Op = F.join()
        np.testing.assert_allclose(Op(snp.blockarray((self.x, self.y))), F(self.x, self.y))

    def test_join_raise(self):
        F = Function(
            (self.shape, self.shape), input_dtypes=(snp.float32, snp.complex64), eval_fn=self.func
        )
        with pytest.raises(ValueError):
            Op = F.join()


@pytest.mark.parametrize("dtype", [snp.float32, snp.complex64])
def test_jacobian(dtype):
    N = 7
    M = 8
    key = None
    fmx, key = randn((M, N), key=key, dtype=dtype)
    gmx, key = randn((M, N), key=key, dtype=dtype)
    F = Function(((N, 1), (N, 1)), input_dtypes=dtype, eval_fn=lambda x, y: fmx @ x + gmx @ y)
    u0, key = randn((N, 1), key=key, dtype=dtype)
    u1, key = randn((N, 1), key=key, dtype=dtype)
    v, key = randn((N, 1), key=key, dtype=dtype)
    w, key = randn((M, 1), key=key, dtype=dtype)

    op = F.slice(0, u1)
    J0op = jacobian(op, u0)
    np.testing.assert_allclose(J0op(v), F.jvp(0, v, u0, u1)[1])
    np.testing.assert_allclose(J0op.H(w), F.vjp(0, u0, u1)[1](w))
    J0fn = F.jacobian(0, u0, u1)
    np.testing.assert_allclose(J0op(v), J0fn(v))
    np.testing.assert_allclose(J0op.H(w), J0fn.H(w))

    op = F.slice(1, u0)
    J1op = jacobian(op, u1)
    np.testing.assert_allclose(J1op(v), F.jvp(1, v, u0, u1)[1])
    np.testing.assert_allclose(J1op.H(w), F.vjp(1, u0, u1)[1](w))
    J1fn = F.jacobian(1, u0, u1)
    np.testing.assert_allclose(J1op(v), J1fn(v))
    np.testing.assert_allclose(J1op.H(w), J1fn.H(w))
