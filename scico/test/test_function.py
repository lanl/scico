import numpy as np

import scico.numpy as snp
from scico.function import Function
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
