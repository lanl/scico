import operator as op

import numpy as np

import jax

import pytest

import scico.numpy as snp
from scico.linop import CircularConvolve, Convolve
from scico.random import randint, randn, uniform
from scico.test.linop.test_linop import adjoint_test

SHAPE_SPECS = [
    ((16,), None, (3,)),  # 1D
    ((16, 10), None, (3, 2)),  # 2D
    ((16, 10, 8), None, (3, 2, 4)),  # 3D
    ((2, 16, 10), 2, (3, 2)),  # batching x
    ((16, 10), None, (2, 3, 2)),  # batching h
    ((2, 16, 10), 2, (2, 3, 2)),  # batching both
    # (M, N, b) x (H, W, 1)  # this was the old way
    # (M, N, b) x (H, W)  # this won't work: Luke, firm-no
    # (M, b, N) x (H, W)  # do we even want this?
    # (M, b, N) x (b, H, W) # no, no, no
]


class TestCircularConvolve:
    def setup_method(self, method):
        self.key = jax.random.PRNGKey(12345)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("axes_shape_spec", SHAPE_SPECS)
    def test_eval(self, axes_shape_spec, input_dtype, jit):

        x_shape, ndims, h_shape = axes_shape_spec

        h, key = randn(tuple(h_shape), dtype=input_dtype, key=self.key)
        x, key = randn(tuple(x_shape), dtype=input_dtype, key=key)

        A = CircularConvolve(h, x_shape, ndims, input_dtype, jit=jit)

        Ax = A @ x

        # check that a specific pixel of Ax computes an inner product between x and
        # (flipped, padded, shifted) h
        h_flipped = np.flip(h, range(-A.ndims, 0))  # flip only in the spatial dims (not batches)

        x_inds = (...,) + tuple(
            slice(-h.shape[a], None) for a in range(-A.ndims, 0)
        )  # bottom right corner of x
        Ax_inds = (...,) + tuple(-1 for _ in range(A.ndims))
        sum_axes = tuple(-(a + 1) for a in range(A.ndims))  # ndims=2 -> -1, -2
        np.testing.assert_allclose(
            np.sum(h_flipped * x[x_inds], axis=sum_axes), Ax[Ax_inds], rtol=1e-5
        )

        # np.testing.assert_allclose(Ax.ravel(), hx.ravel(), rtol=5e-4)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("axes_shape_spec", SHAPE_SPECS)
    def test_adjoint(self, axes_shape_spec, input_dtype, jit):

        x_shape, ndims, h_shape = axes_shape_spec

        h, key = randn(tuple(h_shape), dtype=input_dtype, key=self.key)

        A = CircularConvolve(h, x_shape, ndims, input_dtype, jit=jit)

        adjoint_test(A, self.key)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("axes_shape_spec", SHAPE_SPECS)
    @pytest.mark.parametrize("operator", [op.mul, op.truediv])
    def test_scalar_left(self, axes_shape_spec, operator, jit):
        input_dtype = np.float32
        scalar = np.float32(3.141)

        x_shape, ndims, h_shape = axes_shape_spec

        h, key = randn(tuple(h_shape), dtype=input_dtype, key=self.key)

        A = CircularConvolve(h, x_shape, ndims, input_dtype, jit=jit)

        cA = operator(A, scalar)

        np.testing.assert_allclose(operator(A.h_dft.ravel(), scalar), cA.h_dft.ravel(), rtol=5e-5)

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("axes_shape_spec", SHAPE_SPECS)
    @pytest.mark.parametrize("operator", [op.mul])
    def test_scalar_right(self, axes_shape_spec, operator, jit):
        input_dtype = np.float32
        scalar = np.float32(3.141)

        x_shape, ndims, h_shape = axes_shape_spec

        h, key = randn(tuple(h_shape), dtype=input_dtype, key=self.key)

        A = CircularConvolve(h, x_shape, ndims, input_dtype, jit=jit)
        cA = operator(scalar, A)

        np.testing.assert_allclose(operator(scalar, A.h_dft.ravel()), cA.h_dft.ravel(), rtol=5e-5)

    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("jit", [True, False])
    def test_matches_convolve(self, input_dtype, jit):
        h, key = randint(minval=0, maxval=3, shape=(3, 4), key=self.key)
        x, key = uniform(minval=0, maxval=1, shape=(5, 4), key=key)

        h = h.astype(input_dtype)
        x = (x <= 0.1).astype(input_dtype)

        # pad to m + n -1
        x_pad = snp.pad(x, ((0, h.shape[0] - 1), (0, h.shape[1] - 1)))

        A = Convolve(h=h, input_shape=x.shape, jit=jit, input_dtype=input_dtype)
        B = CircularConvolve(h, input_shape=x_pad.shape, jit=jit, input_dtype=input_dtype)

        actual = B @ x_pad
        desired = A @ x
        np.testing.assert_allclose(actual, desired, atol=1e-6)

    @pytest.mark.parametrize("axes_shape_spec", SHAPE_SPECS)
    @pytest.mark.parametrize("input_dtype", [np.float32, np.complex64])
    @pytest.mark.parametrize("jit_old_op", [True, False])
    @pytest.mark.parametrize("jit_new_op", [True, False])
    def test_from_operator(self, axes_shape_spec, input_dtype, jit_old_op, jit_new_op):
        x_shape, ndims, h_shape = axes_shape_spec

        h, key = randn(tuple(h_shape), dtype=input_dtype, key=self.key)
        x, key = randn(tuple(x_shape), dtype=input_dtype, key=key)

        A = CircularConvolve(h, x_shape, ndims, input_dtype, jit=jit_old_op)

        B = CircularConvolve.from_operator(A, ndims, jit=jit_new_op)

        np.testing.assert_allclose(A @ x, B @ x, atol=1e-5)
