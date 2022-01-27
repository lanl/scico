import numpy as np

import scico.numpy as snp
from scico import metric


class TestSet:
    def setup_method(self, method):
        np.random.seed(12345)

    def test_mae_mse(self):
        N = 16
        x = np.random.randn(N)
        y = x.copy()
        y[0] = 0
        xe = np.abs(x[0])
        e1 = metric.mae(x, y)
        e2 = metric.mse(x, y)
        assert np.abs(e1 - xe / N) < 1e-12
        assert np.abs(e2 - (xe ** 2) / N) < 1e-12

    def test_snr_nrm(self):
        N = 16
        x = np.random.randn(N)
        x /= np.sqrt(np.var(x))
        y = x + 1
        assert np.abs(metric.snr(x, y)) < 1e-6

    def test_snr_signal_range(self):
        N = 16
        x = np.random.randn(N)
        x -= x.min()
        x /= x.max()
        y = x + 1
        assert np.abs(metric.psnr(x, y)) < 1e-6

    def test_psnr(self):
        N = 16
        x = np.random.randn(N)
        y = x + 1
        assert np.abs(metric.psnr(x, y, signal_range=1.0)) < 1e-6

    def test_isnr(self):
        N = 16
        x = np.random.randn(N)
        y = np.random.randn(N)
        assert np.abs(metric.isnr(x, y, y)) < 1e-6

    def test_bsnr(self):
        N = 16
        x = np.random.randn(N)
        x /= np.sqrt(np.var(x))
        n = np.random.randn(N)
        n /= np.sqrt(np.var(n))
        y = x + n
        assert np.abs(metric.bsnr(x, y)) < 1e-6


def test_rel_res():
    A = snp.array([[2, -1], [1, 0], [-1, 1]], dtype=snp.float32)
    x = snp.array([[3], [-2]], dtype=snp.float32)
    Ax = snp.matmul(A, x)
    b = snp.array([[8], [3], [-5]], dtype=snp.float32)
    assert 0.0 == metric.rel_res(Ax, b)

    A = snp.array([[2, -1], [1, 0], [-1, 1]], dtype=snp.float32)
    x = snp.array([[0], [0]], dtype=snp.float32)
    Ax = snp.matmul(A, x)
    b = snp.array([[0], [0], [0]], dtype=snp.float32)
    assert 0.0 == metric.rel_res(Ax, b)
