import numpy as np

import pytest

import scico.numpy as snp
from scico import functional


@pytest.mark.parametrize("axis", [0, 1, (0, 2)])
def test_l21norm(axis):
    x = np.ones((3, 4, 5))
    if isinstance(axis, int):
        l2axis = (axis,)
    else:
        l2axis = axis
    l2shape = [x.shape[k] for k in l2axis]
    l1axis = tuple(set(range(len(x))) - set(l2axis))
    l1shape = [x.shape[k] for k in l1axis]

    l21ana = np.sqrt(np.prod(l2shape)) * np.prod(l1shape)
    F = functional.L21Norm(l2_axis=axis)
    l21num = F(x)
    np.testing.assert_allclose(l21ana, l21num, rtol=1e-5)

    l2ana = np.sqrt(np.prod(l2shape))
    prxana = (l2ana - 1.0) / l2ana * x
    prxnum = F.prox(x, 1.0)
    np.testing.assert_allclose(prxana, prxnum, rtol=1e-5)


def test_l2norm_blockarray():
    xa = np.random.randn(2, 3, 4)
    xb = snp.blockarray((xa[0], xa[1]))

    fa = functional.L21Norm(l2_axis=(1, 2))
    fb = functional.L21Norm(l2_axis=None)

    np.testing.assert_allclose(fa(xa), fb(xb), rtol=1e-6)

    ya = fa.prox(xa)
    yb = fb.prox(xb)

    np.testing.assert_allclose(ya[0], yb[0], rtol=1e-6)
    np.testing.assert_allclose(ya[1], yb[1], rtol=1e-6)
