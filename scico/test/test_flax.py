from functools import partial

import numpy as np

import pytest
from flax import linen as nn

from scico import _flax as sflax
from scico import random
from scico._flax import FlaxMap


class TestSet:
    def test_convnblock_default(self):
        nflt = 16  # number of filters
        conv = partial(nn.Conv, dtype=np.float32)
        norm = partial(nn.BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=nn.relu,
        )
        assert flxm.kernel_size == (3, 3)  # size of kernel
        assert flxm.strides == (1, 1)  # stride of convolution

    def test_convnblock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        conv = partial(nn.Conv, dtype=np.float32)
        norm = partial(nn.BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=nn.leaky_relu,
            kernel_size=ksz,
            strides=strd,
        )
        assert flxm.act == nn.leaky_relu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution


class DnCNNNetTest:
    def __init__(self):
        depth = 3  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        N = 128  # image size
        self.x, key = random.randn((10, N, N, chn), seed=1234)
        self.dncnn = sflax.DnCNNNet(
            depth=depth,
            channels=chn,
            num_filters=num_filters,
        )
        self.variables = self.dncnn.init(key, self.x)


@pytest.fixture(scope="module")
def testobj():
    yield DnCNNNetTest()


def test_DnCNN_call(testobj):
    # Test for the construction / forward pass.
    dnx = testobj.dncnn.apply(testobj.variables, testobj.x, train=False, mutable=False)
    assert testobj.x.dtype == dnx.dtype


def test_DnCNN_train(testobj):
    # Test effect of training flag.
    bn0bias_before = testobj.variables["params"]["ConvBNBlock_0"]["BatchNorm_0"]["bias"]
    bn0mean_before = testobj.variables["batch_stats"]["ConvBNBlock_0"]["BatchNorm_0"]["mean"]
    dnx, new_state = testobj.dncnn.apply(
        testobj.variables, testobj.x, train=True, mutable=["batch_stats"]
    )
    bn0mean_new = new_state["batch_stats"]["ConvBNBlock_0"]["BatchNorm_0"]["mean"]
    bn0bias_after = testobj.variables["params"]["ConvBNBlock_0"]["BatchNorm_0"]["bias"]
    bn0mean_after = testobj.variables["batch_stats"]["ConvBNBlock_0"]["BatchNorm_0"]["mean"]
    try:
        np.testing.assert_allclose(bn0bias_before, bn0bias_after, rtol=1e-5)
        np.testing.assert_allclose(
            bn0mean_new - bn0mean_before, bn0mean_new + bn0mean_after, rtol=1e-5
        )
    except Exception as e:
        print(e)
        assert 0


def test_DnCNN_test(testobj):
    # Test effect of training flag.
    bn0var_before = testobj.variables["batch_stats"]["ConvBNBlock_0"]["BatchNorm_0"]["var"]
    dnx, new_state = testobj.dncnn.apply(
        testobj.variables, testobj.x, train=False, mutable=["batch_stats"]
    )
    bn0var_after = new_state["batch_stats"]["ConvBNBlock_0"]["BatchNorm_0"]["var"]
    np.testing.assert_allclose(bn0var_before, bn0var_after, rtol=1e-5)


def test_FlaxMap_call(testobj):
    # Test for the usage of flax model as a map.
    fmap = FlaxMap(testobj.dncnn, testobj.variables)
    N = 128  # image size
    x, key = random.randn((N, N))
    out = fmap(x)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim
