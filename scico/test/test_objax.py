import numpy as np

import pytest

import objax
from scico import objax as sobjax
from scico.functional._objax import ObjaxMap


class TestSet:
    def test_conv_args_default(self):
        d = sobjax.conv_args()
        assert d["w_init"] == objax.nn.init.kaiming_normal

    def test_conv_args(self):
        d = sobjax.conv_args(objax.nn.init.xavier_normal)
        assert d["w_init"] == objax.nn.init.xavier_normal

    def test_convnblock_default(self):
        chn = 1  # number of channels
        nflt = 16  # number of filters
        ksz = 3  # size of kernel
        strd = 1  # stride of convolution
        objm = sobjax.ConvBN_Block(chn, nflt, ksz, strd)
        assert isinstance(objm.norm, objax.nn.BatchNorm2D)
        assert objm.activation_fn == objax.functional.relu

    def test_convnblock_args(self):
        chn = 1  # number of channels
        nflt = 16  # number of filters
        ksz = 3  # size of kernel
        strd = 1  # stride of convolution
        objm = sobjax.ConvBN_Block(
            chn,
            nflt,
            ksz,
            strd,
            normalization_fn=objax.nn.SyncedBatchNorm2D,
            activation_fn=objax.functional.elu,
        )
        assert isinstance(objm.norm, objax.nn.SyncedBatchNorm2D)
        assert objm.activation_fn == objax.functional.elu


class DnCNN_NetTest:
    def __init__(self):
        depth = 3  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        N = 128  # image size
        objax.random.Generator(1234)
        self.x = objax.random.normal((10, chn, N, N))
        self.dncnn = sobjax.DnCNN_Net(
            depth,
            chn,
            num_filters,
        )


@pytest.fixture(scope="module")
def testobj():
    yield DnCNN_NetTest()


def test_DnCNN_call(testobj):
    # Test for the construction / forward pass
    dnx = testobj.dncnn(testobj.x, training=False)
    assert testobj.x.dtype == dnx.dtype


def test_DnCNN_train(testobj):
    # Test training flag for training
    bn0mean_before = testobj.dncnn.layers[0].norm.running_mean
    dnx = testobj.dncnn(testobj.x, training=True)
    bn0mean_after = testobj.dncnn.layers[0].norm.running_mean
    try:
        np.testing.assert_allclose(bn0mean_before, bn0mean_after, rtol=1e-5)
    except Exception as e:
        print(e)
        assert 0


def test_DnCNN_test(testobj):
    # Test training flag for testing
    bn0mean_before = testobj.dncnn.layers[0].norm.running_var
    dnx = testobj.dncnn(testobj.x, training=False)
    bn0mean_after = testobj.dncnn.layers[0].norm.running_var
    np.testing.assert_allclose(bn0mean_before, bn0mean_after, rtol=1e-5)


def test_ObjaxMap_call(testobj):
    # Test for the usage of objax model as a map
    omap = ObjaxMap(testobj.dncnn)
    N = 128  # image size
    x = objax.random.normal((N, N))
    out = omap.prox(x, 0.1)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim
