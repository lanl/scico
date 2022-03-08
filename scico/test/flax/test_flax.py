from functools import partial

import numpy as np

import pytest
from flax.linen import Conv, BatchNorm, relu, leaky_relu, elu, max_pool

from scico import flax as sflax
from scico import random


class TestSet:
    def test_convnblock_default(self):
        nflt = 16  # number of filters
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=relu,
        )
        assert flxm.kernel_size == (3, 3)  # size of kernel
        assert flxm.strides == (1, 1)  # stride of convolution

    def test_convnblock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=leaky_relu,
            kernel_size=ksz,
            strides=strd,
        )
        assert flxm.act == leaky_relu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution

    def test_convblock_default(self):
        nflt = 16  # number of filters
        conv = partial(Conv, dtype=np.float32)
        flxm = sflax.ConvBlock(
            num_filters=nflt,
            conv=conv,
            act=relu,
        )
        assert flxm.kernel_size == (3, 3)  # size of kernel
        assert flxm.strides == (1, 1)  # stride of convolution

    def test_convblock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        conv = partial(Conv, dtype=np.float32)
        flxm = sflax.ConvBlock(
            num_filters=nflt,
            conv=conv,
            act=elu,
            kernel_size=ksz,
            strides=strd,
        )
        assert flxm.act == elu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution

    def test_convnpblock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        wnd = (2, 2)  # window for pooling
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNPoolBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=relu,
            pool=max_pool,
            kernel_size=ksz,
            strides=strd,
            window_shape=wnd,
        )
        assert flxm.act == relu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution

    def test_convnublock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        upsampling = 2  # upsampling factor
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        upfn = partial(sflax.upscale_nn, scale=upsampling)
        flxm = sflax.ConvBNUpsampleBlock(
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=relu,
            upfn=upfn,
            kernel_size=ksz,
            strides=strd,
        )
        assert flxm.act == relu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution

    def test_convmnblock_default(self):
        nblck = 2  # number of blocks
        nflt = 16  # number of filters
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.ConvBNMultiBlock(
            num_blocks=nblck,
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=relu,
        )
        assert flxm.kernel_size == (3, 3)  # size of kernel
        assert flxm.strides == (1, 1)  # stride of convolution

    def test_resnet_default(self):
        depth = 3  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        N = 128  # image size
        x, key = random.randn((10, N, N, chn), seed=1234)
        resnet = sflax.ResNet(
            depth=depth,
            channels=chn,
            num_filters=num_filters,
        )
        variables = resnet.init(key, x)
        # Test for the construction / forward pass.
        rnx = resnet.apply(variables, x, train=False, mutable=False)
        assert x.dtype == rnx.dtype

    def test_unet_default(self):
        depth = 2  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        N = 128  # image size
        x, key = random.randn((10, N, N, chn), seed=1234)
        unet = sflax.UNet(
            depth=depth,
            channels=chn,
            num_filters=num_filters,
        )
        variables = unet.init(key, x)
        # Test for the construction / forward pass.
        unx = unet.apply(variables, x, train=False, mutable=False)
        assert x.dtype == unx.dtype


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
    fmap = sflax.FlaxMap(testobj.dncnn, testobj.variables)
    N = 128  # image size
    x, key = random.randn((N, N))
    out = fmap(x)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim
