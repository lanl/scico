import os
import tempfile
from functools import partial

import numpy as np

import pytest

from flax.core import unfreeze
from flax.errors import ScopeParamShapeError
from flax.linen import BatchNorm, Conv, elu, leaky_relu, max_pool, relu
from scico import flax as sflax
from scico.data import _flax_data_path
from scico.random import randn


class TestSet:
    def test_convnblock_default(self):
        nflt = 16  # number of filters
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.blocks.ConvBNBlock(
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
        flxm = sflax.blocks.ConvBNBlock(
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
        flxm = sflax.blocks.ConvBlock(
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
        flxm = sflax.blocks.ConvBlock(
            num_filters=nflt,
            conv=conv,
            act=elu,
            kernel_size=ksz,
            strides=strd,
        )
        assert flxm.act == elu
        assert flxm.kernel_size == ksz  # size of kernel
        assert flxm.strides == strd  # stride of convolution

    def test_convblock_call(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        conv = partial(Conv, dtype=np.float32)
        flxb = sflax.blocks.ConvBlock(
            num_filters=nflt,
            conv=conv,
            act=elu,
            kernel_size=ksz,
            strides=strd,
        )
        chn = 1  # number of channels
        N = 128  # image size
        x, key = randn((10, N, N, chn), seed=1234)
        variables = flxb.init(key, x)
        # Test for the construction / forward pass.
        cbx = flxb.apply(variables, x)
        assert x.dtype == cbx.dtype

    def test_convnpblock_args(self):
        nflt = 16  # number of filters
        ksz = (5, 5)  # size of kernel
        strd = (2, 2)  # stride of convolution
        wnd = (2, 2)  # window for pooling
        conv = partial(Conv, dtype=np.float32)
        norm = partial(BatchNorm, dtype=np.float32)
        flxm = sflax.blocks.ConvBNPoolBlock(
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
        upfn = partial(sflax.blocks.upscale_nn, scale=upsampling)
        flxm = sflax.blocks.ConvBNUpsampleBlock(
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
        flxm = sflax.blocks.ConvBNMultiBlock(
            num_blocks=nblck,
            num_filters=nflt,
            conv=conv,
            norm=norm,
            act=relu,
        )
        assert flxm.kernel_size == (3, 3)  # size of kernel
        assert flxm.strides == (1, 1)  # stride of convolution

    def test_upscale(self):
        N = 128  # image size
        chn = 3  # channels
        x, key = randn((10, N, N, chn), seed=1234)

        xups = sflax.blocks.upscale_nn(x)
        assert xups.shape == (10, 2 * N, 2 * N, chn)

    def test_resnet_default(self):
        depth = 3  # depth of model
        chn = 1  # number of channels
        num_filters = 16  # number of filters per layer
        N = 128  # image size
        x, key = randn((10, N, N, chn), seed=1234)
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
        x, key = randn((10, N, N, chn), seed=1234)
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
        self.x, key = randn((10, N, N, chn), seed=1234)
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
    # 2D evaluation signal.
    fmap = sflax.FlaxMap(testobj.dncnn, testobj.variables)
    N = 128  # image size
    x, key = randn((N, N))
    out = fmap(x)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim


def test_FlaxMap_3D_call(testobj):
    # Test for the usage of flax model as a map.
    # 3D evaluation signal.
    fmap = sflax.FlaxMap(testobj.dncnn, testobj.variables)
    N = 128  # image size
    chn = 1  # channels
    x, key = randn((N, N, chn))
    out = fmap(x)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim


def test_FlaxMap_batch_call(testobj):
    # Test for the usage of flax model as a map.
    # 4D evaluation signal.
    fmap = sflax.FlaxMap(testobj.dncnn, testobj.variables)
    N = 128  # image size
    chn = 1  # channels
    batch = 8  # batch size
    x, key = randn((batch, N, N, chn))
    out = fmap(x)
    assert x.dtype == out.dtype
    assert x.ndim == out.ndim


def test_FlaxMap_blockarray_exception(testobj):

    from scico.numpy import BlockArray

    fmap = sflax.FlaxMap(testobj.dncnn, testobj.variables)

    x0, key = randn(shape=(3, 4), seed=4321)
    x1, key = randn(shape=(4, 5, 6), key=key)
    x = BlockArray((x0, x1))

    with pytest.raises(NotImplementedError):
        fmap(x)


@pytest.mark.parametrize("variant", ["6L", "6M", "6H", "17L", "17M", "17H"])
def test_variable_load(variant):
    N = 128  # image size
    chn = 1  # channels
    x, key = randn((10, N, N, chn), seed=1234)

    if variant[0] == "6":
        nlayer = 6
    else:
        nlayer = 17

    model = sflax.DnCNNNet(depth=nlayer, channels=chn, num_filters=64, dtype=np.float32)
    # Load weights for DnCNN.
    variables = sflax.load_weights(_flax_data_path("dncnn%s.npz" % variant))

    try:
        fmap = sflax.FlaxMap(model, variables)
        out = fmap(x)
    except Exception as e:
        print(e)
        assert 0


def test_variable_load_mismatch():
    N = 128  # image size
    chn = 1  # channels
    x, key = randn((10, N, N, chn), seed=1234)

    nlayer = 6
    model = sflax.ResNet(depth=nlayer, channels=chn, num_filters=64, dtype=np.float32)
    # Load weights for DnCNN.
    variables = sflax.load_weights(_flax_data_path("dncnn6L.npz"))

    # created with mismatched parameters
    fmap = sflax.FlaxMap(model, variables)
    with pytest.raises(ScopeParamShapeError):
        fmap(x)


def test_variable_save():
    N = 128  # image size
    chn = 1  # channels
    x, key = randn((10, N, N, chn), seed=1234)

    nlayer = 6
    model = sflax.ResNet(depth=nlayer, channels=chn, num_filters=64, dtype=np.float32)

    aux, key = randn((1,), seed=23432)
    input_shape = (1, N, N, chn)
    variables = model.init({"params": key}, np.ones(input_shape, model.dtype))

    try:
        temp_dir = tempfile.TemporaryDirectory()
        sflax.save_weights(unfreeze(variables), os.path.join(temp_dir.name, "vres6.npz"))
    except Exception as e:
        print(e)
        assert 0
