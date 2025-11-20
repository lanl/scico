import os
import tempfile

import numpy as np

import pytest

from scico import random
from scico.flax.examples.data_generation import (
    distributed_data_generation,
    generate_blur_data,
    generate_ct_data,
    generate_foam1_images,
    generate_foam2_images,
    have_ray,
    have_xdesign,
)
from scico.flax.examples.data_preprocessing import (
    CenterCrop,
    PaddedCircularConvolve,
    PositionalCrop,
    RandomNoise,
    build_image_dataset,
    flip,
    preprocess_images,
    rotation90,
)
from scico.flax.examples.examples import (
    get_cache_path,
    runtime_error_array,
    runtime_error_scalar,
)
from scico.flax.examples.typed_dict import ConfigImageSetDict
from scico.typing import Shape

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# These tests are for the scico.flax.examples module, NOT the example scripts


@pytest.mark.skipif(not have_xdesign, reason="xdesign package not installed")
def test_foam1_gen():
    seed = 4444
    N = 32
    ndata = 2

    dt = generate_foam1_images(seed, N, ndata)
    assert dt.shape == (ndata, N, N, 1)


@pytest.mark.skipif(not have_xdesign, reason="xdesign package not installed")
def test_foam2_gen():
    seed = 4321
    N = 32
    ndata = 2

    dt = generate_foam2_images(seed, N, ndata)
    assert dt.shape == (ndata, N, N, 1)


@pytest.mark.skipif(not have_ray, reason="ray package not installed")
def test_distdatagen():
    N = 16
    nimg = 8

    def random_data_gen(seed, N, ndata):
        np.random.seed(seed)
        dt = np.random.randn(ndata, N, N, 1)
        return dt

    dt = distributed_data_generation(random_data_gen, N, nimg)
    assert dt.ndim == 4
    assert dt.shape == (nimg, N, N, 1)


@pytest.mark.skipif(
    not have_ray or not have_xdesign,
    reason="ray or xdesign package not installed",
)
def test_ct_data_generation():
    N = 32
    nimg = 8
    nproj = 45

    def random_img_gen(seed, size, ndata):
        np.random.seed(seed)
        shape = (ndata, size, size, 1)
        return np.random.randn(*shape)

    img, sino, fbp = generate_ct_data(nimg, N, nproj, imgfunc=random_img_gen)
    assert img.shape == (nimg, N, N, 1)
    assert sino.shape == (nimg, nproj, sino.shape[2], 1)
    assert fbp.shape == (nimg, N, N, 1)


@pytest.mark.skipif(not have_ray or not have_xdesign, reason="ray or xdesign package not installed")
def test_blur_data_generation():
    N = 32
    nimg = 8
    n = 3  # convolution kernel size
    blur_kernel = np.ones((n, n)) / (n * n)

    def random_img_gen(seed, size, ndata):
        np.random.seed(seed)
        shape = (ndata, size, size, 1)
        return np.random.randn(*shape)

    img, blurn = generate_blur_data(nimg, N, blur_kernel, noise_sigma=0.01, imgfunc=random_img_gen)
    assert img.shape == (nimg, N, N, 1)
    assert blurn.shape == (nimg, N, N, 1)


def test_rotation90():
    N = 128
    x, key = random.randn((N, N), seed=4321)
    x2, key = random.randn((10, N, N, 1), key=key)
    x_rot = rotation90(x)
    x2_rot = rotation90(x2)

    np.testing.assert_allclose(x_rot, np.swapaxes(x, 0, 1), rtol=1e-5)
    np.testing.assert_allclose(x2_rot, np.swapaxes(x2, 1, 2), rtol=1e-5)


def test_flip():
    N = 128
    x, key = random.randn((N, N), seed=4321)
    x2, key = random.randn((10, N, N, 1), key=key)
    x_flip = flip(x)
    x2_flip = flip(x2)

    np.testing.assert_allclose(x_flip, x[:, ::-1, ...], rtol=1e-5)
    np.testing.assert_allclose(x2_flip, x2[..., ::-1, :], rtol=1e-5)


@pytest.mark.parametrize("output_size", [128, (128, 128), (128, 64)])
def test_center_crop(output_size):
    N = 256
    x, key = random.randn((N, N), seed=4321)
    if isinstance(output_size, int):
        ccrop = CenterCrop(output_size)
    else:
        shp: Shape = output_size
        ccrop = CenterCrop(shp)

    x_crop = ccrop(x)
    if isinstance(output_size, int):
        assert x_crop.shape[0] == output_size
        assert x_crop.shape[1] == output_size
    else:
        assert x_crop.shape == output_size


@pytest.mark.parametrize("output_size", [128, (128, 128), (128, 64)])
def test_positional_crop(output_size):
    N = 256
    x, key = random.randn((N, N), seed=4321)
    top, key = random.randint(shape=(1,), minval=0, maxval=N - 128, key=key)
    left, key = random.randint(shape=(1,), minval=0, maxval=N - 128, key=key)
    pcrop = PositionalCrop(output_size)

    x_crop = pcrop(x, top[0], left[0])
    if isinstance(output_size, int):
        assert x_crop.shape[0] == output_size
        assert x_crop.shape[1] == output_size
    else:
        assert x_crop.shape == output_size


@pytest.mark.parametrize("range_flag", [False, True])
def test_random_noise1(range_flag):
    N = 128
    x, key = random.randn((N, N), seed=4321)
    noise = RandomNoise(0.1, range_flag)
    xn = noise(x)
    x2, key = random.randn((10, N, N, 1), key=key)
    xn2 = noise(x2)

    assert x.shape == xn.shape
    assert x2.shape == xn2.shape


@pytest.mark.parametrize("shape", [(128, 128), (128, 128, 3), (5, 128, 128, 1)])
def test_random_noise2(shape):
    x, key = random.randn(shape, seed=4321)
    noise = RandomNoise(0.1, True)
    xn = noise(x)

    assert x.shape == xn.shape


@pytest.mark.parametrize("output_size", [64, (64, 64)])
@pytest.mark.parametrize("gray_flag", [False, True])
@pytest.mark.parametrize("num_img_req", [None, 4])
def test_preprocess_images(output_size, gray_flag, num_img_req):

    num_img = 10
    N = 128
    C = 3
    shape = (num_img, N, N, C)
    images, key = random.randn(shape, seed=4444)

    stride = 1
    try:
        output = preprocess_images(
            images, output_size, gray_flag, num_img_req, multi_flag=False, stride=stride
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert output.shape[1] == 64
        assert output.shape[2] == 64

        if gray_flag:
            assert output.shape[-1] == 1
        else:
            assert output.shape[-1] == C

        if num_img_req is None:
            assert output.shape[0] == num_img
        else:
            assert output.shape[0] == num_img_req


def test_preprocess_images_multi_flag():
    num_img = 10
    N = 128
    C = 3
    shape = (num_img, N, N, C)
    images, key = random.randn(shape, seed=4444)

    output_size = (64, 64)
    gray_flag = True
    num_img_req = 4

    stride = 64  # 2 per side = 4 patches per image
    try:
        output = preprocess_images(
            images, output_size, gray_flag, num_img_req, multi_flag=True, stride=stride
        )
    except Exception as e:
        print(e)
        assert 0
    else:
        assert output.shape[0] == (4 * num_img_req)
        assert output.shape[1] == 64
        assert output.shape[2] == 64
        assert output.shape[-1] == 1


class SetupTest:
    def __init__(self):
        # Data configuration
        self.dtconf: ConfigImageSetDict = {
            "seed": 0,
            "output_size": 64,
            "stride": 1,
            "multi": False,
            "augment": False,
            "run_gray": True,
            "num_img": 10,
            "test_num_img": 4,
            "data_mode": "dn",
            "noise_level": 0.01,
            "noise_range": False,
            "test_split": 0.1,
        }


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


@pytest.mark.parametrize("augment", [False, True])
def test_build_image_dataset(testobj, augment):
    num_train = testobj.dtconf["num_img"]
    num_test = testobj.dtconf["test_num_img"]
    N = 128
    C = 3
    shape = (num_train, N, N, C)
    img_train, key = random.randn(shape, seed=4444)
    img_test, key = random.randn((num_test, N, N, C), key=key)

    dtconf = dict(testobj.dtconf)
    dtconf["augment"] = augment

    train_ds, test_ds = build_image_dataset(img_train, img_test, dtconf)
    assert train_ds["image"].shape == train_ds["label"].shape
    assert test_ds["image"].shape == test_ds["label"].shape
    assert test_ds["label"].shape[0] == num_test
    if augment:
        assert train_ds["label"].shape[0] == num_train * 3
    else:
        assert train_ds["label"].shape[0] == num_train


def test_padded_circular_convolve():
    N = 64
    C = 3
    kernel_size = 5
    blur_sigma = 2.1

    x, key = random.randn((N, N, C), seed=2468)

    pcc_op = PaddedCircularConvolve(N, C, kernel_size, blur_sigma)
    xblur = pcc_op(x)
    assert xblur.shape == x.shape


def test_runtime_error_scalar():
    with pytest.raises(RuntimeError):
        runtime_error_scalar("channels", "testing ", 3, 1)


def test_runtime_error_array():
    with pytest.raises(RuntimeError):
        runtime_error_array("channels", "testing ", 1e-2)


def test_default_cache_path():
    try:
        cache_path, cache_path_display = get_cache_path()
    except Exception as e:
        print(e)
        assert 0
    else:
        cache_path_display == "~/.cache/scico/examples/data"


def test_cache_path():
    try:
        temp_dir = tempfile.TemporaryDirectory()
        cache_path = os.path.join(temp_dir.name, ".cache")
        cache_path_, cache_path_display = get_cache_path(cache_path)
    except Exception as e:
        print(e)
        assert 0
    else:
        cache_path_ == cache_path
        cache_path_display == cache_path
