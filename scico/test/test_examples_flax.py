import pytest
import os

import numpy as np

from scico import random
from scico.typing import Shape
from scico.examples_flax import ( generate_foam2_images, distributed_data_generation,
    rotation90,
    flip,
    CenterCrop,
    PositionalCrop,
    RandomNoise,
)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# These tests are for the scico.examples_flax module, NOT the example scripts


def test_foam2_gen():
    seed = 4321
    N = 32
    ndata = 2

    dt = generate_foam2_images(seed, N, ndata)
    assert dt.shape == (ndata, N, N, 1)


def test_distdatagen():
    N = 32
    nimg = 16
    dt = distributed_data_generation(generate_foam2_images, N, nimg)
    assert dt.ndim == 5
    assert dt.shape[0] * dt.shape[1] == nimg
    assert dt.shape[2:] == (N, N, 1)


def test_distdatagen_flag():
    N = 32
    nimg = 16
    dt = distributed_data_generation(generate_foam2_images, N, nimg, False)
    assert dt.ndim == 4
    assert dt.shape == (nimg, N, N, 1)


def test_distdatagen_exception():
    N = 32
    nimg = 15
    with pytest.raises(ValueError):
        distributed_data_generation(generate_foam2_images, N, nimg)


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
    top, key = random.randint(shape=(1,), minval=0, maxval=N-128, key=key)
    left, key = random.randint(shape=(1,), minval=0, maxval=N-128, key=key)
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

    assert x.shape == x.shape


@pytest.mark.parametrize("shape", [(128, 128), (128, 128, 3), (5, 128, 128, 1)])
def test_random_noise2(shape):
    x, key = random.randn(shape, seed=4321)
    noise = RandomNoise(0.1, True)
    xn = noise(x)

    assert x.shape == x.shape
