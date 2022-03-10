import pytest
import os

from scico.examples_flax import generate_foam2_images, distributed_data_generation

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
