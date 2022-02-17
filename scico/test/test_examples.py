import os
import tempfile

import numpy as np

import imageio
import pytest

from scico.examples import (
    create_circular_phantom,
    create_cone,
    downsample_volume,
    epfl_deconv_data,
    rgb2gray,
    tile_volume_slices,
    volume_read,
)

# These tests are for the scico.examples module, NOT the example scripts


def test_rgb2gray():
    rgb = np.ones((31, 32, 3), dtype=np.float32)
    gry = rgb2gray(rgb)
    assert np.abs(gry.mean() - 1.0) < 1e-6


def test_volume_read():
    temp_dir = tempfile.TemporaryDirectory()
    v0 = np.zeros((32, 32), dtype=np.uint16)
    v1 = np.ones((32, 32), dtype=np.uint16)
    imageio.imwrite(os.path.join(temp_dir.name, "v0.tif"), v0)
    imageio.imwrite(os.path.join(temp_dir.name, "v1.tif"), v1)
    vol = volume_read(temp_dir.name, ext="tif")
    assert np.allclose(v0, vol[..., 0]) and np.allclose(v1, vol[..., 1])


def test_epfl_deconv_data():
    temp_dir = tempfile.TemporaryDirectory()
    y0 = np.zeros((32, 32), dtype=np.uint16)
    psf0 = np.ones((32, 32), dtype=np.uint16)
    np.savez(os.path.join(temp_dir.name, "epfl_big_deconv_0.npz"), y=y0, psf=psf0)
    y, psf = epfl_deconv_data(0, cache_path=temp_dir.name)
    assert np.allclose(y0, y) and np.allclose(psf0, psf)


def test_downsample_volume():
    v0 = np.zeros((32, 32, 16))
    v1 = downsample_volume(v0, rate=1)
    assert v0.shape == v1.shape
    v0 = np.zeros((32, 32, 16))
    v1 = downsample_volume(v0, rate=2)
    assert tuple([n // 2 for n in v0.shape]) == v1.shape
    v0 = np.zeros((32, 32, 16))
    v1 = downsample_volume(v0, rate=3)
    assert tuple([round(n / 3) for n in v0.shape]) == v1.shape


def test_tile_volume_slices():
    v = np.ones((16, 16, 16))
    tvs = tile_volume_slices(v)
    assert tvs.ndim == 2
    v = np.ones((16, 16, 16, 3))
    tvs = tile_volume_slices(v)
    assert tvs.ndim == 3 and tvs.shape[-1] == 3


def test_create_circular_phantom():
    img_shape = (32, 32)
    radius_list = [2, 4, 8]
    val_list = [2, 4, 8]
    x_gt = create_circular_phantom(img_shape, radius_list, val_list)

    assert x_gt.shape == img_shape
    assert np.max(x_gt) == max(val_list)
    assert np.min(x_gt) == 0


@pytest.mark.parametrize(
    "img_shape",
    (
        (3, 3),
        (50, 51),
        (3, 3, 3),
    ),
)
def test_create_cone(img_shape):
    x_gt = create_cone(img_shape)
    assert x_gt.shape == img_shape
    # check symmetry
    assert np.abs(x_gt[(0,) * len(img_shape)] - x_gt[(-1,) * len(img_shape)]) < 1e-6
