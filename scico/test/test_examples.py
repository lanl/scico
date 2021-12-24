import os
import tempfile

import numpy as np

import imageio

from scico.examples import (
    downsample_volume,
    epfl_deconv_data,
    tile_volume_slices,
    volume_read,
)

# These tests are for the scico.examples module, NOT the example scripts


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
