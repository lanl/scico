import numpy as np

import pytest

import scico.numpy as snp
from scico.examples import (
    create_3d_foam_phantom,
    create_block_phantom,
    create_circular_phantom,
    create_cone,
    create_conv_sparse_phantom,
    create_laminar_phantom,
    create_tangle_phantom,
    downsample_volume,
    gaussian,
    phase_diff,
    rgb2gray,
    spnoise,
    tile_volume_slices,
)

# These tests are for the scico.examples module, NOT the example scripts


def test_rgb2gray():
    rgb = np.ones((31, 32, 3), dtype=np.float32)
    gry = rgb2gray(rgb)
    assert np.abs(gry.mean() - 1.0) < 1e-6


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


def test_gaussian():
    g0 = gaussian((5, 5))
    assert g0.shape == (5, 5)
    g1 = gaussian((5, 5), sigma=np.array([[3, 0], [0, 2]]))
    assert np.sum(g1 / g1.max()) > np.sum(g0 / g0.max())
    with pytest.raises(ValueError):
        g2 = gaussian((5, 5), sigma=np.array([[2, 2], [2, 2]]))


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


@pytest.mark.parametrize(
    "img_shape",
    (
        (3, 3, 3),
        (20, 21, 22),
        (15, 15, 5),
    ),
)
@pytest.mark.parametrize("N_sphere", (3, 10, 20))
def test_create_3d_foam_phantom(img_shape, N_sphere):
    x_gt = create_3d_foam_phantom(img_shape, N_sphere)
    assert x_gt.shape == img_shape


def test_conv_sparse_phantom():
    h, x = create_conv_sparse_phantom(64, 32)
    assert h.shape == (3, 15, 15)
    assert x.shape == (3, 64, 64)
    assert np.sum(x > 0) == 32


def test_tangle_phantom():
    v = create_tangle_phantom(3, 4, 5)
    assert v.shape == (5, 4, 3)


def test_create_block_phantom():
    x = create_block_phantom((16, 15, 14))
    assert x.shape == (16, 15, 14)
    assert x.dtype == np.float32


def test_create_laminar_phantom():
    x = create_laminar_phantom()
    assert x.shape == (64, 256, 256)
    assert x.dtype == np.float32


def test_spnoise():
    x = 0.5 * np.ones((10, 11))
    y = spnoise(x, 0.5, nmin=0.01, nmax=0.99)
    assert np.all(y >= 0.01)
    assert np.all(y <= 0.99)
    x = 0.5 * snp.ones((10, 11))
    y = spnoise(x, 0.5, nmin=0.01, nmax=0.99)
    assert np.all(y >= 0.01)
    assert np.all(y <= 0.99)


def test_phase_diff():
    x = np.pi * np.random.randn(16)
    y = np.pi * np.random.randn(16)
    d = phase_diff(x, y)
    assert np.all(d >= 0)
    assert np.all(d <= np.pi)
