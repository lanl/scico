import os
import tempfile

import numpy as np

from scico.data import epfl_deconv_data, ucb_diffusercam_data


def test_epfl_deconv_data():
    temp_dir = tempfile.TemporaryDirectory()
    y0 = np.zeros((32, 32), dtype=np.uint16)
    psf0 = np.ones((32, 32), dtype=np.uint16)
    np.savez(os.path.join(temp_dir.name, "epfl_big_deconv_0.npz"), y=y0, psf=psf0)
    y, psf = epfl_deconv_data(0, cache_path=temp_dir.name)
    assert np.allclose(y0, y) and np.allclose(psf0, psf)


def test_ucb_diffusercam_data():
    temp_dir = tempfile.TemporaryDirectory()
    y0 = np.zeros((32, 32), dtype=np.uint16)
    psf0 = np.ones((8, 32, 32), dtype=np.uint16)
    np.savez(os.path.join(temp_dir.name, "ucb_diffcam_data.npz"), y=y0, psf=psf0)
    y, psf = ucb_diffusercam_data(cache_path=temp_dir.name)
    assert np.allclose(y0, y) and np.allclose(psf0, psf)
