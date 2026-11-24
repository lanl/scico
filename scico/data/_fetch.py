# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functions for downloading and caching example data."""

import os
import tempfile
import zipfile
from typing import Optional, Tuple

import numpy as np

import imageio.v3 as iio

from scico.util import url_get
from scipy.io import loadmat

from ._examples import volume_read


def get_epfl_deconv_data(channel: int, path: str, verbose: bool = False):  # pragma: no cover
    """Download example data from EPFL Biomedical Imaging Group.

    Download deconvolution problem data from EPFL Biomedical Imaging
    Group. The downloaded data is converted to `.npz` format for
    convenient access via :func:`numpy.load`. The converted data is saved
    in a file `epfl_big_deconv_<channel>.npz` in the directory specified
    by `path`.

    Args:
        channel: Channel number between 0 and 2.
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """

    # data source URL and filenames
    data_base_url = "http://bigwww.epfl.ch/deconvolution/bio/"
    data_zip_files = ["CElegans-CY3.zip", "CElegans-DAPI.zip", "CElegans-FITC.zip"]
    psf_zip_files = ["PSF-" + data for data in data_zip_files]

    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory.")

    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    # download data and psf files for selected channel into temporary directory
    for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
        if verbose:
            print(f"Downloading {zip_file} from {data_base_url}")
        data = url_get(data_base_url + zip_file)
        f = open(os.path.join(temp_dir.name, zip_file), "wb")
        f.write(data.read())
        f.close()
        if verbose:
            print("Download complete")

    # unzip downloaded data into temporary directory
    for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
        if verbose:
            print(f"Extracting content from zip file {zip_file}")
        with zipfile.ZipFile(os.path.join(temp_dir.name, zip_file), "r") as zip_ref:
            zip_ref.extractall(temp_dir.name)

    # read unzipped data files into 3D arrays and save as .npz
    zip_file = data_zip_files[channel]
    y = volume_read(os.path.join(temp_dir.name, zip_file[:-4]))
    zip_file = psf_zip_files[channel]
    psf = volume_read(os.path.join(temp_dir.name, zip_file[:-4]))

    npz_file = os.path.join(path, f"epfl_big_deconv_{channel}.npz")
    if verbose:
        print(f"Saving as {npz_file}")
    np.savez(npz_file, y=y, psf=psf)


def epfl_deconv_data(
    channel: int, verbose: bool = False, cache_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get deconvolution problem data from EPFL Biomedical Imaging Group.

    If the data has previously been downloaded, it will be retrieved from
    a local cache.

    Args:
        channel: Channel number between 0 and 2.
        verbose: Flag indicating whether to print status messages.
        cache_path: Directory in which downloaded data is cached. The
           default is `~/.cache/scico/examples`, where `~` represents
           the user home directory.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (np.ndarray): Blurred channel data.
           - **psf** : (np.ndarray): Channel psf.
    """

    # set default cache path if not specified
    if cache_path is None:  # pragma: no cover
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples")

    # create cache directory and download data if not already present
    npz_file = os.path.join(cache_path, f"epfl_big_deconv_{channel}.npz")
    if not os.path.isfile(npz_file):  # pragma: no cover
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        get_epfl_deconv_data(channel, path=cache_path, verbose=verbose)

    # load data and return y and psf arrays converted to float32
    npz = np.load(npz_file)
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float32)
    return y, psf


def get_ucb_diffusercam_data(path: str, verbose: bool = False):  # pragma: no cover
    """Download data from UC Berkeley Waller Lab diffusercam project.

    Download deconvolution problem data from UC Berkeley Waller Lab
    diffusercam project.  The downloaded data is converted to `.npz`
    format for convenient access via :func:`numpy.load`.  The
    converted data is saved in a file `ucb_diffcam_data.npz.npz` in
    the directory specified by `path`.

    Args:
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """

    # data source URL, filenames, and request header
    data_base_url = "https://github.com/Waller-Lab/DiffuserCam/blob/master/example_data/"
    data_files = ["example_psfs.mat", "example_raw.png"]
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Referer": data_base_url}

    # ensure path directory exists
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist or is not a directory.")

    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    # download data files into temporary directory
    for data_file in data_files:
        if verbose:
            print(f"Downloading {data_file} from {data_base_url}")
        data = url_get(data_base_url + data_file + "?raw=true", headers=headers)
        f = open(os.path.join(temp_dir.name, data_file), "wb")
        f.write(data.read())
        f.close()
        if verbose:
            print("Download complete")

    # load data, normalize it, and save as npz
    y = iio.imread(os.path.join(temp_dir.name, "example_raw.png"))
    y = y.astype(np.float32)
    y -= 100.0
    y /= y.max()
    mat = loadmat(os.path.join(temp_dir.name, "example_psfs.mat"))
    psf = mat["psf"].astype(np.float64)
    psf -= 102.0
    psf /= np.linalg.norm(psf, axis=(0, 1)).min()

    # save as .npz
    npz_file = os.path.join(path, "ucb_diffcam_data.npz")
    if verbose:
        print(f"Saving as {npz_file}")
    np.savez(npz_file, y=y, psf=psf)


def ucb_diffusercam_data(
    verbose: bool = False, cache_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get example data from UC Berkeley Waller Lab diffusercam project.

    If the data has previously been downloaded, it will be retrieved from
    a local cache.

    Args:
        verbose: Flag indicating whether to print status messages.
        cache_path: Directory in which downloaded data is cached. The
           default is `~/.cache/scico/examples`, where `~` represents
           the user home directory.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (np.ndarray): Measured image
           - **psf** : (np.ndarray): Stack of psfs.
    """

    # set default cache path if not specified
    if cache_path is None:  # pragma: no cover
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples")

    # create cache directory and download data if not already present
    npz_file = os.path.join(cache_path, "ucb_diffcam_data.npz")
    if not os.path.isfile(npz_file):  # pragma: no cover
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        get_ucb_diffusercam_data(path=cache_path, verbose=verbose)

    # load data and return y and psf arrays converted to float32
    npz = np.load(npz_file)
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float64)
    return y, psf
