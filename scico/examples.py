# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions used by example scripts."""


import glob
import os
import tempfile
import zipfile
from typing import List, Optional

import numpy as np

import imageio

import scico.numpy as snp
from scico import util
from scico.typing import Array, JaxArray, Shape
from scipy.ndimage import zoom


def rgb2gray(rgb: JaxArray) -> JaxArray:
    """Convert an RGB image (or images) to grayscale.

    Args:
        rgb: RGB image as Nr x Nc x 3 or Nr x Nc x 3 x K array.

    Returns:
        Grayscale image as Nr x Nc or Nr x Nc x K array.
    """

    w = snp.array([0.299, 0.587, 0.114], dtype=rgb.dtype)[np.newaxis, np.newaxis]
    return snp.sum(w * rgb, axis=2)


def volume_read(path: str, ext: str = "tif") -> Array:
    """Read a 3D volume from a set of files in the specified directory.

    All files with extension `ext` (i.e. matching glob `*.ext`)
    in directory `path` are assumed to be image files and are read.
    The filenames are assumed to be such that their alphanumeric
    ordering corresponds to their order as volume slices.

    Args:
        path: Path to directory containing the image files.
        ext: Filename extension.

    Returns:
        Volume as a 3D array.
    """

    slices = []
    for file in sorted(glob.glob(os.path.join(path, "*." + ext))):
        image = imageio.imread(file)
        slices.append(image)
    return np.dstack(slices)


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
        raise ValueError(f"Path {path} does not exist or is not a directory")

    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    # download data and psf files for selected channel into temporary directory
    for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
        if verbose:
            print(f"Downloading {zip_file} from {data_base_url}")
        data = util.url_get(data_base_url + zip_file)
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


def epfl_deconv_data(channel: int, verbose: bool = False, cache_path: str = None) -> Array:
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

           - **y** : (DeviceArray): Blurred channel data.
           - **psf** : (DeviceArray): Channel psf.
    """

    # set default cache path if not specified
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "examples")

    # create cache directory and download data if not already present
    npz_file = os.path.join(cache_path, f"epfl_big_deconv_{channel}.npz")
    if not os.path.isfile(npz_file):
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        get_epfl_deconv_data(channel, path=cache_path, verbose=verbose)

    # load data and return y and psf arrays converted to float32
    npz = np.load(npz_file)
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float32)
    return y, psf


def downsample_volume(vol: Array, rate: int) -> Array:
    """Downsample a 3D array.

    Downsample a 3D array. If the volume dimensions can be divided by
    `rate`, this is achieved via averaging distinct `rate` x `rate` x
    `rate` block in `vol`. Otherwise it is achieved via a call to
    :func:`scipy.ndimage.zoom`.

    Args:
        vol: Input volume.
        rate: Downsampling rate.

    Returns:
        Downsampled volume.
    """

    if rate == 1:
        return vol

    if np.all([n % rate == 0 for n in vol.shape]):
        vol = snp.mean(snp.reshape(vol, (-1, rate, vol.shape[1], vol.shape[2])), axis=1)
        vol = snp.mean(snp.reshape(vol, (vol.shape[0], -1, rate, vol.shape[2])), axis=2)
        vol = snp.mean(snp.reshape(vol, (vol.shape[0], vol.shape[1], -1, rate)), axis=3)
    else:
        vol = zoom(vol, 1.0 / rate)

    return vol


def tile_volume_slices(x: Array, sep_width: int = 10) -> Array:
    """Make an image with tiled slices from an input volume.

    Make an image with tiled `xy`, `xz`, and `yz` slices from an input
    volume.

    Args:
        x: Input volume consisting of a 3D or 4D array. If the input is
           4D, the final axis represents a channel index.
        sep_width: Number of pixels separating the slices in the output
           image.

    Returns:
        Image containing tiled slices.
    """

    if x.ndim == 3:
        fshape = (x.shape[0], sep_width)
    else:
        fshape = (x.shape[0], sep_width, 3)
    out = snp.concatenate(
        (
            x[:, :, x.shape[2] // 2],
            snp.full(fshape, snp.nan),
            x[:, x.shape[1] // 2, :],
        ),
        axis=1,
    )

    if x.ndim == 3:
        fshape0 = (sep_width, out.shape[1])
        fshape1 = (x.shape[2], x.shape[2] + sep_width)
        trans = (1, 0)

    else:
        fshape0 = (sep_width, out.shape[1], 3)
        fshape1 = (x.shape[2], x.shape[2] + sep_width, 3)
        trans = (1, 0, 2)
    out = snp.concatenate(
        (
            out,
            snp.full(fshape0, snp.nan),
            snp.concatenate(
                (
                    x[x.shape[0] // 2, :, :].transpose(trans),
                    snp.full(fshape1, snp.nan),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )

    out = snp.where(snp.isnan(out), snp.nanmax(out), out)

    return out


def create_cone(img_shape: Shape, center: Optional[List[float]] = None) -> Array:
    """Compute a 2D map of the distance from a center pixel.

    Args:
        img_shape: Shape of the image for which the distance map is being
            computed.
        center: Tuple of center pixel coordinates. If ``None``, this is
            set to the center of the image.

    Returns:
        An image containing a 2D map of the distances.
    """

    if center is None:
        center = [(img_dim - 1) / 2 for img_dim in img_shape]

    coords = [snp.arange(0, img_dim) for img_dim in img_shape]
    coord_mesh = snp.meshgrid(*coords, sparse=True, indexing="ij")

    dist_map = sum([(coord_mesh[i] - center[i]) ** 2 for i in range(len(coord_mesh))])
    dist_map = snp.sqrt(dist_map)

    return dist_map


def create_circular_phantom(
    img_shape: Shape, radius_list: list, val_list: list, center: Optional[list] = None
) -> Array:
    """Construct a circular phantom with given radii and intensities.

    Args:
        img_shape: Shape of the phantom to be created.
        radius_list: List of radii of the rings in the phantom.
        val_list: List of intensity values of the rings in the phantom.
        center: Tuple of center pixel coordinates. If ``None``, this is
           set to the center of the image.

    Returns:
        The computed circular phantom.
    """

    dist_map = create_cone(img_shape, center)

    img = snp.zeros(img_shape)
    for r, val in zip(radius_list, val_list):
        # img[dist_map < r] = val
        img = img.at[dist_map < r].set(val)

    return img
