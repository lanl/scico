# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utility functions used by example scripts."""


import glob
import os
import tempfile
import zipfile

import imageio

import scico.numpy as snp
from scico.typing import JaxArray

__author__ = """\n""".join(
    ["Brendt Wohlberg <brendt@ieee.org>", "Michael McCann <mccann@lanl.gov>"]
)


def volume_read(path: str, ext: str = "tif") -> JaxArray:
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


def epfl_deconv_data(channel: int, cache_path: str = None) -> JaxArray:
    """Get deconvolution problem data from EPFL Biomedical Imaging Group.

    If the data has previously been downloaded, it will be retrieved from
    a local cache.

    Args:
        channel: Channel number between 0 and 2.
        cache_path: Directory in which downloaded data is cached. The
           default is `~/.cache/scico/epfl_big`, where `~` represents
           the user home directory.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (DeviceArray): Blurred channel data.
           - **psf** : (DeviceArray): Channel psf.
    """

    # data source URL and filenames
    data_base_url = "http://bigwww.epfl.ch/deconvolution/bio/"
    data_zip_files = ["CElegans-CY3.zip", "CElegans-DAPI.zip", "CElegans-FITC.zip"]
    psf_zip_files = ["PSF-" + data for data in data_zip_files]
    # set default cache path if not specified
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "scico", "epfl_big")

    # if cache path exists, data is assumed to aleady be downloaded
    if not os.path.isdir(os.path.join(cache_path, data_zip_files[channel][:-4])):
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        # temporary directory for downloads
        temp_dir = tempfile.TemporaryDirectory()
        # download data and psf files for selected channel into temporary directory
        for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
            data = util.url_get(data_base_url + zip_file)
            f = open(os.path.join(temp_dir.name, zip_file), "wb")
            f.write(data.read())
            f.close()
        # unzip downloaded data into cache path
        for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
            with zipfile.ZipFile(os.path.join(temp_dir.name, zip_file), "r") as zip_ref:
                zip_ref.extractall(cache_path)

    # read unzipped data files into 3D arrays
    zip_file = data_zip_files[channel]
    y = volume_read(os.path.join(cache_path, zip_file[:-4])).astype(np.float32)
    zip_file = psf_zip_files[channel]
    psf = volume_read(os.path.join(cache_path, zip_file[:-4])).astype(np.float32)
    return y, psf


def block_avg(im: JaxArray, N: int) -> JaxArray:
    """Average distinct NxNxN blocks of an image.

    Average distinct NxNxN blocks of input `im`, return the resulting
    smaller image.

    Args:
        im: Input image.
        N: Size of cube over which to average.

    Returns:
        Averaged and downsampled image.
    """

    im = snp.mean(snp.reshape(im, (-1, N, im.shape[1], im.shape[2])), axis=1)
    im = snp.mean(snp.reshape(im, (im.shape[0], -1, N, im.shape[2])), axis=2)
    im = snp.mean(snp.reshape(im, (im.shape[0], im.shape[1], -1, N)), axis=3)

    return im


def make_slices(x: JaxArray, sep_width: int = 10, fill_val: float = -1.0):
    """Make an image with x-y, x-z, and y-z slices from an input volume.

    Args:
        x: Input volume consisting of a 3D or 4D array. If the input is
           4D, the final axis represents a channel index.
        sep_width: Number of pixels separating the slices in the output
           image.
        fill_val: A pixel value used to during construction of the output
           image from slices of the input volume. Should be chosen as a
           value that does not occur in input `x`.
    """

    if x.ndim == 3:
        fshape = (x.shape[0], sep_width)
    else:
        fshape = (x.shape[0], sep_width, 3)
    out = snp.concatenate(
        (
            x[:, :, x.shape[2] // 2],
            snp.full(fshape, fill_val),
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
            snp.full(fshape0, fill_val),
            snp.concatenate(
                (
                    x[x.shape[0] // 2, :, :].transpose(trans),
                    snp.full(fshape1, fill_val),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )

    out = snp.where(out == fill_val, out.max(), out)

    return out
