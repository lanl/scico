# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Data files for usage examples."""

import os.path
from typing import Optional

import numpy as np

from imageio.v3 import imread

__all__ = ["kodim23", "foam_phantom"]


def _imread(filename: str, path: Optional[str] = None, asfloat: bool = False) -> np.ndarray:
    """Read an image from disk.

    Args:
        filename: Base filename (i.e. without path) of image file.
        path: Path to directory containing the image file.
        asfloat: Flag indicating whether the returned image should be
          converted to :attr:`~numpy.float32` dtype with a range [0, 1].

    Returns:
       Image data array.
    """

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "examples")
    im = imread(os.path.join(path, filename))
    if asfloat:
        im = im.astype(np.float32) / 255.0
    return im


def kodim23(asfloat: bool = False) -> np.ndarray:
    """Return the `kodim23` test image.

    Args:
        asfloat: Flag indicating whether the returned image should be
          converted to :attr:`~numpy.float32` dtype with a range [0, 1].

    Returns:
       Image data array.
    """

    return _imread("kodim23.png", asfloat=asfloat)


def _npzread(filename: str, array: str, path: Optional[str] = None) -> np.ndarray:
    """Read an array from an npz file.

    Args:
        filename: Base filename (i.e. without path) of npz file.
        array: Name of the array in the npz file.
        path: Path to directory containing the npz file.

    Returns:
       Named array from npz file.
    """

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "examples")
    npz = np.load(os.path.join(path, filename))
    return npz[array]


def foam_phantom(size: int = 512, asfloat: bool = True):
    """Return a 3D foam phantom.

    Args:
        size: Size of cubic volume. Default value is `512`, selecting the
           512 x 512 x 512 voxel version. Other valid options are `1024`
           and `2048`.
        asfloat: Flag indicating whether the returned image should be
          converted to :attr:`~numpy.float32` dtype with a range [0, 1]
          from the original :attr:`~numpy.uint8` dtype with a range
          [0, 255].

    Returns:
        Foam phantom as 3D array.
    """
    vol = _npzread(f"foam_3d_{size:04d}.npz", "volume")
    if asfloat:
        return vol.astype(np.float32) / 255.0
    else:
        return vol


def _flax_data_path(filename: str) -> str:
    """Get the full filename of a flax data file.

    Args:
        filename: Base filename (i.e. without path) of data file.

    Returns:
       Full filename, with path, of data file.
    """

    return os.path.join(os.path.dirname(__file__), "flax", filename)
