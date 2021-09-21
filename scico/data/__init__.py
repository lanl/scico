# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Data files for usage examples."""

import os.path

from jax.interpreters.xla import DeviceArray

from imageio import imread

import scico.numpy as snp

__all__ = ["kodim23"]


def _imread(filename: str, path: str = None, asfloat: bool = False) -> DeviceArray:
    """Read an image from disk.

    Args:
        str: Base filename (i.e. without path) of image file
        path: Path to directory containing the image file
        asfloat: Flag indicating whether the returned image should be
          converted to float32 dtype with a range [0, 1]

    Returns:
       DeviceArray: image data array
    """

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "examples")
    im = imread(os.path.join(path, filename))
    if asfloat:
        im = im.astype(snp.float32) / 255.0
    return im


def kodim23(asfloat: bool = False) -> DeviceArray:
    """Return the `kodim23` test image.

    Args:
        asfloat: Flag indicating whether the returned image should be
          converted to float32 dtype with a range [0, 1]

    Returns:
       DeviceArray: image data array
    """

    return _imread("kodim23.png", asfloat=asfloat)


def _objax_data_path(filename: str) -> str:
    """Get the full filename of an objax data file.

    Args:
        str: Base filename (i.e. without path) of data file

    Returns:
       str: Full filename, with path, of data file
    """

    return os.path.join(os.path.dirname(__file__), "objax", filename)
