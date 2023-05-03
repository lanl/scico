# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Definition of typed dictionaries for training data."""

import sys
from typing import Optional, Union

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from scico.numpy import Array
from scico.typing import Shape


class CTDataSetDict(TypedDict):
    """Definition of the structure to store generated CT data."""

    img: Array  # original image
    sino: Array  # sinogram
    fbp: Array  # filtered back projection


class ConfigImageSetDict(TypedDict):
    """Definition of the configuration for image data preprocessing."""

    output_size: Union[int, Shape]
    stride: Optional[Union[Shape, int]]
    multi: bool
    augment: bool
    run_gray: bool
    num_img: int
    test_num_img: int
    data_mode: str
    noise_level: float
    noise_range: bool
    test_split: float
    seed: float
