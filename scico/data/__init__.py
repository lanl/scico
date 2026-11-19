# -*- coding: utf-8 -*-
# Copyright (C) 2021-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Data management functions."""

import sys

from ._blocks import get_image_blocks, select_image_files, split_list
from ._examples import foam_phantom, kodim23

__all__ = ["kodim23", "foam_phantom", "split_list", "select_image_files", "get_image_blocks"]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
