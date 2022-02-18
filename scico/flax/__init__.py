# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Neural network models implemented in Flax and functionality for using inside scico."""

import sys

# isort: off
from ._flax import FlaxMap, load_weights

__all__ = [
    "FlaxMap",
    "load_weights",
]

# Imported items in __all__ appear to originate in top-level flax module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
