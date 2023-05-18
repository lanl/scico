# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Test support functions."""

import numpy as np

from . import _wrappers

_wrappers.add_attributes(
    to_dict=vars(),
    from_dict=np.testing.__dict__,
)

# clean up
del np, _wrappers
