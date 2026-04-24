# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Test support functions."""

from functools import partial

import numpy

from . import _wrappers
from ._wrapped_function_lists import TESTING_FUNCTIONS

# copy most of np testing functions
_wrappers.add_attributes(
    to_dict=vars(),
    from_dict=numpy.testing.__dict__,
)

# wrap testing funcs
_wrappers.wrap_recursively(
    vars(), TESTING_FUNCTIONS, partial(_wrappers.map_func_over_args, is_void=True)
)
