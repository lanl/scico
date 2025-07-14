# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

r""":class:`.BlockArray` and compatible functions.

This module consists of :class:`.BlockArray` and functions that support
both instances of this class and jax arrays. This includes all the
functions from :mod:`jax.numpy` and :mod:`numpy.testing`, where many have
been extended to automatically map over block array blocks as described
in :ref:`numpy_functions_blockarray`. Also included are additional
functions unique to SCICO in :mod:`.util`.
"""

import sys

import numpy as np

import jax.numpy as jnp
from jax import Array

from . import _wrappers, fft, linalg, testing, util
from ._blockarray import BlockArray
from ._wrapped_function_lists import *

# allow snp.blockarray(...) to create BlockArrays
blockarray = BlockArray.blockarray
blockarray.__module__ = __name__  # so that blockarray can be referenced in docs

# BlockArray appears to originate in this module
sys.modules[__name__].BlockArray.__module__ = __name__

# copy most of jnp without wrapping
_wrappers.add_attributes(to_dict=vars(), from_dict=jnp.__dict__)

# wrap jnp funcs
_wrappers.wrap_recursively(vars(), creation_routines, _wrappers.map_func_over_tuple_of_tuples)
_wrappers.wrap_recursively(vars(), mathematical_functions, _wrappers.map_func_over_blocks)
_wrappers.wrap_recursively(vars(), reduction_functions, _wrappers.add_full_reduction)

# wrap testing funcs
_wrappers.wrap_recursively(vars(), testing_functions, _wrappers.map_void_func_over_blocks)

# clean up
del np, jnp, _wrappers
