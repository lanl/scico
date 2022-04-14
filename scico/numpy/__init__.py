# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

""":class:`scico.numpy.BlockArray`-compatible
versions of :mod:`jax.numpy` functions.

This modules consists of functions from :mod:`jax.numpy` wrapped to
support compatibility with :class:`scico.numpy.BlockArray`. This
module is a work in progress and therefore not all functions are
wrapped. Functions that have not been wrapped yet have WARNING text in
their documentation, below.
"""
import numpy as np

import jax.numpy as jnp

from . import _util
from .blockarray import BlockArray
from .function_lists import *
from .util import *

# copy most of jnp without wrapping
_util.add_attributes(
    to_dict=vars(),
    from_dict=jnp.__dict__,
    modules_to_recurse=("linalg", "fft"),
)

# wrap jnp funcs
_util.wrap_recursively(vars(), creation_routines, _util.map_func_over_tuple_of_tuples)
_util.wrap_recursively(
    vars(), mathematical_functions + reduction_functions, _util.map_func_over_blocks
)
_util.wrap_recursively(vars(), reduction_functions, _util.add_full_reduction)

# copy np.testing
_util.add_attributes(
    to_dict=vars(),
    from_dict={"testing": np.testing},
    modules_to_recurse=("testing",),
)

# wrap testing funcs
_util.wrap_recursively(vars(), testing_functions, _util.map_func_over_blocks)

# clean up
del np, jnp, _util
