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
from .util import *

# wrap jnp
_util.wrap_attributes(
    to_dict=vars(),
    from_dict=jnp.__dict__,
    modules_to_recurse=("linalg", "fft"),
    reductions=("sum", "norm"),
    no_wrap=(
        "dtype",
        "broadcast_shapes",  # nested tuples as normal input (*shapes)
        "array",  # no meaning mapped over blocks
        "stack",  # no meaning mapped over blocks
        "concatenate",  # no meaning mapped over blocks
        "pad",
    ),  # nested tuples as normal input
)

# wrap np.testing
_util.wrap_attributes(
    to_dict=vars(),
    from_dict={k: v for k, v in np.__dict__.items() if k == "testing"},
    modules_to_recurse=("testing"),
)


__all__ = ["BlockArray"]

# clean up
del np, jnp, _util
