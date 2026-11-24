# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear algebra functions."""

from typing import TYPE_CHECKING, Any

import numpy as np

import jax.numpy as jnp

from . import _wrappers

_wrappers.add_attributes(
    to_dict=vars(),
    from_dict=jnp.linalg.__dict__,
)

# clean up
del np, jnp, _wrappers


if TYPE_CHECKING:
    # Most of the public names in this module are injected into its namespace
    # at import time by the _wrappers calls above, which static analysis cannot
    # follow. Without this declaration every reference to one of them is
    # reported as a missing module attribute. Defined only for type checkers, so
    # attribute errors at run time are unaffected.
    def __getattr__(name: str) -> Any: ...
