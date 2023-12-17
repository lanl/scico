# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionality to traverse, select, and update model parameters."""

from typing import Any

import jax.numpy as jnp

from flax.traverse_util import ModelParamTraversal

PyTree = Any


def construct_traversal(prmname: str) -> ModelParamTraversal:
    """Construct utility to select model parameters using a name filter.

    Args:
        prmname: Name of parameter to select.

    Returns:
        Flax utility to traverse and select model parameters.
    """
    return ModelParamTraversal(lambda path, _: prmname in path)


def clip_positive(params: PyTree, traversal: ModelParamTraversal, minval: float = 1e-4) -> PyTree:
    """Clip parameters to positive range.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        minval: Minimum value to clip selected model parameters and keep
            them in a positive range. Default: 1e-4.
    """
    params_out = traversal.update(lambda x: jnp.clip(x, a_min=minval), params)

    return params_out


def clip_range(
    params: PyTree, traversal: ModelParamTraversal, minval: float = 1e-4, maxval: float = 1
) -> PyTree:
    """Clip parameters to specified range.

    Args:
        params: Current model parameters.
        traversal: Utility to select model parameters.
        minval: Minimum value to clip selected model parameters.
            Default: 1e-4.
        maxval: Maximum value to clip selected model parameters.
            Default: 1.
    """
    params_out = traversal.update(lambda x: jnp.clip(x, a_min=minval, a_max=maxval), params)

    return params_out
