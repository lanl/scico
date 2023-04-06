# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Functionality to traverse, select and update model parameters."""

from typing import Any, Callable

import jax.numpy as jnp

from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import (
    _EmptyNode,
    _get_params_dict,
    _sorted_items,
    flatten_dict,
    unflatten_dict,
)

PyTree = Any

empty_node = _EmptyNode()

# From https://flax.readthedocs.io/en/latest/_modules/flax/traverse_util.html#Traversal
# This is marked as deprecated in Flax since v0.4.1. Copied here to keep needed functionality
class ModelParamTraversal:
    """Select model parameters using a name filter.

    This traversal operates on a nested dictionary of parameters and
    selects a subset based on the `filter_fn` argument.

    """

    def __init__(self, filter_fn: Callable):
        """Constructor a new ModelParamTraversal.

        Args:
          filter_fn: a function that takes a parameter's full name and
            its value and returns whether this parameter should be
            selected or not. The name of a parameter is determined by the
            module hierarchy and the parameter name (for example:
            '/module/sub_module/parameter_name').
        """
        self._filter_fn = filter_fn

    def iterate(self, inputs: PyTree):
        """Iterate over the values selected by this traversal.

        Args:
            inputs: the object that should be traversed.

        Returns:
            An iterator over the traversed values.
        """
        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params)
        for key, value in _sorted_items(flat_dict):
            path = "/" + "/".join(key)
            if self._filter_fn(path, value):
                yield value

    def update(self, fn: Callable, inputs: PyTree) -> PyTree:
        """Update the focused items.

        Args:
            fn: the callback function that maps each traversed item to
                its updated value.
            inputs: the object that should be traversed.

        Returns:
            A new object with the updated values.
        """
        params = _get_params_dict(inputs)
        flat_dict = flatten_dict(params, keep_empty_nodes=True)
        new_dict = {}
        for key, value in _sorted_items(flat_dict):
            # empty_node is not an actual leave. It's just a stub for empty nodes
            # in the nested dict.
            if value is not empty_node:
                path = "/" + "/".join(key)
                if self._filter_fn(path, value):
                    value = fn(value)
            new_dict[key] = value
        new_params = unflatten_dict(new_dict)
        if isinstance(inputs, FrozenDict):
            return FrozenDict(new_params)
        else:
            return new_params


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
    params_out = traversal.update(lambda x: jnp.clip(x, a_min=minval), unfreeze(params))

    return freeze(params_out)


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
    params_out = traversal.update(
        lambda x: jnp.clip(x, a_min=minval, a_max=maxval), unfreeze(params)
    )

    return freeze(params_out)
