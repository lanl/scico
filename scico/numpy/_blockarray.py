# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Block array class."""

import inspect
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import Callable

import jax
import jax.numpy as jnp

from ._wrapped_function_lists import binary_ops, unary_ops
from .util import is_collapsible

# Determine type of "standard" jax array since jax.Array is an abstract
# base class type that is not suitable for use here.
JaxArray = type(jnp.array([0]))


class BlockArray:
    """Block array class.

    A block array provides a way to combine arrays of different shapes
    into a single object for use with other SCICO classes. For further
    information, see the
    :ref:`detailed BlockArray documentation <blockarray_class>`.

    Example
    -------

    >>> x = snp.blockarray((
    ...     [[1, 3, 7],
    ...      [2, 2, 1]],
    ...     [2, 4, 8]
    ... ))
    >>> x.shape
    ((2, 3), (3,))
    >>> snp.sum(x)
    Array(30, dtype=int32)
    """

    # Ensure we use BlockArray.__radd__, __rmul__, etc for binary
    # operations of the form op(np.ndarray, BlockArray) See
    # https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.beyond-basics.html#ndarray.__array_priority__
    __array_priority__ = 1

    def __init__(self, inputs):
        # convert inputs to jax arrays
        self.arrays = [x if isinstance(x, jnp.ndarray) else jnp.array(x) for x in inputs]

        # check that dtypes match
        if not all(a.dtype == self.arrays[0].dtype for a in self.arrays):
            raise ValueError("Heterogeneous dtypes not supported.")

    @property
    def dtype(self):
        """Return the dtype of the blocks, which must currently be homogeneous.

        This allows `snp.zeros(x.shape, x.dtype)` to work without a mechanism
        to handle lists of dtypes.
        """
        return self.arrays[0].dtype

    def __len__(self):
        return self.arrays.__len__()

    def __getitem__(self, key):
        """Indexing method equivalent to x[key].

        This is overridden to make, e.g., x[:2] return a BlockArray
        rather than a list.
        """
        result = self.arrays[key]
        if not isinstance(result, jnp.ndarray):
            return BlockArray(result)  # x[k:k+1] returns a BlockArray
        return result  # x[k] returns a jax array

    def __setitem__(self, key, value):
        self.arrays[key] = value

    @staticmethod
    def blockarray(iterable):
        """Construct a :class:`.BlockArray` from a list or tuple of existing array-like."""
        return BlockArray(iterable)

    def __repr__(self):
        return f"BlockArray({repr(self.arrays)})"

    def stack(self, axis=0):
        """Collapse a :class:`.BlockArray` to :class:`jax.Array`.

        Collapse a :class:`.BlockArray` to :class:`jax.Array` by stacking
        the blocks on axis `axis`.

        Args:
            axis: Index of new axis on which blocks are to be stacked.

        Returns:
            A :class:`jax.Array` obtained by stacking.

        Raises:
            ValueError: When called on a :class:`.BlockArray` that is not
               stackable.
        """
        if is_collapsible(self.shape):
            return jnp.stack(self.arrays, axis=axis)
        else:
            raise ValueError(f"BlockArray of shape {self.shape} cannot be collapsed to an Array.")


# Register BlockArray as a jax pytree, without this, jax autograd won't work.
# taken from what is done with tuples in jax._src.tree_util
jax.tree_util.register_pytree_node(
    BlockArray,
    lambda xs: (xs, None),  # to iter
    lambda _, xs: BlockArray(xs),  # from iter
)


# Wrap unary ops like -x.
def _unary_op_wrapper(op_name):
    op = getattr(JaxArray, op_name)

    @wraps(op)
    def op_block_array(self):
        return BlockArray(op(x) for x in self)

    return op_block_array


for op_name in unary_ops:
    setattr(BlockArray, op_name, _unary_op_wrapper(op_name))


# Wrap binary ops like x + y. """
def _binary_op_wrapper(op_name):
    op = getattr(JaxArray, op_name)

    @wraps(op)
    def op_block_array(self, other):
        # If other is a block array, we can assume the operation is implemented
        # (because block arrays must contain jax arrays)
        if isinstance(other, BlockArray):
            return BlockArray(op(x, y) for x, y in zip(self, other))

        # If not, need to handle possible NotImplemented
        # without this, block_array + 'hi' -> [NotImplemented, NotImplemented, ...]
        result = list(op(x, other) for x in self)
        if NotImplemented in result:
            return NotImplemented
        return BlockArray(result)

    return op_block_array


for op_name in binary_ops:
    setattr(BlockArray, op_name, _binary_op_wrapper(op_name))


# Wrap jax array properties.
def _jax_array_prop_wrapper(prop_name):
    prop = getattr(JaxArray, prop_name)

    @property
    @wraps(prop)
    def prop_block_array(self):
        result = tuple(getattr(x, prop_name) for x in self)

        # If each jax_array.prop is a jax array, ...
        if all([isinstance(x, jnp.ndarray) for x in result]):
            # ...return a block array...
            return BlockArray(result)

        # ...otherwise return a tuple.
        return result

    return prop_block_array


skip_props = ("at",)
jax_array_props = [
    k
    for k, v in dict(inspect.getmembers(JaxArray)).items()  # (name, method) pairs
    if isinstance(v, property) and k[0] != "_" and k not in dir(BlockArray) and k not in skip_props
]

for prop_name in jax_array_props:
    setattr(BlockArray, prop_name, _jax_array_prop_wrapper(prop_name))


# Wrap jax array methods.
def _jax_array_method_wrapper(method_name):
    method = getattr(JaxArray, method_name)

    # Don't try to set attributes that are None. Not clear why some
    # functions/methods (e.g. block_until_ready) have None values
    # for these attributes.
    wrapper_assignments = WRAPPER_ASSIGNMENTS
    for attr in ("__name__", "__qualname__"):
        if getattr(method, attr) is None:
            wrapper_assignments = tuple(x for x in wrapper_assignments if x != attr)

    @wraps(method, assigned=wrapper_assignments)
    def method_block_array(self, *args, **kwargs):
        result = tuple(getattr(x, method_name)(*args, **kwargs) for x in self)

        # If each jax_array.method(...) call returns a jax array, ...
        if all([isinstance(x, jnp.ndarray) for x in result]):
            # ...return a block array...
            return BlockArray(result)

        # ...otherwise return a tuple.
        return result

    return method_block_array


skip_methods = ()
jax_array_methods = [
    k
    for k, v in dict(inspect.getmembers(JaxArray)).items()  # (name, method) pairs
    if isinstance(v, Callable)
    and k[0] != "_"
    and k not in dir(BlockArray)
    and k not in skip_methods
]

for method_name in jax_array_methods:
    setattr(BlockArray, method_name, _jax_array_method_wrapper(method_name))
