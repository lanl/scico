# -*- coding: utf-8 -*-
# Copyright (C) 2020-2026 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Block array class."""

import inspect
from collections.abc import Iterable, Iterator
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union, cast

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

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

    arrays: list[Union[jax.Array, jax.ShapeDtypeStruct]]

    def __init__(self, inputs: Iterable[Union[jax.ShapeDtypeStruct, ArrayLike]]) -> None:
        # convert inputs to jax arrays
        self.arrays = [x if isinstance(x, jax.ShapeDtypeStruct) else jnp.array(x) for x in inputs]

        # check that dtypes match
        if not all(a.dtype == self.arrays[0].dtype for a in self.arrays):
            raise ValueError("Heterogeneous dtypes not supported.")

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the blocks, which must currently be homogeneous.

        This allows `snp.zeros(x.shape, x.dtype)` to work without a mechanism
        to handle lists of dtypes.
        """
        return self.arrays[0].dtype

    def __len__(self) -> int:
        return self.arrays.__len__()

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union["BlockArray", jax.Array, jax.ShapeDtypeStruct]:
        """Indexing method equivalent to x[key].

        This is overridden to make, e.g., x[:2] return a BlockArray
        rather than a list.
        """
        result = self.arrays[key]
        if isinstance(result, list):
            return BlockArray(result)  # x[k:k+1] returns a BlockArray
        return result  # x[k] returns a jax array

    def __setitem__(self, key: int, value: Union[jax.ShapeDtypeStruct, ArrayLike]) -> None:
        # Apply the same conversion as __init__. Storing the value unconverted
        # leaves the BlockArray inconsistent: after assigning a list, dtype
        # raises AttributeError and shape reports the shape of the list's
        # elements rather than of the block.
        self.arrays[key] = value if isinstance(value, jax.ShapeDtypeStruct) else jnp.array(value)

    @staticmethod
    def blockarray(iterable: Iterable[ArrayLike]) -> "BlockArray":
        """Construct a :class:`.BlockArray` from a list or tuple of existing array-like."""
        return BlockArray(iterable)

    def __repr__(self) -> str:
        return f"BlockArray({repr(self.arrays)})"

    def stack(self, axis: int = 0) -> jax.Array:
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
            # Only a BlockArray of arrays can be stacked. The sole source of
            # ShapeDtypeStruct blocks is scico.numpy.util.shape_dtype_rep, which
            # builds a shape and dtype representation rather than data; calling
            # this method on one raises a TypeError from jnp.stack.
            return jnp.stack(cast(List[jax.Array], self.arrays), axis=axis)
        else:
            raise ValueError(f"BlockArray of shape {self.shape} cannot be collapsed to an Array.")

    if TYPE_CHECKING:
        # The operators, properties and jax array methods listed in
        # unary_ops, binary_ops, jax_array_props and jax_array_methods are
        # attached to this class by the setattr loops below the class body,
        # which static analysis cannot follow. Declaring them here is what
        # allows mypy to check operations on a BlockArray. The block is never
        # executed, so it has no effect at run time, but it must be kept in
        # step with those lists by hand. The operators are declared
        # individually because __getattr__ does not satisfy the operator
        # protocols; the properties and methods are not, being too numerous
        # and untyped either way.
        def __abs__(self) -> "BlockArray": ...
        def __neg__(self) -> "BlockArray": ...
        def __pos__(self) -> "BlockArray": ...
        def __add__(self, other: Any) -> Any: ...
        def __eq__(self, other: Any) -> Any: ...
        def __floordiv__(self, other: Any) -> Any: ...
        def __ge__(self, other: Any) -> Any: ...
        def __gt__(self, other: Any) -> Any: ...
        def __le__(self, other: Any) -> Any: ...
        def __lt__(self, other: Any) -> Any: ...
        def __matmul__(self, other: Any) -> Any: ...
        def __mod__(self, other: Any) -> Any: ...
        def __mul__(self, other: Any) -> Any: ...
        def __ne__(self, other: Any) -> Any: ...
        def __pow__(self, other: Any) -> Any: ...
        def __radd__(self, other: Any) -> Any: ...
        def __rfloordiv__(self, other: Any) -> Any: ...
        def __rmatmul__(self, other: Any) -> Any: ...
        def __rmul__(self, other: Any) -> Any: ...
        def __rpow__(self, other: Any) -> Any: ...
        def __rsub__(self, other: Any) -> Any: ...
        def __rtruediv__(self, other: Any) -> Any: ...
        def __sub__(self, other: Any) -> Any: ...
        def __truediv__(self, other: Any) -> Any: ...

        # Iteration works through the legacy __getitem__ and __len__ protocol,
        # which mypy does not recognise, so it is declared explicitly. There is
        # no __iter__ attribute at run time.
        def __iter__(self) -> Iterator[jax.Array]: ...

        def __getattr__(self, name: str) -> Any: ...


# Register BlockArray as a jax pytree; without this, jax autograd won't work.
# Taken from what is done with tuples in jax._src.tree_util
jax.tree_util.register_pytree_node(
    BlockArray,
    lambda xs: (xs, None),  # to iter
    lambda _, xs: BlockArray(xs),  # from iter
)


# Wrap unary ops like -x.
def _unary_op_wrapper(op_name: str) -> Callable[["BlockArray"], "BlockArray"]:
    op = getattr(JaxArray, op_name)

    @wraps(op)
    def op_block_array(self: "BlockArray") -> "BlockArray":
        return BlockArray(op(x) for x in self)

    return op_block_array


for op_name in unary_ops:
    setattr(BlockArray, op_name, _unary_op_wrapper(op_name))


# Wrap binary ops like x + y. """
def _binary_op_wrapper(op_name: str) -> Callable[["BlockArray", Any], Union["BlockArray", Any]]:
    op = getattr(JaxArray, op_name)

    @wraps(op)
    def op_block_array(self: "BlockArray", other: Any) -> Union["BlockArray", Any]:
        # If other is a block array, we can assume the operation is
        # implemented (because block arrays must contain jax arrays)
        if isinstance(other, BlockArray):
            return BlockArray(op(x, y) for x, y in zip(self, other))

        # If not, need to handle possible NotImplemented. Without this,
        # block_array + 'hi' -> [NotImplemented, NotImplemented, ...]
        result = list(op(x, other) for x in self)
        if NotImplemented in result:
            return NotImplemented
        return BlockArray(result)

    return op_block_array


for op_name in binary_ops:
    setattr(BlockArray, op_name, _binary_op_wrapper(op_name))


# Wrap jax array properties.
def _jax_array_prop_wrapper(prop_name: str) -> property:
    prop = getattr(JaxArray, prop_name)

    @wraps(prop)
    def prop_block_array(self: "BlockArray") -> Union["BlockArray", tuple]:
        result = tuple(getattr(x, prop_name) for x in self)

        # If each jax_array.prop is a jax array, ...
        if all([isinstance(x, jnp.ndarray) for x in result]):
            # ...return a block array...
            return BlockArray(result)

        # ... otherwise return a tuple.
        return result

    # property is applied here rather than as a decorator above wraps, which is
    # the same call but which mypy cannot follow through the wrapper object.
    return property(prop_block_array)


skip_props = ("at",)
jax_array_props = [
    k
    for k, v in dict(inspect.getmembers(JaxArray)).items()  # (name, method) pairs
    if isinstance(v, property) and k[0] != "_" and k not in dir(BlockArray) and k not in skip_props
]

for prop_name in jax_array_props:
    setattr(BlockArray, prop_name, _jax_array_prop_wrapper(prop_name))


# Wrap jax array methods.
def _jax_array_method_wrapper(method_name: str) -> Callable[..., Any]:
    method = getattr(JaxArray, method_name)

    # Don't try to set attributes that are None. Not clear why some
    # functions/methods (e.g. block_until_ready) have None values
    # for these attributes.
    wrapper_assignments: Tuple[str, ...] = WRAPPER_ASSIGNMENTS
    for attr in ("__name__", "__qualname__"):
        if getattr(method, attr) is None:
            wrapper_assignments = tuple(x for x in wrapper_assignments if x != attr)

    @wraps(method, assigned=wrapper_assignments)
    def method_block_array(
        self: "BlockArray", *args: Any, **kwargs: Any
    ) -> Union["BlockArray", tuple]:
        result = tuple(getattr(x, method_name)(*args, **kwargs) for x in self)

        # If each jax_array.method(...) call returns a jax array, ...
        if all([isinstance(x, jnp.ndarray) for x in result]):
            # ... return a block array...
            return BlockArray(result)

        # ... otherwise return a tuple.
        return result

    return method_block_array


skip_methods = ()
jax_array_methods = [
    k
    for k, v in dict(inspect.getmembers(JaxArray)).items()  # (name, method) pairs
    if callable(v) and k[0] != "_" and k not in dir(BlockArray) and k not in skip_methods
]

for method_name in jax_array_methods:
    setattr(BlockArray, method_name, _jax_array_method_wrapper(method_name))
