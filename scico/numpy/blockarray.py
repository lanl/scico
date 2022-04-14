# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Block array class.

 .. testsetup::

   >>> import scico
   >>> import scico.numpy as snp
   >>> from scico.numpy import BlockArray
   >>> import numpy as np
   >>> import jax.numpy

The class :class:`.BlockArray` provides a way to combine arrays of
different shapes into a single object for use with other SCICO classes.
A :class:`.BlockArray` consists of a list of `DeviceArray` objects,
which we refer to as blocks. :class:`.BlockArray`s differ from lists in
that, whenever possible, :class:`.BlockArray` properties and methods
(including unary and binary operators like +, -, *, ...) automatically
map along the blocks, returning another :class:`.BlockArray` or tuple as
appropriate. For example,

  ::

    >>> x = BlockArray((
    ...     [[1, 3, 7],
    ...      [2, 2, 1]],
    ...     [2, 4, 8]
    ... ))

    >>> x.shape  # returns tuple
    ((2, 3), (3,))

    >>> x * 2  # returns BlockArray
    [DeviceArray([[ 2,  6, 14],
                 [ 4,  4,  2]], dtype=int32), DeviceArray([ 4,  8, 16], dtype=int32)]

    >>> y = BlockArray((
    ...        [[.2],
    ...         [.3]],
    ...        [.4]
    ... ))

    >>> x + y  # returns BlockArray
    [DeviceArray([[1.2, 3.2, 7.2],
                  [2.3, 2.3, 1.3]], dtype=float32), DeviceArray([2.4, 4.4, 8.4], dtype=float32)]


NumPy Functions
===============

:mod:`scico.numpy` provides a wrapper around :mod:`jax.numpy` where many
of the functions have been extended to work with `BlockArray`s. In
particular:

* When a tuple of tuples is passed as the `shape`
argument to an array creation routine, a `BlockArray` is created.

* When a `BlockArray` is passed to a reduction function, the blocks are
ravelled (i.e., reshaped to be 1D) and concatenated before the reduction
is applied. This behavior may be prevented by passing the `axis`
argument, in which case the function is mapped over the blocks.

* When one or more `BlockArray`s is passed to a mathematical
function that is not a reduction, the function is mapped over
(corresponding) blocks.

For lists of array creation routines, reduction functions, and mathematical
functions that have been wrapped in this manner,
see `scico.numpy.creation_routines`, `scico.numpy.reduction_fuctions`,
and
`scico.numpy.mathematical_functions`.

:mod:`scico.numpy.testing` provides a wrapper around :mod:`numpy.testing`
where some functions have been extended to map over blocks,
notably `scico.numpy.testing.allclose`.
For a list of the extended functions, see `scico.numpy.testing_functions`.


Motivating Example
==================

Consider a two dimensional array :math:`\mb{x} \in \mathbb{R}^{n \times m}`.

We compute the discrete differences of :math:`\mb{x}` in the horizontal
and vertical directions, generating two new arrays: :math:`\mb{x}_h \in
\mathbb{R}^{n \times (m-1)}` and :math:`\mb{x}_v \in \mathbb{R}^{(n-1)
\times m}`.

As these arrays are of different shapes, we cannot combine them into a
single `ndarray`. Instead, we might vectorize each array and concatenate
the resulting vectors, leading to :math:`\mb{\bar{x}} \in
\mathbb{R}^{n(m-1) + m(n-1)}`, which can be stored as a one-dimensional
`ndarray`. Unfortunately, this makes it hard to access the individual
components :math:`\mb{x}_h` and :math:`\mb{x}_v`.

Instead, we can form a :class:`.BlockArray`: :math:`\mb{x}_B =
[\mb{x}_h, \mb{x}_v]`


  ::

    >>> n = 32
    >>> m = 16
    >>> x_h, key = scico.random.randn((n, m-1))
    >>> x_v, _ = scico.random.randn((n-1, m), key=key)

    # Form the blockarray
    >>> x_B = BlockArray.array([x_h, x_v])

    # The blockarray shape is a tuple of tuples
    >>> x_B.shape
    ((32, 15), (31, 16))

    # Each block component can be easily accessed
    >>> x_B[0].shape
    (32, 15)
    >>> x_B[1].shape
    (31, 16)


Constructing a BlockArray
=========================

  .. doctest::

     >>> from scico.numpy import BlockArray
     >>> import numpy as np
     >>> x0, key = scico.random.randn((32, 32))
     >>> x1, _ = scico.random.randn((16,), key=key)
     >>> X = BlockArray.array((x0, x1))
     >>> X.shape
     ((32, 32), (16,))
     >>> X.size
     (1024, 16)
     >>> len(X)
     2

While :func:`.BlockArray.array` will accept either `ndarray` or
`DeviceArray` as input, the resulting :class:`.BlockArray` will be backed
by a `DeviceArray` memory buffer.

**Note**: constructing a :class:`.BlockArray` always involves a copy to
a new `DeviceArray` memory buffer.

**Note**: by default, the resulting :class:`.BlockArray` is cast to
single precision and will have dtype ``float32`` or ``complex64``.



Operating on a BlockArray
=========================

.. _blockarray_indexing:

Indexing
--------

`BlockArray` indexing works just like indexing on a list.

Multiplication Between BlockArray and :class:`.LinearOperator`
--------------------------------------------------------------

The :class:`.Operator` and :class:`.LinearOperator` classes are designed
to work on :class:`.BlockArray`\ s in addition to `DeviceArray`\ s.
For example


   ::

      >>> x, key = scico.random.randn((3, 4))
      >>> A_1 = scico.linop.Identity(x.shape)
      >>> A_1.shape  # array -> array
      ((3, 4), (3, 4))

      >>> A_2 = scico.linop.FiniteDifference(x.shape)
      >>> A_2.shape  # array -> BlockArray
      (((2, 4), (3, 3)), (3, 4))

      >>> diag = BlockArray.array([np.array(1.0), np.array(2.0)])
      >>> A_3 = scico.linop.Diagonal(diag, input_shape=(A_2.output_shape))
      >>> A_3.shape  # BlockArray -> BlockArray
      (((2, 4), (3, 3)), ((2, 4), (3, 3)))

"""

import inspect
from functools import wraps
from typing import Callable

import jax
import jax.numpy as jnp

from jaxlib.xla_extension import DeviceArray

from .function_lists import binary_ops, unary_ops


class BlockArray(list):
    """BlockArray"""

    # Ensure we use BlockArray.__radd__, __rmul__, etc for binary
    # operations of the form op(np.ndarray, BlockArray) See
    # https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.beyond-basics.html#ndarray.__array_priority__
    __array_priority__ = 1

    def __init__(self, inputs):
        # convert inputs to DeviceArrays
        arrays = [x if isinstance(x, jnp.ndarray) else jnp.array(x) for x in inputs]

        # check that dtypes match
        if not all(a.dtype == arrays[0].dtype for a in arrays):
            raise ValueError("Heterogeneous dtypes not supported")

        return super().__init__(arrays)

    def _full_ravel(self) -> DeviceArray:
        """Return a copy of ``self._data`` as a contiguous, flattened `DeviceArray`.

        Note that a copy, rather than a view, of the underlying array is
        returned. This is consistent with :func:`jax.numpy.ravel`.

        Returns:
            Copy of underlying flattened array.

        """
        return jnp.concatenate(tuple(x_i.ravel() for x_i in self))

    @property
    def dtype(self):
        """Allow snp.zeros(x.shape, x.dtype) to work."""
        return self[0].dtype

    def __getitem__(self, key):
        """Make, e.g., x[:2] return a BlockArray, not a list."""
        result = super().__getitem__(key)
        if not isinstance(result, jnp.ndarray):
            return BlockArray(result)
        return result

    """ backwards compatibility methods, could be removed """

    @staticmethod
    def array(iterable):
        """Construct a :class:`.BlockArray` from a list or tuple of existing array-like."""
        return BlockArray(iterable)


# Register BlockArray as a jax pytree, without this, jax autograd won't work.
# taken from what is done with tuples in jax._src.tree_util
jax.tree_util.register_pytree_node(
    BlockArray,
    lambda xs: (xs, None),  # to iter
    lambda _, xs: BlockArray(xs),  # from iter
)

""" Wrap unary ops like -x. """


def _unary_op_wrapper(op):
    op = getattr(DeviceArray, op_name)

    @wraps(op)
    def op_ba(self):
        return BlockArray(op(x) for x in self)

    return op_ba


for op_name in unary_ops:
    setattr(BlockArray, op_name, _unary_op_wrapper(op_name))

""" Wrap binary ops like x+y. """


def _binary_op_wrapper(op_name):
    op = getattr(DeviceArray, op_name)

    @wraps(op)
    def op_ba(self, other):
        if isinstance(other, BlockArray):
            return BlockArray(op(x, y) for x, y in zip(self, other))

        result = BlockArray(op(x, other) for x in self)
        if NotImplemented in result:
            return NotImplemented
        return result

    return op_ba


for op_name in binary_ops:
    setattr(BlockArray, op_name, _binary_op_wrapper(op_name))


""" Wrap DeviceArray properties. """


def _da_prop_wrapper(prop_name):
    prop = getattr(DeviceArray, prop_name)

    @property
    @wraps(prop)
    def prop_ba(self):
        result = tuple(getattr(x, prop_name) for x in self)
        if isinstance(result[0], jnp.ndarray):
            return BlockArray(result)
        return result

    return prop_ba


skip_props = ("at",)
da_props = [
    k
    for k, v in dict(inspect.getmembers(DeviceArray)).items()
    if isinstance(v, property) and k[0] != "_" and k not in dir(BlockArray) and k not in skip_props
]

for prop_name in da_props:
    setattr(BlockArray, prop_name, _da_prop_wrapper(prop_name))

""" Wrap DeviceArray methods. """


def _da_method_wrapper(method):
    @wraps(method)
    def method_ba(self, *args, **kwargs):
        result = tuple(getattr(x, method)(*args, **kwargs) for x in self)
        if isinstance(result[0], jnp.ndarray):
            return BlockArray(result)
        return result

    return method_ba


skip_methods = ()
da_methods = [
    k
    for k, v in dict(inspect.getmembers(DeviceArray)).items()
    if isinstance(v, Callable)
    and k[0] != "_"
    and k not in dir(BlockArray)
    and k not in skip_methods
]

for method in da_methods:
    setattr(BlockArray, method, _da_method_wrapper(method))
