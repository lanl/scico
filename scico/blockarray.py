# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Extensions of numpy ndarray class.

 .. testsetup::

   >>> import scico.numpy as snp
   >>> from scico.blockarray import BlockArray
   >>> import numpy as np
   >>> import jax.numpy

The class :class:`.BlockArray` is a `jagged array
<https://en.wikipedia.org/wiki/Jagged_array>`_ that aims to mimic the
:class:`numpy.ndarray` interface where appropriate.

A :class:`.BlockArray` object consists of a tuple of `DeviceArray`
objects that share their memory buffers with non-overlapping, contiguous
regions of a common one-dimensional `DeviceArray`. A :class:`.BlockArray`
contains the following size attributes:

* `shape`:  A tuple of tuples containing component dimensions.
* `size`: The sum of the size of each component block; this is the length
  of the underlying one-dimensional `DeviceArray`.
* `num_blocks`: The number of components (blocks) that comprise the
  :class:`.BlockArray`.


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
    >>> x_h = np.random.randn(n, m-1)
    >>> x_v = np.random.randn(n-1, m)

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

Construct from a tuple of arrays (either `ndarray` or `DeviceArray`)
--------------------------------------------------------------------

  .. doctest::

     >>> from scico.blockarray import BlockArray
     >>> import numpy as np
     >>> x0 = np.random.randn(32, 32)
     >>> x1 = np.random.randn(16)
     >>> X = BlockArray.array((x0, x1))
     >>> X.shape
     ((32, 32), (16,))
     >>> X.size
     1040
     >>> X.num_blocks
     2

While :func:`.BlockArray.array` will accept either `ndarray` or
`DeviceArray` as input, the resulting :class:`.BlockArray` will be backed
by a `DeviceArray` memory buffer.

**Note**: constructing a :class:`.BlockArray` always involves a copy to
a new `DeviceArray` memory buffer.

**Note**: by default, the resulting :class:`.BlockArray` is cast to
single precision and will have dtype `float32` or `complex64`.


Construct from a single vector and tuple of shapes
--------------------------------------------------

  ::

     >>> x_flat = np.random.randn(1040)
     >>> shape_tuple = ((32, 32), (16,))
     >>> X = BlockArray.array_from_flattened(x_flat, shape_tuple=shape_tuple)
     >>> X.shape
     ((32, 32), (16,))



Operating on a BlockArray
=========================

.. _blockarray_indexing:

Indexing
--------

The block index is required to be an integer, selecting a single block and
returning it as an array (*not* a singleton BlockArray). If the index
expression has more than one component, then the initial index indexes the
block, and the remainder of the indexing expression indexes within the
selected block, e.g. ``x[2, 3:4]`` is equivalent to ``y[3:4]`` after
setting ``y = x[2]``.


Indexed Updating
----------------

BlockArrays support the JAX DeviceArray `indexed update syntax
<https://jax.readthedocs.io/en/latest/jax.ops.html#indexed-update-operators>`_


The index must be of the form [ibk] or [ibk, idx], where `ibk` is the
index of the block to be updated, and `idx` is a general index of the
elements to be updated in that block. In particular, `ibk` cannot be a
`slice`. The general index `idx` can be omitted, in which case an entire
block is updated.


==============================   ==============================================
Alternate syntax                 Equivalent in-place expression
==============================   ==============================================
``x.at[ibk, idx].set(y)``        ``x[ibk, idx] = y``
``x.at[ibk, idx].add(y)``        ``x[ibk, idx] += y``
``x.at[ibk, idx].multiply(y)``   ``x[ibk, idx] *= y``
``x.at[ibk, idx].divide(y)``     ``x[ibk, idx] /= y``
``x.at[ibk, idx].power(y)``      ``x[ibk, idx] **= y``
``x.at[ibk, idx].min(y)``        ``x[ibk, idx] = np.minimum(x[idx], y)``
``x.at[ibk, idx].max(y)``        ``x[ibk, idx] = np.maximum(x[idx], y)``
==============================   ==============================================


Arithmetic and Broadcasting
---------------------------

Suppose :math:`\mb{x}` is a BlockArray with shape :math:`((n, n), (m,))`.

  ::

    >>> x1 = np.random.randn(4, 4)
    >>> x2 = np.random.randn(5)
    >>> x = BlockArray.array( (x1, x2) )
    >>> x.shape
    ((4, 4), (5,))
    >>> x.num_blocks
    2
    >>> x.size  # 4*4 + 5
    21

Illustrated for the operation ``+``, but equally valid for operators
``+, -, *, /, //, **, <, <=, >, >=, ==``


Operations with BlockArrays with same number of blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mb{y}` be a BlockArray with the same number of blocks as
:math:`\mb{x}`.

  .. math::
     \mb{x} + \mb{y}
     =
     \begin{bmatrix}
       \mb{x}[0] + \mb{y}[0] \\
       \mb{x}[1] + \mb{y}[1] \\
     \end{bmatrix}

This operation depends on pair of blocks from :math:`\mb{x}` and
:math:`\mb{y}` being broadcastable against each other.



Operations with a scalar
^^^^^^^^^^^^^^^^^^^^^^^^

The scalar is added to each element of the :class:`.BlockArray`:

  .. math::
     \mb{x} + 1
     =
     \begin{bmatrix}
       \mb{x}[0] + 1 \\
       \mb{x}[1] + 1\\
     \end{bmatrix}


  ::

     >>> y = x + 1
     >>> np.testing.assert_allclose(y[0], x[0] + 1)
     >>> np.testing.assert_allclose(y[1], x[1] + 1)



Operations with a 1D `ndarray` of size equal to `num_blocks`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *i*\th scalar is added to the *i*\th  block of the
:class:`.BlockArray`:

  .. math::
   \mb{x}
     +
   \begin{bmatrix}
     1 \\
     2
   \end{bmatrix}
   =
   \begin{bmatrix}
     \mb{x}[0] + 1 \\
     \mb{x}[1] + 2\\
   \end{bmatrix}


  ::

     >>> y = x + np.array([1, 2])
     >>> np.testing.assert_allclose(y[0], x[0] + 1)
     >>> np.testing.assert_allclose(y[1], x[1] + 2)


Operations with an ndarray of `size` equal to :class:`.BlockArray` size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first cast the `ndarray` to a BlockArray with same shape as
:math:`\mb{x}`, then apply the operation on the resulting BlockArrays.
With ``y.size = x.size``, we have:

  .. math::
   \mb{x}
   +
   \mb{y}
   =
   \begin{bmatrix}
     \mb{x}[0] + \mb{y}[0] \\
     \mb{x}[1] + \mb{y}[1]\\
   \end{bmatrix}

Equivalently, the BlockArray is first flattened, then added to the
flattened `ndarray`, and the result is reformed into a BlockArray with
the same shape as :math:`\mb{x}`



MatMul
------

Between two BlockArrays
^^^^^^^^^^^^^^^^^^^^^^^

The matmul is computed between each block of the two BlockArrays.

The BlockArrays must have the same number of blocks, and each pair of
blocks must be broadcastable.

  .. math::
   \mb{x} @ \mb{y}
   =
   \begin{bmatrix}
     \mb{x}[0] @ \mb{y}[0] \\
     \mb{x}[1] @ \mb{y}[1]\\
   \end{bmatrix}



Between BlockArray and Ndarray/DeviceArray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This operation is not defined.


Between BlockArray and :class:`.LinearOperator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


NumPy ufuncs
------------

`NumPy universal functions (ufuncs) <https://numpy.org/doc/stable/reference/ufuncs.html>`_
are functions that operate on an `ndarray` on an element-by-element
fashion and support array broadcasting. Examples of ufuncs are ``abs``,
``sign``, ``conj``, and ``exp``.

The JAX library implements most NumPy ufuncs in the :mod:`jax.numpy`
module. However, as JAX does not support subclassing of `DeviceArray`,
the JAX ufuncs cannot be used on :class:`.BlockArray`. As a workaround,
we have wrapped several JAX ufuncs for use on :class:`.BlockArray`; these
are defined in the :mod:`scico.numpy` module.


Reductions
^^^^^^^^^^

Reductions are functions that take an array-like as an input and return
an array of lower dimension. Examples include ``mean``, ``sum``, ``norm``.
BlockArray reductions are located in the :mod:`scico.numpy` module

:class:`.BlockArray` tries to mirror `ndarray` reduction semantics where
possible, but cannot provide a one-to-one match as the block components
may be of different size.

Consider the example BlockArray

  .. math::
   \mb{x} = \begin{bmatrix}
   \begin{bmatrix}
    1 & 1 \\
     1 & 1
   \end{bmatrix} \\
   \begin{bmatrix}
    2 \\
    2
   \end{bmatrix}
   \end{bmatrix}.

We have

  .. doctest::

    >>> import scico.numpy as snp
    >>> x = BlockArray.array((np.ones((2,2)), 2*np.ones((2))))
    >>> x.shape
    ((2, 2), (2,))
    >>> x.size
    6
    >>> x.num_blocks
    2



  If no axis is specified, the reduction is applied to the flattened
  array:

  .. doctest::

    >>> snp.sum(x, axis=None).item()
    8.0

  Reducing along the 0-th axis crushes the `BlockArray` down into a
  single `DeviceArray` and requires all blocks to have the same shape
  otherwise, an error is raised.

  .. doctest::

    >>> snp.sum(x, axis=0)
    Traceback (most recent call last):
    ValueError: Evaluating sum of BlockArray along axis=0 requires all blocks to be same shape; got ((2, 2), (2,))

    >>> y = BlockArray.array((np.ones((2,2)), 2*np.ones((2, 2))))
    >>> snp.sum(y, axis=0)
    DeviceArray([[3., 3.],
                 [3., 3.]], dtype=float32)

  Reducing along axis :math:`n` is equivalent to reducing each component
  along axis :math:`n-1`:

  .. math::
   \text{sum}(x, axis=1) = \begin{bmatrix}
   \begin{bmatrix}
    2 \\
     2
   \end{bmatrix} \\
   \begin{bmatrix}
    4 \\
   \end{bmatrix}
   \end{bmatrix}


  If a component does not have axis :math:`n-1`, the reduction is not
  applied to that component. In this example, ``x[1].ndim == 1``, so no
  reduction is applied to block ``x[1]``.

  .. math::
   \text{sum}(x, axis=2) = \begin{bmatrix}
   \begin{bmatrix}
    2 \\
     2
   \end{bmatrix} \\
   \begin{bmatrix}
    2 \\
    2
   \end{bmatrix}
   \end{bmatrix}


Code version

  .. doctest::

    >>> snp.sum(x, axis=1)  # doctest: +SKIP
    BlockArray([[2, 2],
                [4,] ])

    >>> snp.sum(x, axis=2)  # doctest: +SKIP
    BlockArray([ [2, 2],
                 [2,] ])


"""

from __future__ import annotations

from functools import wraps
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import xla
from jax.interpreters.xla import DeviceArray
from jax.tree_util import register_pytree_node, tree_flatten

from jaxlib.xla_extension import Buffer

from scico import array
from scico.typing import Axes, AxisIndex, BlockShape, DType, JaxArray, Shape

_arraylikes = (Buffer, DeviceArray, np.ndarray)

__author__ = """\n""".join(
    ["Luke Pfister <luke.pfister@gmail.com>", "Brendt Wohlberg <brendt@ieee.org>"]
)


def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension.

    A wrapper for :func:`jax.numpy.atleast_1d` that acts as usual on
    ndarrays and DeviceArrays, and returns BlockArrays unmodified.
    """

    if len(arys) == 1:
        arr = arys[0]
        return arr if isinstance(arr, BlockArray) else jnp.atleast_1d(arr)

    out = []
    for arr in arys:
        if isinstance(arr, BlockArray):
            out.append(arr)
        else:
            out.append(jnp.atleast_1d(arr))
    return out


# Append docstring from original jax.numpy function
atleast_1d.__doc__ = (
    atleast_1d.__doc__.replace("\n    ", "\n")  # deal with indentation differences
    + "\nDocstring for :func:`jax.numpy.atleast_1d`:\n\n"
    + "\n".join(jax.numpy.atleast_1d.__doc__.split("\n")[2:])
)


def reshape(
    a: Union[JaxArray, BlockArray], newshape: Union[Shape, BlockShape]
) -> Union[JaxArray, BlockArray]:
    """Change the shape of an array without changing its data.

    Args:
        a: Array to be reshaped.
        newshape: The new shape should be compatible with the original
            shape. If an integer, then the result will be a 1-D array of
            that length. One shape dimension can be -1. In this case,
            the value is inferred from the length of the array and
            remaining dimensions. If a tuple of tuple of ints, a
            :class:`.BlockArray` is returned.

    Returns:
        The reshaped array. Unlike :func:`numpy.reshape`, a copy is
        always returned.
    """

    if array.is_nested(newshape):
        # x is a blockarray
        return BlockArray.array_from_flattened(a, newshape)

    return jnp.reshape(a, newshape)


def block_sizes(shape: Union[Shape, BlockShape]) -> Axes:
    r"""Compute the 'sizes' of (possibly nested) block shapes.

    This function computes ``block_sizes(z.shape) == (_.size for _ in z)``


    Args:
       shape: A shape tuple; possibly containing nested tuples.


    Examples:

    .. doctest::
        >>> import scico.numpy as snp

        >>> x = BlockArray.ones( ( (4, 4), (2,)))
        >>> x.size
        18

        >>> y = snp.ones((3, 3))
        >>> y.size
        9

        >>> z = BlockArray.array([x, y])
        >>> block_sizes(z.shape)
        (18, 9)

        >>> zz = BlockArray.array([z, z])
        >>> block_sizes(zz.shape)
        (27, 27)
    """

    if isinstance(shape, BlockArray):
        raise TypeError(
            "Expected a `shape` (possibly nested tuple of ints); got :class:`.BlockArray`."
        )

    out = []
    if array.is_nested(shape):
        # shape is nested -> at least one element came from a blockarray
        for y in shape:
            if array.is_nested(y):
                # recursively calculate the block size until we arrive at
                # a tuple (shape of a non-block array)
                while array.is_nested(y):
                    y = block_sizes(y)
                out.append(np.sum(y))  # adjacent block sizes are added together
            else:
                # this is a tuple; size given by product of elements
                out.append(np.prod(y))
        return tuple(out)

    # shape is a non-nested tuple; return the product
    return np.prod(shape)


def _decompose_index(idx: Union[int, Tuple(AxisIndex)]) -> Tuple:
    """Decompose a BlockArray indexing expression into components.

    Decompose a BlockArray indexing expression into block and array
    components.

    Args:
        idx: BlockArray indexing expression.

    Returns:
        A tuple (idxblk, idxarr) with entries corresponding to the
        integer block index and the indexing to be applied to the
        selected block, respectively. The latter is ``None`` if the
        indexing expression simply selects one of the blocks (i.e.
        it consists of a single integer).

    Raises:
        TypeError: If the block index is not an integer.
    """
    if isinstance(idx, tuple):
        idxblk = idx[0]
        idxarr = idx[1:]
    else:
        idxblk = idx
        idxarr = None
    if not isinstance(idxblk, int):
        raise TypeError("Block index must be an integer")
    return idxblk, idxarr


def indexed_shape(shape: Shape, idx: Union[int, Tuple[AxisIndex, ...]]) -> Tuple[int, ...]:
    """Determine the shape of the result of indexing a BlockArray.

    Args:
        shape: Shape of BlockArray.
        idx: BlockArray indexing expression.

    Returns:
        Shape of the selected block, or slice of that block if ``idx`` is a tuple
        rather than an integer.
    """
    idxblk, idxarr = _decompose_index(idx)
    if idxblk < 0:
        idxblk = len(shape) + idxblk
    if idxarr is None:
        return shape[idxblk]
    return array.indexed_shape(shape[idxblk], idxarr)


def _flatten_blockarrays(inp, *args, **kwargs):
    """Flatten any blockarrays present in inp, args, or kwargs."""

    def _flatten_if_blockarray(inp):
        if isinstance(inp, BlockArray):
            return inp._data
        return inp

    inp_ = _flatten_if_blockarray(inp)
    args_ = (_flatten_if_blockarray(_) for _ in args)
    kwargs_ = {key: _flatten_if_blockarray(val) for key, val in kwargs.items()}
    return inp_, args_, kwargs_


def _block_array_ufunc_wrapper(func):
    """Wrap a "ufunc" to allow for joint operation on `DeviceArray` and `BlockArray`."""

    @wraps(func)
    def wrapper(inp, *args, **kwargs):
        all_args = (inp,) + args + tuple(kwargs.items())
        if any([isinstance(_, BlockArray) for _ in all_args]):
            # If 'inp' is a BlockArray, call func on inp._data
            # Then return a BlockArray of the same shape as inp

            inp_, args_, kwargs_ = _flatten_blockarrays(inp, *args, **kwargs)
            flat_out = func(inp_, *args_, **kwargs_)
            return BlockArray.array_from_flattened(flat_out, inp.shape)

        # Otherwise call the function normally
        return func(inp, *args, **kwargs)

    if not hasattr(func, "__doc__") or func.__doc__ is None:
        return wrapper

    wrapper.__doc__ = (
        f":func:`{func.__name__}` wrapped to operate on :class:`BlockArray`" + "\n\n" + func.__doc__
    )
    return wrapper


def _block_array_reduction_wrapper(func):
    """Wrap a reduction (eg. sum, norm) to allow for joint operation on
    `DeviceArray` and `BlockArray`."""

    @wraps(func)
    def wrapper(inp, *args, axis=None, **kwargs):

        all_args = (inp,) + args + tuple(kwargs.items())
        if any([isinstance(_, BlockArray) for _ in all_args]):
            if axis is None:
                # Treat as a single long vector
                inp_, args_, kwargs_ = _flatten_blockarrays(inp, *args, **kwargs)
                return func(inp_, *args_, **kwargs_)

            if type(axis) == tuple:
                raise Exception(
                    f"""Evaluating {func.__name__} on a BlockArray with a tuple argument to
                    axis is not currently supported"""
                )

            if axis == 0:  # reduction along block axis
                # reduction along axis=0 only makes sense if all blocks are the same shape
                # so we can convert to a standard DeviceArray of shape (inp.num_blocks, ...)
                # and reduce along axis = 0
                if all([bk_shape == inp.shape[0] for bk_shape in inp.shape]):
                    view_shape = (inp.num_blocks,) + inp.shape[0]
                    return func(inp._data.reshape(view_shape), *args, axis=0, **kwargs)

                raise ValueError(
                    f"Evaluating {func.__name__} of BlockArray along axis=0 requires "
                    f"all blocks to be same shape; got {inp.shape}"
                )

            # Reduce each block individually along axis-1
            out = []
            for bk in inp:
                if isinstance(bk, BlockArray):
                    # This block is itself a blockarray, so call this wrapped reduction
                    # on axis-1
                    tmp = _block_array_reduction_wrapper(func)(bk, *args, axis=axis - 1, **kwargs)
                else:
                    if axis - 1 >= bk.ndim:
                        # Trying to reduce along a dim that doesn't exist for this block,
                        # so just return the block.
                        # ie broadcast to shape (..., 1) and reduce along axis=-1
                        tmp = bk
                    else:
                        tmp = func(bk, *args, axis=axis - 1, **kwargs)
                out.append(atleast_1d(tmp))
            return BlockArray.array(out)

        if axis is None:
            # 'axis' might not be a valid kwarg (eg dot, vdot), so don't pass it
            return func(inp, *args, **kwargs)

        return func(inp, *args, axis=axis, **kwargs)

    if not hasattr(func, "__doc__") or func.__doc__ is None:
        return wrapper

    wrapper.__doc__ = (
        f":func:`{func.__name__}` wrapped to operate on :class:`BlockArray`" + "\n\n" + func.__doc__
    )
    return wrapper


def _block_array_matmul_wrapper(func):
    @wraps(func)
    def wrapper(self, other):
        if isinstance(self, BlockArray):
            if isinstance(other, BlockArray):
                # Both blockarrays, work block by block
                return BlockArray.array([func(x, y) for x, y in zip(self, other)])
            raise TypeError(
                f"Operation {func.__name__} not implemented between {type(self)} and {type(other)}"
            )
        return func(self, other)

    if not hasattr(func, "__doc__") or func.__doc__ is None:
        return wrapper
    wrapper.__doc__ = (
        f":func:`{func.__name__}` wrapped to operate on :class:`BlockArray`" + "\n\n" + func.__doc__
    )
    return wrapper


def _block_array_binary_op_wrapper(func):
    """Return a decorator that performs type and shape checking for
    :class:`.BlockArray` arithmetic.
    """

    @wraps(func)
    def wrapper(self, other):
        if isinstance(other, BlockArray):
            if other.shape == self.shape:
                # Same shape blocks, can operate on flattened arrays
                return BlockArray.array_from_flattened(func(self._data, other._data), self.shape)
            if other.num_blocks == self.num_blocks:
                # Will work as long as the shapes are broadcastable
                return BlockArray.array([func(x, y) for x, y in zip(self, other)])
            raise ValueError(
                f"operation not valid on operands with shapes {self.shape}  {other.shape}"
            )
        if any([isinstance(other, _) for _ in _arraylikes]):
            if other.size == 1:
                # Same as operating on a scalar
                return BlockArray.array_from_flattened(func(self._data, other), self.shape)
            if other.size == self.size:
                # A little fast and loose, treat the block array as a length self.size vector
                return BlockArray.array_from_flattened(func(self._data, other), self.shape)
            if other.size == self.num_blocks:
                return BlockArray.array([func(blk, other_) for blk, other_ in zip(self, other)])
            raise ValueError(
                f"operation not valid on operands with shapes {self.shape}  {other.shape}"
            )
        if jnp.isscalar(other) or isinstance(other, core.Tracer):
            return BlockArray.array_from_flattened(func(self._data, other), self.shape)
        raise TypeError(
            f"Operation {func.__name__} not implemented between {type(self)} and {type(other)}"
        )

    if not hasattr(func, "__doc__") or func.__doc__ is None:
        return wrapper
    wrapper.__doc__ = (
        f":func:`{func.__name__}` wrapped to operate on :class:`BlockArray`" + "\n\n" + func.__doc__
    )
    return wrapper


class _AbstractBlockArray(core.ShapedArray):
    """Abstract BlockArray class for JAX tracing.

    See https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
    """

    array_abstraction_level = 0  # Same as jax.core.ConcreteArray

    def __init__(self, shapes, dtype):

        sizes = block_sizes(shapes)
        size = np.sum(sizes)

        super(_AbstractBlockArray, self).__init__((size,), dtype)

        #: Abstract data value
        self._data_aval: core.ShapedArray = core.ShapedArray((size,), dtype)

        #: Array dtype
        self.dtype: DType = dtype

        #: Shape of each block
        self.shapes: BlockShape = shapes

        #: Size of each block
        self.sizes: Shape = sizes

        #: Array specifying boundaries of components as indices in base array
        self.bndpos: np.ndarray = np.r_[0, np.cumsum(sizes)]


# The Jax class is heavily inspired by SparseArray/AbstractSparseArray here:
# https://github.com/google/jax/blob/7724322d1c08c13008815bfb52759a29c2a6823b/tests/custom_object_test.py
class BlockArray:
    """A tuple of :class:`jax.interpreters.xla.DeviceArray` objects.

    A tuple of `DeviceArray` objects that all share their memory buffers
    with non-overlapping, contiguous regions of a common one-dimensional
    `DeviceArray`. It can be used as the common one-dimensional array via
    the :func:`BlockArray.ravel` method, or individual component arrays
    can be accessed individually.
    """

    # Ensure we use BlockArray.__radd__, __rmul__, etc for binary operations of the form
    #    op(np.ndarray, BlockArray)
    # See https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.beyond-basics.html#ndarray.__array_priority__
    __array_priority__ = 1

    def __init__(self, aval: _AbstractBlockArray, data: JaxArray):
        """BlockArray init method.

        Args:
            aval: `Abstract value`_  associated with this array (shape+dtype+weak_type)
            data: The underlying contiguous, flattened `DeviceArray`.

        .. _Abstract value:  https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html

        """
        self._aval = aval
        self._data = data

    def __repr__(self):
        return "scico.blockarray.BlockArray: \n" + self._data.__repr__()

    def __getitem__(self, idx: Union[int, Tuple[AxisIndex, ...]]) -> JaxArray:
        idxblk, idxarr = _decompose_index(idx)
        if idxblk < 0:
            idxblk = self.num_blocks + idxblk
        blk = reshape(self._data[self.bndpos[idxblk] : self.bndpos[idxblk + 1]], self.shape[idxblk])
        if idxarr is not None:
            blk = blk[idxarr]
        return blk

    @_block_array_matmul_wrapper
    def __matmul__(self, other: Union[np.ndarray, BlockArray, JaxArray]) -> JaxArray:
        return self @ other

    @_block_array_matmul_wrapper
    def __rmatmul__(self, other: Union[np.ndarray, BlockArray, JaxArray]) -> JaxArray:
        return other @ self

    @_block_array_binary_op_wrapper
    def __sub__(a, b):
        return a - b

    @_block_array_binary_op_wrapper
    def __rsub__(a, b):
        return b - a

    @_block_array_binary_op_wrapper
    def __mul__(a, b):
        return a * b

    @_block_array_binary_op_wrapper
    def __rmul__(a, b):
        return a * b

    @_block_array_binary_op_wrapper
    def __add__(a, b):
        return a + b

    @_block_array_binary_op_wrapper
    def __radd__(a, b):
        return a + b

    @_block_array_binary_op_wrapper
    def __truediv__(a, b):
        return a / b

    @_block_array_binary_op_wrapper
    def __rtruediv__(a, b):
        return b / a

    @_block_array_binary_op_wrapper
    def __floordiv__(a, b):
        return a // b

    @_block_array_binary_op_wrapper
    def __rfloordiv__(a, b):
        return b // a

    @_block_array_binary_op_wrapper
    def __pow__(a, b):
        return a ** b

    @_block_array_binary_op_wrapper
    def __rpow__(a, b):
        return b ** a

    @_block_array_binary_op_wrapper
    def __gt__(a, b):
        return a > b

    @_block_array_binary_op_wrapper
    def __ge__(a, b):
        return a >= b

    @_block_array_binary_op_wrapper
    def __lt__(a, b):
        return a < b

    @_block_array_binary_op_wrapper
    def __le__(a, b):
        return a <= b

    @_block_array_binary_op_wrapper
    def __eq__(a, b):
        return a == b

    @_block_array_binary_op_wrapper
    def __ne__(a, b):
        return a != b

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_blocks):
            yield self[i]

    @property
    def blocks(self) -> Iterator[int]:
        """Return an iterator yielding component blocks."""
        return self.__iter__()

    @property
    def bndpos(self) -> np.ndarray:
        """Array specifying boundaries of components as indices in base array."""
        return self._aval.bndpos

    @property
    def dtype(self) -> DType:
        """Array dtype."""
        return self._data.dtype

    @property
    def device_buffer(self) -> Buffer:
        """The :class:`jaxlib.xla_extension.Buffer` that backs the
        underlying data array."""
        return self._data.device_buffer

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        return self._aval.size

    @property
    def num_blocks(self) -> int:
        """Number of :class:`.BlockArray` components."""

        return len(self.shape)

    @property
    def ndim(self) -> Shape:
        """Tuple of component ndims."""

        return tuple(len(c) for c in self.shape)

    @property
    def shape(self) -> BlockShape:
        """Tuple of component shapes."""

        return self._aval.shapes

    @property
    def split(self) -> Tuple[JaxArray, ...]:
        """Tuple of component arrays."""

        return tuple(self[k] for k in range(self.num_blocks))

    def conj(self) -> BlockArray:
        """Return a :class:`.BlockArray` with complex-conjugated elements."""

        # Much faster than BlockArray.array([_.conj() for _ in self.blocks])
        return BlockArray.array_from_flattened(self.ravel().conj(), self.shape)

    @property
    def real(self) -> BlockArray:
        """Return a :class:`.BlockArray` with the real part of this array."""
        return BlockArray.array_from_flattened(self.ravel().real, self.shape)

    @property
    def imag(self) -> BlockArray:
        """Return a :class:`.BlockArray` with the imaginary part of this array."""
        return BlockArray.array_from_flattened(self.ravel().imag, self.shape)

    @classmethod
    def array(
        cls, alst: List[Union[np.ndarray, JaxArray]], dtype: Optional[np.dtype] = None
    ) -> BlockArray:
        """Construct a :class:`.BlockArray` from a list or tuple of existing array-like.

        Args:
            alst: Initializers for array components.
                Can be :class:`numpy.ndarray` or
                :class:`jax.interpreters.xla.DeviceArray`
            dtype: Data type of array. If none, dtype is derived from
                dtype of initializers

        Returns:
            :class:`.BlockArray` initialized from `alst` tuple
        """

        if isinstance(alst, (tuple, list)) is False:
            raise TypeError("Input to `array` must be a list or tuple of existing arrays")

        if dtype is None:
            present_types = jax.tree_flatten(jax.tree_map(lambda x: x.dtype, alst))[0]
            dtype = np.find_common_type(present_types, [])

        # alst can be a list/tuple of arrays, or a list/tuple containing list/tuples of arrays
        # consider alst to be a tree where leaves are arrays (possibly abstract arrays)
        # use tree_map to find the shape of each leaf
        # `shapes` will be a tuple of ints and tuples containing ints (possibly nested further)

        # ensure any scalar leaves are converted to (1,) arrays
        def shape_atleast_1d(x):
            return x.shape if x.shape != () else (1,)

        shapes = tuple(
            jax.tree_map(shape_atleast_1d, alst, is_leaf=lambda x: not isinstance(x, (list, tuple)))
        )

        _aval = _AbstractBlockArray(shapes, dtype)
        data_ravel = jnp.hstack(jax.tree_map(lambda x: x.ravel(), jax.tree_flatten(alst)[0]))
        return cls(_aval, data_ravel)

    @classmethod
    def array_from_flattened(
        cls, data_ravel: Union[np.ndarray, JaxArray], shape_tuple: BlockShape
    ) -> BlockArray:
        """Construct a :class:`.BlockArray` from a flattened array and tuple of shapes.

        Args:
            data_ravel: Flattened data array
            shape_tuple: Tuple of tuples containing desired block shapes.

        Returns:
            :class:`.BlockArray` initialized from `data_ravel` and `shape_tuple`
        """

        if not isinstance(data_ravel, DeviceArray):
            data_ravel = jax.device_put(data_ravel)

        shape_tuple_size = np.sum(block_sizes(shape_tuple))

        if shape_tuple_size != data_ravel.size:
            raise ValueError(
                f"""The specified shape_tuple is incompatible with provided data_ravel
    shape_tuple = {shape_tuple}
    shape_tuple_size = {shape_tuple_size}
    len(data_ravel) = {len(data_ravel)}
            """
            )

        _aval = _AbstractBlockArray(shape_tuple, dtype=data_ravel.dtype)
        return cls(_aval, data_ravel)

    @classmethod
    def ones(cls, shape_tuple: BlockShape, dtype: DType = np.float32) -> BlockArray:
        """
        Return a new :class:`.BlockArray` with given block shapes and type, filled with ones.

        Args:
            shape_tuple: Tuple of shapes for component blocks
            dtype: Desired data-type for the :class:`.BlockArray`.
                Default is `numpy.float32`.

        Returns:
            :class:`.BlockArray` of ones with the given component shapes
            and dtype.
        """
        _aval = _AbstractBlockArray(shape_tuple, dtype=dtype)
        data_ravel = jnp.ones(_aval.size, dtype=dtype)
        return cls(_aval, data_ravel)

    @classmethod
    def zeros(cls, shape_tuple: BlockShape, dtype: DType = np.float32) -> BlockArray:
        """
        Return a new :class:`.BlockArray` with given block shapes and type, filled with zeros.

        Args:
            shape_tuple: Tuple of shapes for component blocks.
            dtype: Desired data-type for the :class:`.BlockArray`.
               Default is `numpy.float32`.

        Returns:
            :class:`.BlockArray` of zeros with the given component shapes
            and dtype.
        """
        _aval = _AbstractBlockArray(shape_tuple, dtype=dtype)
        data_ravel = jnp.zeros(_aval.size, dtype=dtype)
        return cls(_aval, data_ravel)

    @classmethod
    def empty(cls, shape_tuple: BlockShape, dtype: DType = np.float32) -> BlockArray:
        """
        Return a new :class:`.BlockArray` with given block shapes and type, filled with zeros.

        Note: like :func:`jax.numpy.empty`, this does not return an
        uninitalized array.

        Args:
            shape_tuple: Tuple of shapes for component blocks
            dtype: Desired data-type for the :class:`.BlockArray`.
               Default is `numpy.float32`.

        Returns:
            :class:`.BlockArray` of zeros with the given component shapes
               and dtype.
        """
        _aval = _AbstractBlockArray(shape_tuple, dtype=dtype)
        data_ravel = jnp.empty(_aval.size, dtype=dtype)
        return cls(_aval, data_ravel)

    @classmethod
    def full(
        cls,
        shape_tuple: BlockShape,
        fill_value: Union[float, complex, int],
        dtype: DType = np.float32,
    ) -> BlockArray:
        """
        Return a new :class:`.BlockArray` with given block shapes and type, filled with
        `fill_value`.

        Args:
            shape_tuple: Tuple of shapes for component blocks.
            fill_value: Fill value
            dtype: Desired data-type for the BlockArray. The default,
               None, means `np.array(fill_value).dtype`.

        Returns:
            :class:`.BlockArray` with the given component shapes and
            dtype and all entries equal to `fill_value`.
        """
        if dtype is None:
            dtype = np.asarray(fill_value).dtype

        _aval = _AbstractBlockArray(shape_tuple, dtype=dtype)
        data_ravel = jnp.full(_aval.size, fill_value=fill_value, dtype=dtype)
        return cls(_aval, data_ravel)

    def copy(self) -> BlockArray:
        """Return a copy of this :class:`.BlockArray`.

        This method is not implemented for BlockArray.

        See Also:
            :meth:`.to_numpy`: Convert a :class:`.BlockArray` into a
            flattened NumPy array.
        """
        # jax DeviceArray copies return a NumPy ndarray. This blockarray class must be backed
        # by a DeviceArray, so cannot be converted to a NumPy-backed BlockArray. The BlockArray
        # .to_numpy() method returns a flattened ndarray.
        #
        # This method may be implemented in the future if jax DeviceArray .copy() is modified to
        # return another DeviceArray.
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        """Return a :class:`numpy.ndarray` containing the flattened form of this
        :class:`.BlockArray`."""

        if isinstance(self._data, DeviceArray):
            host_arr = jax.device_get(self._data.copy())
        else:
            host_arr = self._data.copy()
        return host_arr

    def blockidx(self, idx: int) -> jax._src.ops.scatter._Indexable:
        """Return :class:`jax.ops.index` for a given component block.

        Args:
            idx: Desired block index.

        Returns:
            :class:`jax.ops.index` pointing to desired block.
        """
        return jax.ops.index[self.bndpos[idx] : self.bndpos[idx + 1]]

    def ravel(self) -> JaxArray:
        """Return a copy of ``self._data`` as a contiguous, flattened `DeviceArray`.

        Note that a copy, rather than a view, of the underlying array is
        returned. This is consistent with :func:`jax.numpy.ravel`.

        Returns:
            Copy of underlying flattened array.

        """
        return self._data[:]

    def flatten(self) -> JaxArray:
        """Return a copy of ``self._data`` as a contiguous, flattened `DeviceArray`.

        Note that a copy, rather than a view, of the underlying array is
        returned. This is consistent with :func:`jax.numpy.ravel`.

        Returns:
            Copy of underlying flattened array.

        """
        return self._data[:]

    def sum(self, axis=None, keepdims=False):
        """Return the sum of the blockarray elements over the given axis.

        Refer to :func:`scico.numpy.sum` for full documentation.
        """
        # Can't just call scico.numpy.sum due to pesky circular import...
        return _block_array_reduction_wrapper(jnp.sum)(self, axis=axis, keepdims=keepdims)


## Register BlockArray as a Jax type
# Our BlockArray is just a single large vector with some extra sugar
class _ConcreteBlockArray(_AbstractBlockArray):
    pass


def _block_array_result_handler(device, _aval):
    def build_block_array(data_buf):
        data = xla.DeviceArray(_aval._data_aval, device, None, data_buf)
        return BlockArray(_aval, data)

    return build_block_array


def _block_array_shape_handler(a):
    return (xla.xc.Shape.array_shape(a._data_aval.dtype, a._data_aval.shape),)


def _block_array_device_put_handler(a, device):
    return (xla.xb.get_device_backend(device).buffer_from_pyval(a._data, device),)


core.pytype_aval_mappings[BlockArray] = lambda x: x._aval
core.raise_to_shaped_mappings[_AbstractBlockArray] = lambda _aval, _: _aval
xla.pytype_aval_mappings[BlockArray] = lambda x: x._aval
xla.canonicalize_dtype_handlers[BlockArray] = lambda x: x
jax._src.dispatch.device_put_handlers[BlockArray] = _block_array_device_put_handler
jax._src.dispatch.result_handlers[_AbstractBlockArray] = _block_array_result_handler
xla.xla_shape_handlers[_AbstractBlockArray] = _block_array_shape_handler


## Handlers to use jax.device_put on BlockArray
def _block_array_tree_flatten(block_arr):
    """Flatten a :class:`.BlockArray` pytree.

    See :func:`jax.tree_util.tree_flatten`.

    Args:
        block_arr (:class:`.BlockArray`): :class:`.BlockArray` to flatten

    Returns:
        children (tuple): :class:`.BlockArray` leaves.
        aux_data (tuple): Extra metadata used to reconstruct BlockArray.
    """

    data_children, data_aux_data = tree_flatten(block_arr._data)
    return (data_children, block_arr._aval)


def _block_array_tree_unflatten(aux_data, children):
    """Construct a :class:`.BlockArray` from a flattened pytree.

    See jax.tree_utils.tree_unflatten

    Args:
        aux_data (tuple): Metadata needed to construct block array.
        children (tuple): Contains block array elements.

    Returns:
        block_arr: Constructed :class:`.BlockArray`.
    """
    return BlockArray(aux_data, children[0])


register_pytree_node(BlockArray, _block_array_tree_flatten, _block_array_tree_unflatten)

# Syntactic sugar for the .at operations
# see https://github.com/google/jax/blob/56e9f7cb92e3a099adaaca161cc14153f024047c/jax/_src/numpy/lax_numpy.py#L5900
class _BlockArrayIndexUpdateHelper:
    """The helper class for the `at` property to call indexed update functions.

    The `at` property is syntactic sugar for calling the indexed update
    functions as is done in jax. The index must be of the form [ibk] or
    [ibk,idx], where `ibk` is the index of the block to be updated, and
    `idx` is a general index of the elements to be updated in that block.

    In particular:
    - ``x = x.at[ibk].set(y)`` is an equivalent of ``x[ibk] = y``.
    - ``x = x.at[ibk,idx].set(y)`` is an equivalent of ``x[ibk,idx] = y``.

    The methods ``set, add, multiply, divide, power, maximum, minimum``
    are supported.
    """

    __slots__ = ("_block_array",)

    def __init__(self, block_array):
        self._block_array = block_array

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if isinstance(index[0], slice):
                raise TypeError(f"Slicing not supported along block index")
        return _BlockArrayIndexUpdateRef(self._block_array, index)

    def __repr__(self):
        print(f"_BlockArrayIndexUpdateHelper({repr(self._block_array)})")


class _BlockArrayIndexUpdateRef:
    """Helper object to call indexed update functions for an (advanced) index.

    This object references a source block array and a specific indexer,
    with the first integer specifying the block being updated, and rest
    being the indexer into the array of that block. Methods on this
    object return copies of the source block array that have been
    modified at the positions specified by the indexer in the given block.
    """

    __slots__ = ("_block_array", "bk_index", "index")

    def __init__(self, block_array, index):
        self._block_array = block_array
        if isinstance(index, int):
            self.bk_index = index
            self.index = Ellipsis
        elif index == Ellipsis:
            self.bk_index = Ellipsis
            self.index = Ellipsis
        else:
            self.bk_index = index[0]
            self.index = index[1:]

    def __repr__(self):
        return f"_BlockArrayIndexUpdateRef({repr(self._block_array)}, {repr(self.bk_index)}, {repr(self.index)})"

    def _index_wrapper(self, func_str, values):
        bk_index = self.bk_index
        index = self.index
        arr_tuple = self._block_array.split
        if bk_index == Ellipsis:
            # This may result in multiple copies: one per sub-blockarray,
            # then one to combine into a nested BA.
            retval = BlockArray.array([getattr(_.at[index], func_str)(values) for _ in arr_tuple])
        else:
            retval = BlockArray.array(
                arr_tuple[:bk_index]
                + (getattr(arr_tuple[bk_index].at[index], func_str)(values),)
                + arr_tuple[bk_index + 1 :]
            )
        return retval

    def set(self, values):
        """Pure equivalent of ``x[idx] = y``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("set", values)

    def add(self, values):
        """Pure equivalent of ``x[idx] += y``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("add", values)

    def multiply(self, values):
        """Pure equivalent of ``x[idx] *= y``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("multiply", values)

    def divide(self, values):
        """Pure equivalent of ``x[idx] /= y``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("divide", values)

    def power(self, values):
        """Pure equivalent of ``x[idx] **= y``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("power", values)

    def min(self, values):
        """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] = minimum(x[idx], y)``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("min", values)

    def max(self, values):
        """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

        Return the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] = maximum(x[idx], y)``.

        See :mod:`jax.ops` for details.
        """
        return self._index_wrapper("max", values)


setattr(BlockArray, "at", property(_BlockArrayIndexUpdateHelper))
