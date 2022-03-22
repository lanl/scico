# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Extensions of numpy ndarray class.

 .. testsetup::

   >>> import scico
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

Construct from a tuple of arrays (either `ndarray` or `DeviceArray`)
--------------------------------------------------------------------

  .. doctest::

     >>> from scico.blockarray import BlockArray
     >>> import numpy as np
     >>> x0, key = scico.random.randn((32, 32))
     >>> x1, _ = scico.random.randn((16,), key=key)
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
single precision and will have dtype ``float32`` or ``complex64``.


Construct from a single vector and tuple of shapes
--------------------------------------------------

  ::

     >>> x_flat, _ = scico.random.randn((1040,))
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
selected block, e.g. `x[2, 3:4]` is equivalent to `y[3:4]` after
setting `y = x[2]`.


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
`x.at[ibk, idx].set(y)`          `x[ibk, idx] = y`
`x.at[ibk, idx].add(y)`          `x[ibk, idx] += y`
`x.at[ibk, idx].multiply(y)`     `x[ibk, idx] *= y`
`x.at[ibk, idx].divide(y)`       `x[ibk, idx] /= y`
`x.at[ibk, idx].power(y)`        `x[ibk, idx] **= y`
`x.at[ibk, idx].min(y)`          `x[ibk, idx] = np.minimum(x[idx], y)`
`x.at[ibk, idx].max(y)`          `x[ibk, idx] = np.maximum(x[idx], y)`
==============================   ==============================================


Arithmetic and Broadcasting
---------------------------

Suppose :math:`\mb{x}` is a BlockArray with shape :math:`((n, n), (m,))`.

  ::

    >>> x1, key = scico.random.randn((4, 4))
    >>> x2, _ = scico.random.randn((5,), key=key)
    >>> x = BlockArray.array( (x1, x2) )
    >>> x.shape
    ((4, 4), (5,))
    >>> x.num_blocks
    2
    >>> x.size  # 4*4 + 5
    21

Illustrated for the operation `+`, but equally valid for operators
`+, -, *, /, //, **, <, <=, >, >=, ==`


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
With `y.size = x.size`, we have:

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
fashion and support array broadcasting. Examples of ufuncs are `abs`,
`sign`, `conj`, and `exp`.

The JAX library implements most NumPy ufuncs in the :mod:`jax.numpy`
module. However, as JAX does not support subclassing of `DeviceArray`,
the JAX ufuncs cannot be used on :class:`.BlockArray`. As a workaround,
we have wrapped several JAX ufuncs for use on :class:`.BlockArray`; these
are defined in the :mod:`scico.numpy` module.


Reductions
^^^^^^^^^^

Reductions are functions that take an array-like as an input and return
an array of lower dimension. Examples include `mean`, `sum`, `norm`.
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
  applied to that component. In this example, `x[1].ndim == 1`, so no
  reduction is applied to block `x[1]`.

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
from functools import wraps

import jax
import jax.numpy as jnp

from jaxlib.xla_extension import DeviceArray


class BlockArray(tuple):
    """BlockArray"""

    # Ensure we use BlockArray.__radd__, __rmul__, etc for binary
    # operations of the form op(np.ndarray, BlockArray) See
    # https://docs.scipy.org/doc/numpy-1.10.1/user/c-info.beyond-basics.html#ndarray.__array_priority__
    __array_priority__ = 1

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        return sum(x_i.size for x_i in self)

    def ravel(self) -> DeviceArray:
        """Return a copy of ``self._data`` as a contiguous, flattened `DeviceArray`.

        Note that a copy, rather than a view, of the underlying array is
        returned. This is consistent with :func:`jax.numpy.ravel`.

        Returns:
            Copy of underlying flattened array.

        """
        return jnp.concatenate(tuple(x_i.ravel() for x_i in self))

    """ backwards compatibility methods, could be removed """

    @staticmethod
    def array(iterable):
        """Construct a :class:`.BlockArray` from a list or tuple of existing array-like."""
        return BlockArray(iterable)


# register BlockArray as a jax pytree, without this, jax autograd won't work
# taken from what is done with tuples in jax._src.tree_util
jax.tree_util.register_pytree_node(
    BlockArray,
    lambda xs: (xs, None),  # to iter
    lambda _, xs: BlockArray(xs),  # from iter
)

""" wrap binary ops like +, @ """
binary_ops = (
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__pow__",
    "__rpow__",
    "__gt__",
    "__ge__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
)


def _binary_op_wrapper(func):
    @wraps(func)
    def func_ba(self, other):
        if isinstance(other, BlockArray):
            result = BlockArray(map(func, self, other))
        else:
            result = BlockArray(map(lambda self_n: func(self_n, other), self))
        if NotImplemented in result:
            return NotImplemented
        else:
            return result

    return func_ba


for op in binary_ops:
    setattr(BlockArray, op, _binary_op_wrapper(getattr(DeviceArray, op)))


""" wrap blockwise DeviceArray methods, like conj """

da_methods = ("conj",)


def _da_method_wrapper(func):
    @wraps(func)
    def func_ba(self):
        return BlockArray(map(func, self))

    return func_ba


for meth in da_methods:
    setattr(BlockArray, meth, _da_method_wrapper(getattr(DeviceArray, meth)))

""" wrap blockwise DeviceArray properties, like real """

da_props = (
    "real",
    "imag",
    "shape",
    "ndim",
)


def _da_prop_wrapper(prop):
    @property
    def prop_ba(self):
        return BlockArray((getattr(x, prop) for x in self))

    return prop_ba


for prop in da_props:
    setattr(BlockArray, prop, _da_prop_wrapper(prop))
