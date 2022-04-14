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
            [[1, 3, 7],
             [2, 2, 1]],
            [2, 4, 8]
    ))
    >>> x.shape
    ((2, 3), (3,))  # tuple

    >>> x * 2
    (DeviceArray([[2, 6, 14],
                  [4, 4, 2]], dtype=int32),
     DeviceArray([4, 8, 16], dtype=int32))  # BlockArray

    >>> y = BlockArray((
            [[.2],
             [.3]],
            [.4]
    ))
    >>> x + y
    [DeviceArray([[1.2, 3.2, 7.2],
                  [2.3, 2.3, 1.3]], dtype=float32),
     DeviceArray([2.4, 4.4, 8.4], dtype=float32)]  # BlockArray


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




TODO: working with SCICO operators
TODO: indexing
TODO: -x doesn't work



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

     >>> from scico.numpy import BlockArray
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

import inspect
from functools import wraps
from typing import Callable

import jax
import jax.numpy as jnp

from jaxlib.xla_extension import DeviceArray

from .function_lists import binary_ops, unary_ops

# CANCELED: .sum(), etc. should call snp


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
        return self[0].dtype  # TODO: a better solution is beyond current scope

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
