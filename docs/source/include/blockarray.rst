.. _blockarray_class:

BlockArray
==========

 .. testsetup::

   >>> import scico
   >>> import scico.numpy as snp
   >>> from scico.numpy import BlockArray
   >>> import numpy as np
   >>> import jax.numpy

The class :class:`.BlockArray` provides a way to combine arrays of
different shapes into a single object for use with other SCICO classes.
A :class:`.BlockArray` consists of a list of :obj:`~jax.numpy.DeviceArray` objects,
which we refer to as blocks. A :class:`.BlockArray` differs from a list in
that, whenever possible, :class:`.BlockArray` properties and methods
(including unary and binary operators like +, -, \*, ...) automatically
map along the blocks, returning another :class:`.BlockArray` or tuple as
appropriate. For example,

  ::

    >>> x = snp.blockarray((
    ...     [[1, 3, 7],
    ...      [2, 2, 1]],
    ...     [2, 4, 8]
    ... ))

    >>> x.shape  # returns tuple
    ((2, 3), (3,))

    >>> x * 2  # returns BlockArray   # doctest: +ELLIPSIS
    BlockArray([DeviceArray([[ 2,  6, 14],
		 [ 4,  4,  2]], dtype=...), DeviceArray([ 4,  8, 16], dtype=...)])

    >>> y = snp.blockarray((
    ...        [[.2],
    ...         [.3]],
    ...        [.4]
    ... ))

    >>> x + y  # returns BlockArray   # doctest: +ELLIPSIS
    BlockArray([DeviceArray([[1.2, 3.2, 7.2],
		  [2.3, 2.3, 1.3]], dtype=...), DeviceArray([2.4, 4.4, 8.4], dtype=...)])


.. _numpy_functions_blockarray:

NumPy and SciPy Functions
-------------------------

:mod:`scico.numpy`, :mod:`scico.numpy.testing`, and
:mod:`scico.scipy.special` provide wrappers around :mod:`jax.numpy`,
:mod:`numpy.testing` and :mod:`jax.scipy.special` where many of the
functions have been extended to work with instances of :class:`.BlockArray`.
In particular:

 * When a tuple of tuples is passed as the `shape`
   argument to an array creation routine, a :class:`.BlockArray` is created.
 * When a :class:`.BlockArray` is passed to a reduction function, the blocks are
   ravelled (i.e., reshaped to be 1D) and concatenated before the reduction
   is applied. This behavior may be prevented by passing the `axis`
   argument, in which case the function is mapped over the blocks.
 * When one or more :class:`.BlockArray` instances are passed to a mathematical
   function that is not a reduction, the function is mapped over
   (corresponding) blocks.

For a list of array creation routines, see

  ::

    >>> scico.numpy.creation_routines  # doctest: +ELLIPSIS
    ('empty', ...)

For a list of  reduction functions, see

  ::

    >>> scico.numpy.reduction_functions  # doctest: +ELLIPSIS
    ('sum', ...)

For lists of the remaining wrapped functions, see

  ::

    >>> scico.numpy.mathematical_functions  # doctest: +ELLIPSIS
    ('sin', ...)
    >>> scico.numpy.testing_functions  # doctest: +ELLIPSIS
    ('testing.assert_allclose', ...)
    >>> import scico.scipy
    >>> scico.scipy.special.functions  # doctest: +ELLIPSIS
    ('betainc', ...)


Motivating Example
------------------

Consider a two-dimensional array :math:`\mb{x} \in \mbb{R}^{n \times m}`.

We compute the discrete differences of :math:`\mb{x}` in the horizontal
and vertical directions, generating two new arrays: :math:`\mb{x}_h \in
\mbb{R}^{n \times (m-1)}` and :math:`\mb{x}_v \in \mbb{R}^{(n-1)
\times m}`.

As these arrays are of different shapes, we cannot combine them into a
single :class:`~numpy.ndarray`. Instead, we might vectorize each array and concatenate
the resulting vectors, leading to :math:`\mb{\bar{x}} \in
\mbb{R}^{n(m-1) + m(n-1)}`, which can be stored as a one-dimensional
:class:`~numpy.ndarray`. Unfortunately, this makes it hard to access the individual
components :math:`\mb{x}_h` and :math:`\mb{x}_v`.

Instead, we can form a :class:`.BlockArray`: :math:`\mb{x}_B =
[\mb{x}_h, \mb{x}_v]`


  ::

    >>> n = 32
    >>> m = 16
    >>> x_h, key = scico.random.randn((n, m-1))
    >>> x_v, _ = scico.random.randn((n-1, m), key=key)

    # Form the blockarray
    >>> x_B = snp.blockarray([x_h, x_v])

    # The blockarray shape is a tuple of tuples
    >>> x_B.shape
    ((32, 15), (31, 16))

    # Each block component can be easily accessed
    >>> x_B[0].shape
    (32, 15)
    >>> x_B[1].shape
    (31, 16)


Constructing a BlockArray
-------------------------

The recommended way to construct a :class:`.BlockArray` is by using the
`snp.blockarray` function.

  ::

     >>> import scico.numpy as snp
     >>> x0, key = scico.random.randn((32, 32))
     >>> x1, _ = scico.random.randn((16,), key=key)
     >>> X = snp.blockarray((x0, x1))
     >>> X.shape
     ((32, 32), (16,))
     >>> X.size
     (1024, 16)
     >>> len(X)
     2

While :func:`.snp.blockarray` will accept either :class:`~numpy.ndarray`\ s or
:obj:`~jax.numpy.DeviceArray`\ s as input, :class:`~numpy.ndarray`\ s
will be converted to :obj:`~jax.numpy.DeviceArray`\ s.


Operating on a BlockArray
-------------------------


.. _blockarray_indexing:

Indexing
^^^^^^^^

:class:`.BlockArray` indexing works just like indexing a list.


Multiplication Between BlockArray and :class:`.LinearOperator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.Operator` and :class:`.LinearOperator` classes are designed
to work on instances of :class:`.BlockArray` in addition to instances of
:obj:`~jax.numpy.DeviceArray`. For example


   ::

      >>> x, key = scico.random.randn((3, 4))
      >>> A_1 = scico.linop.Identity(x.shape)
      >>> A_1.shape  # array -> array
      ((3, 4), (3, 4))

      >>> A_2 = scico.linop.FiniteDifference(x.shape)
      >>> A_2.shape  # array -> BlockArray
      (((2, 4), (3, 3)), (3, 4))

      >>> diag = snp.blockarray([np.array(1.0), np.array(2.0)])
      >>> A_3 = scico.linop.Diagonal(diag, input_shape=(A_2.output_shape))
      >>> A_3.shape  # BlockArray -> BlockArray
      (((2, 4), (3, 3)), ((2, 4), (3, 3)))
