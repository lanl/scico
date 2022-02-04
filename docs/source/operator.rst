Operators
=========
An operator is a map from :math:`\mathbb{R}^n` or :math:`\mathbb{C}^n`
to :math:`\mathbb{R}^m` or :math:`\mathbb{C}^m`.
In SCICO, operators are primarily used to represent imaging systems
and provide regularization.
SCICO operators are represented by instances of the :class:`.Operator` class.


SCICO :class:`.Operator` objects extend the notion of "shape" and "size" from the usual NumPy ``ndarray`` class.
Each :class:`.Operator` object has an ``input_shape`` and ``output_shape``; these shapes can be either tuples or a tuple of tuples
(in the case of a :class:`.BlockArray`).
The ``matrix_shape`` attribute describes the shape of the :class:`.LinearOperator` if it were to act on vectorized, or flattened, inputs.


For example, consider a two dimensional array :math:`\mb{x} \in \mathbb{R}^{n \times m}`.
We compute the discrete differences of :math:`\mb{x}` in the horizontal and vertical directions,
generating two new arrays: :math:`\mb{x}_h \in \mathbb{R}^{n \times (m-1)}` and :math:`\mb{x}_v \in
\mathbb{R}^{(n-1) \times m}`. We represent this linear operator by
:math:`\mb{A} : \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times (m-1)} \otimes \mathbb{R}^{(n-1) \times m}`.
In SCICO, this linear operator will return a :class:`.BlockArray` with the horizontal and vertical differences
stored as blocks. Letting :math:`y = \mb{A} x`, we have ``y.shape = ((n, m-1), (n-1, m))``
and

   ::

      A.input_shape = (n, m)
      A.output_shape = ((n, m-1), (n-1, m)], (n, m))
      A.shape = ( ((n, m-1), (n-1, m)), (n, m))   # (output_shape, input_shape)
      A.input_size = n*m
      A.output_size = n*(n-1)*m*(m-1)
      A.matrix_shape = (n*(n-1)*m*(m-1), n*m)    # (output_size, input_size)


Operator Calculus
-----------------
SCICO supports a variety of operator calculus rules,
allowing new operators to be defined in terms of old ones.
The following table summarizes the available operations.

+----------------+-----------------+
| Operation      |  Result         |
+----------------+-----------------+
| ``(A+B)(x)``   | ``A(x) + B(x)`` |
+----------------+-----------------+
| ``(A-B)(x)``   | ``A(x) - B(x)`` |
+----------------+-----------------+
| ``(c * A)(x)`` | ``c * A(x)``    |
+----------------+-----------------+
| ``(A/c)(x)``   | ``A(x)/c``      |
+----------------+-----------------+
| ``(-A)(x)``    | ``-A(x)``       |
+----------------+-----------------+
| ``A(B)(x)``    | ``A(B(x))``     |
+----------------+-----------------+
| ``A(B)``       | ``Operator``    |
+----------------+-----------------+


Defining A New Operator
-----------------------
To define a new operator,
pass a callable to the :class:`.Operator` constructor:

  ::

      A = Operator(input_shape=(32,),
                   eval_fn = lambda x: 2 * x)


Or use subclassing:

  ::

     >>> from scico.operator import Operator
     >>> class MyOp(Operator):
     ...
     ...     def _eval(self, x):
     ...         return 2 * x

     >>> A = MyOp(input_shape=(32,))

At a minimum, the ``_eval`` function must be overridden.
If either ``output_shape`` or ``output_dtype`` are unspecified, they are determined by evaluating
the operator on an input of appropriate shape and dtype.


Linear Operators
================

Linear operators are those for which

  .. math::

    H(a \mb{x} + b \mb{y}) = a H(\mb{x}) + b H(\mb{y}).

SCICO represents linear operators as instances of the class :class:`.LinearOperator`.
While finite-dimensional linear operators
can always be associated with a matrix,
it is often useful to represent them in a matrix-free manner.
Most of SCICO's linear operators are implemented matrix-free.



Using A LinearOperator
----------------------

We implement two ways to evaluate a :class:`.LinearOperator`. The first is using standard
callable syntax: ``A(x)``. The second mimics the NumPy matrix multiplication
syntax: ``A @ x``. Both methods perform shape and type checks to validate the
input before ultimately either calling `A._eval` or generating a new :class:`.LinearOperator`.

For linear operators that map real-valued inputs to real-valued outputs, there are two ways to apply the adjoint:
``A.adj(y)`` and ``A.T @ y``.

For complex-valued linear operators, there are three ways to apply the adjoint ``A.adj(y)``, ``A.H @ y``, and ``A.conj().T @ y``.
Note that in this case, ``A.T`` returns the non-conjugated transpose of the LinearOperator.

While the cost of evaluating the linear operator is virtually identical for ``A(x)`` and ``A @ x``,
the ``A.H`` and ``A.conj().T`` methods are somewhat slower; especially the latter. This is because two
intermediate linear operators must be created before the function is evaluated.  Evaluating ``A.conj().T @ y``
is equivalent to:

::

  def f(y):
    B = A.conj()  # New LinearOperator #1
    C = B.T       # New LinearOperator #2
    return C @ y

**Note**: the speed differences between these methods vanish if applied inside of a jit-ed function.
For instance:

::

   f = jax.jit(lambda x:  A.conj().T @ x)


+------------------+-----------------+
|  Public Method   |  Private Method |
+------------------+-----------------+
|  ``__call__``    |  ``._eval``     |
+------------------+-----------------+
|  ``adj``         |  ``._adj``      |
+------------------+-----------------+
|  ``gram``        |  ``._gram``     |
+------------------+-----------------+

The public methods perform shape and type checking to validate the input before either calling the corresponding
private method or returning a composite LinearOperator.


Linear Operator Calculus
------------------------
SCICO supports several linear operator calculus rules.
Given
``A`` and ``B`` of class :class:`.LinearOperator` and of appropriate shape,
``x`` an array of appropriate shape,
``c`` a scalar, and
``O`` an :class:`.Operator`,
we have

+----------------+----------------------------+
| Operation      |  Result                    |
+----------------+----------------------------+
| ``(A+B)(x)``   | ``A(x) + B(x)``            |
+----------------+----------------------------+
| ``(A-B)(x)``   | ``A(x) - B(x)``            |
+----------------+----------------------------+
| ``(c * A)(x)`` | ``c * A(x)``               |
+----------------+----------------------------+
| ``(A/c)(x)``   | ``A(x)/c``                 |
+----------------+----------------------------+
| ``(-A)(x)``    | ``-A(x)``                  |
+----------------+----------------------------+
| ``(A@B)(x)``   | ``A@B@x``                  |
+----------------+----------------------------+
| ``A @ B``      | ``ComposedLinearOperator`` |
+----------------+----------------------------+
| ``A @ O``      | ``Operator``               |
+----------------+----------------------------+
| ``O(A)``       | ``Operator``               |
+----------------+----------------------------+



Defining A New Linear Operator
------------------------------

To define a new linear operator,
pass a callable to the :class:`.LinearOperator` constructor

  ::

     >>> from scico.linop import LinearOperator
     >>> A = LinearOperator(input_shape=(32,),
     ...       eval_fn = lambda x: 2 * x)


Or, use subclassing:

  ::

     >>> class MyLinearOperator(LinearOperator):
     ...    def _eval(self, x):
     ...        return 2 * x

     >>> A = MyLinearOperator(input_shape=(32,))

At a minimum, the ``_eval`` method must be overridden.
If the ``_adj`` method is not overriden, the adjoint is determined using :func:`scico.linear_adjoint`.
If either ``output_shape`` or ``output_dtype`` are unspecified, they are determined by evaluating
the Operator on an input of appropriate shape and dtype.


🔪 Sharp Edges 🔪
------------------

Strict Types in Adjoint
***********************

SCICO silently promotes real types to complex types in forward application,
but enforces strict type checking in the adjoint.
This is due to the strict type-safe nature of jax adjoints.


LinearOperators From External Code
**********************************

External code may be wrapped as a subclass of :class:`.Operator` or :class:`.LinearOperator`
and used in SCICO optimization routines;
however this process can be complicated and error-prone.
As a starting point,
look at the source for :class:`.radon_svmbir.ParallelBeamProjector` or :class:`.radon_astra.ParallelBeamProjector`
and the JAX documentation for the
`vector-jacobian product <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff>`_
and `ustom VJP rules <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_.
