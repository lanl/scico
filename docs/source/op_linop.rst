Operators
=========

.. todo::
  * link to PyLops, scipy abstract linear operators
  * Document :func:`._wrap_mul_div_scalar`, :func:`._wrap_sum`

Matrix-free representation of operators.

* Operator:  generic operator
* LinearOperator:  a linear map

SCICO :class:`.LinearOperator` is a linear operator that is designed to work with :class:`.BlockArray`.



Consider a two dimensional array :math:`\mb{x} \in \mathbb{R}^{n \times m}`.
We compute the discrete differences of :math:`\mb{x}` in the horizontal and vertical directions,
generating two new arrays: :math:`\mb{x}_h \in \mathbb{R}^{n \times (m-1)}` and :math:`\mb{x}_v \in
\mathbb{R}^{(n-1) \times m}`.  We represent this linear operator by
:math:`\mb{A} : \mathbb{R}^{n \times m} \to \mathbb{R}^{n \times (m-1)} \otimes \mathbb{R}^{(n-1) \times m}`.

In SCICO, this linear operator will return a :class:`.BlockArray` with the horizontal and vertical differences
stored as blocks.  Letting :math:`y = \mb{A} x`, we have ``y.shape = ((n, m-1), (n-1, m))``.


SCICO :class:`.LinearOperator` objects extend the notion of "shape" and "size" from the usual NumPy ``ndarray`` class.
Each :class:`.LinearOperator` object has an ``input_shape`` and ``output_shape``; these shapes can be either tuples or a tuple of tuples
(in the case of a :class:`.BlockArray`).   For the finite difference operator above,

   ::

      A.input_shape = (n, m)
      A.output_shape = ((n, m-1), (n-1, m)], (n, m))
      A.shape = ( ((n, m-1), (n-1, m)), (n, m))   # (output_shape, input_shape)
      A.input_size = n*m
      A.output_size = n*(n-1)*m*(m-1)
      A.matrix_shape = (n*(n-1)*m*(m-1), n*m)    # (output_size, input_size)


The ``matrix_shape`` attribute describes the shape of the :class:`.LinearOperator` if it were to act on vectorized, or flattened,
inputs.




Using An Operator
-----------------
Call: ``A(x)``


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

Pass a callable to the Operator constructor:

  ::

      A = Operator(input_shape=(32,),
                   eval_fn = lambda x: 2 * x)


Or via subclassing:

At a minimum, the ``_eval`` function must be overridden:

  ::

     >>> from scico.operator import Operator
     >>> class MyOp(Operator):
     ...
     ...     def _eval(self, x):
     ...         return 2 * x

     >>> A = MyOp(input_shape=(32,))



* If either ``output_shape`` or ``output_dtype`` are unspecified, they are determined by evaluating
  the Operator on an input of appropriate shape and dtype.



Linear Operators
================

Specialization of :class:`.Operator` to linear operators.





Operations using LinearOperator
-------------------------------

``A`` and ``B`` are :class:`.LinearOperator` of the same shape,
``x`` is an array of appropriate shape,
``c`` is a scalar, and
``O`` is :class:`.Operator`

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



Using A LinearOperator
----------------------

Evaluating a LinearOperator


We implement two ways to evaluate the LinearOperator. The first is using standard
callable syntax: ``A(x)``. The second mimics the NumPy matrix multiplication
syntax: ``A @ x``. Both methods perform shape and type checks to validate the
input before ultimately either calling `A._eval` or generating a new LinearOperator.

For LinearOperators that map real-valued inputs to real-valued outputs, there are two ways to apply the adjoint:
``A.adj(y)`` and ``A.T @ y``.

For complex-valued LinearOperators, there are three ways to apply the adjoint ``A.adj(y)``, ``A.H @ y``, and ``A.conj().T @ y``.
Note that in this case, ``A.T`` returns the non-conjugated transpose of the LinearOperator.

While the cost of evaluating the LinearOperator is virtually identical for ``A(x)`` and ``A @ x``,
the ``A.H`` and ``A.conj().T`` methods are somewhat slower; especially the latter. This is because two
intermediate LinearOperators must be created before the function is evaluated.   Evaluating ``A.conj().T @ y``
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

Jit Options
-----------

.. todo::

   details


Defining A New LinearOperator
-----------------------------

Pass a callable to the LinearOperator constructor

  ::

     >>> from scico.linop import LinearOperator
     >>> A = LinearOperator(input_shape=(32,),
     ...       eval_fn = lambda x: 2 * x)


Subclassing:

At a minimum, the ``_eval`` method must be overridden:

  ::

     >>> class MyLinearOperator(LinearOperator):
     ...    def _eval(self, x):
     ...        return 2 * x

     >>> A = MyLinearOperator(input_shape=(32,))


* If the ``_adj`` method is not overriden, the adjoint is determined using :func:`scico.linear_adjoint`.
* If either ``output_shape`` or ``output_dtype`` are unspecified, they are determined by evaluating
  the Operator on an input of appropriate shape and dtype.



🔪 Sharp Edges 🔪
------------------

Strict types in adjoint
***********************

.. todo::

   We silently promote real->complex types in forward application, but have strict type checking in the adjoint.
   This is due to the strict type-safe nature of jax adjoints


LinearOperators from External Code
**********************************

.. todo::

  Fill this out!

* Pain point:  adjoint and defining VJP for gradient computations
  For example, might want to compute :math:`\nabla_x \norm{y - A x}_2^2` where :math:`A` is not a pure  jax function
* Discuss VJP framework
* Can't use ``jax.linear_transpose``; must use VJP framework to determine adjoint
* Complexities for complex functions

`Vector-Jacobian Product <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff>`_
  .. math::
    \begin{aligned}
    &f : \mathbb{R}^n \to \mathbb{R}^m \\
    &\partial f(x) : \mathbb{R}^n \times \mathbb{R}^m \\
    &v \in \mathbb{R}^m \\
    &\mathrm{vjp}_f (x, v) \to v \partial f(x)
    \end{aligned}

  .. math::
    \begin{aligned}
    &A \in \mathbb{R}^{m \times n} \\
    &f(x) = A x \\
    & \partial f(x) = A \\
    &\mathrm{vjp}_f (x, v) \to v \partial f = v A = A^T v
    \end{aligned}


`Custom VJP rules <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_
