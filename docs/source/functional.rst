Functionals
===========

.. raw:: html

    <style type='text/css'>
    div.document ul blockquote {
       margin-bottom: 8px !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    div.document ul > li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    section {
      padding-bottom: 1em;
    }
    ul {
      margin-bottom: 1em;
    }
    </style>

A functional is
a mapping from :math:`\mathbb{R}^n` or :math:`\mathbb{C}^n` to :math:`\mathbb{R}`.
In SCICO, functionals are
primarily used to represent a cost to be minimized
and are represented by instances of the :class:`.Functional` class.
An instance of :class:`.Functional`, ``f``, may provide three core operations.

* Evaluation
   - ``f(x)`` returns the value of the functional
     evaluated at the point ``x``.
   - A functional that can be evaluated
     has the attribute ``f.has_eval == True``.
   - Not all functionals can be evaluated:  see `Plug-and-Play`_.
* Gradient
   - ``f.grad(x)`` returns the gradient of the functional evaluated at ``x``.
   - Gradients are calculated using JAX reverse-mode automatic differentiation,
     exposed through :func:`scico.grad`.
   - *Note:*  The gradient of a functional ``f`` can be evaluated even if that functional is not smooth.
     All that is required is that the functional can be evaluated, ``f.has_eval == True``.
     However, the result may not be a valid gradient (or subgradient) for all inputs.
* Proximal operator
   - ``f.prox(v, lam)`` returns the result of the scaled proximal
     operator of ``f``, i.e., the proximal operator of ``lambda x:
     lam * f(x)``, evaluated at the point ``v``.
   - The proximal operator of a functional :math:`f : \mathbb{R}^n \to
     \mathbb{R}` is the mapping :math:`\mathrm{prox}_f : \mathbb{R}^n
     \to \mathbb{R}^n` defined as

     .. math::
      \mathrm{prox}_f (\mb{v}) =  \argmin_{\mb{x}} f(\mb{x}) +
      \frac{1}{2} \norm{\mb{v} - \mb{x}}_2^2\;.


Plug-and-Play
-------------

For the plug-and-play framework :cite:`sreehari-2016-plug`,
we encapsulate generic denoisers including CNNs
in :class:`.Functional` objects that **cannot be evaluated**.
The denoiser is applied via the the proximal operator.
For examples, see :ref:`example_notebooks`.


Proximal Calculus
-----------------

We support a limited subset of proximal calculus rules:


Scaled Functionals
******************

Given a scalar ``c`` and a functional ``f`` with a defined proximal method, we can
determine the proximal method of ``c * f`` as

  .. math::

     \begin{align}
      \mathrm{prox}_{c f} (v, \lambda) &=  \argmin_x \lambda (c f)(x) + \frac{1}{2} \norm{v - x}_2^2  \\
      &=  \argmin_x (\lambda c) f(x) + \frac{1}{2} \norm{v - x}_2^2 \\
      &= \mathrm{prox}_{f} (v, c \lambda)
      \end{align}

Note that we have made no assumptions regarding homogeneity of ``f``;
rather, only that the proximal method of ``f`` is given
in the parameterized form :math:`\mathrm{prox}_{c f}`.

In SCICO, multiplying a :class:`.Functional` by a scalar
will return a :class:`.ScaledFunctional`.
This :class:`.ScaledFunctional` retains the ``has_eval`` and ``has_prox`` attributes
from the original :class:`.Functional`,
but the proximal method is modified to accomodate the additional scalar.


Separable Functionals
*********************

A separable functional :math:`f : \mathbb{C}^N \to \mathbb{R}` can be written as the sum
of functionals :math:`f_i : \mathbb{C}^{N_i} \to \mathbb{R}` with :math:`\sum_i N_i = N`. In particular,

    .. math::
       f(\mb{x}) = f(\mb{x}_1, \dots, \mb{x}_N) = f_1(\mb{x}_1) + \dots + f_N(\mb{x}_N)

The proximal operator of a separable :math:`f` can be written
in terms of the proximal operators of the :math:`f_i`
(see Theorem 6.6 of :cite:`beck-2017-first`):

    .. math::
        \mathrm{prox}_f(\mb{x}, \lambda)
        =
        \begin{bmatrix}
          \mathrm{prox}_{f_1}(\mb{x}_1, \lambda) \\
          \vdots \\
          \mathrm{prox}_{f_N}(\mb{x}_N, \lambda) \\
        \end{bmatrix}

Separable Functionals are implemented in the :class:`.SeparableFunctional` class. Separable functionals naturally accept :class:`.BlockArray` inputs and return the prox as a :class:`.BlockArray`.



Adding New Functionals
----------------------
To add a new functional,
create a class which

1. inherits from base :class:`.Functional`;
2. has ``has_eval`` and ``has_prox`` flags;
3. has ``_eval`` and ``prox`` methods, as necessary.

For example,

   ::

      class MyFunctional(scico.functional.Functional):

          has_eval = True
          has_prox = True

          def _eval(self, x: JaxArray) -> float:
               return snp.sum(x)

          def prox(self, x: JaxArray, lam : float) -> JaxArray:
               return x - lam


Losses
------

In SCICO, a loss is a special type of functional

  .. math::
     f(\mb{x}) = \alpha l( \mb{y}, A(\mb{x}) )

where :math:`\alpha` is a scaling parameter,
:math:`l` is a functional,
:math:`\mb{y}` is a set of measurements,
and :math:`A` is an operator.
SCICO uses the class :class:`.Loss` to represent losses.
Loss functionals commonly arrise in the context of solving
inverse problems in scientific imaging,
where they are used to represent the mismatch
between predicted measurements :math:`A(\mb{x})`
and actual ones :math:`\mb{y}`.
