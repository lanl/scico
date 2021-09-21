Functionals and Losses
======================

A functional maps an :code:`array-like` to a scalar; abstractly, a functional is
a mapping from :math:`\mathbb{R}^n` or :math:`\mathbb{C}^n` to :math:`\mathbb{R}`.

A functional ``f`` can have three core operations.

* Evaluation
   - ``f(x)`` returns the value of the functional evaluated at :math:`\mb{x}`.
   - A functional that can be evaluated has the attribute ``f.has_eval == True``.
   - Not all functionals can be evaluated:  see `Plug-and-Play`_.

* Gradient
   - ``f.grad(x)`` returns the gradient of the functional evaluated at :math:`\mb{x}`.
   - Calculated using JAX reverse-mode automatic differentiation, exposed through :func:`scico.grad`.
   - A functional that is smooth has the attribute ``f.is_smooth == True``
   - NOTE:  The gradient of a functional ``f`` can be evaluated even if ``f.is_smooth == False``.  All that is required is that the functional can be evaluated, ``f.has_eval == True``.  However, the result may not be a valid gradient (or subgradient) for all inputs :math:`\mb{x}`.


* Proximal operator
   - The proximal operator of a functional :math:`f : \mathbb{R}^n \to \mathbb{R}` is the mapping
     :math:`\mathrm{prox}_f : \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n` defined as

     .. math::

      \mathrm{prox}_f (v, \lambda) =  \argmin_x \lambda f(x) + \frac{1}{2} \norm{v - x}_2^2.


Plug-and-Play
-------------

* For the Plug-and-Play framework :cite:`sreehari-2016-plug`, we encapsulate denoisers/CNNs in a Functional object that **cannot be evaluated**.
* Only the proximal operator is exposed.


Proximal Calculus
-----------------

We support a limited subset of proximal calculus rules.

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

Note that we have made no assumptions regarding homogeneity of ``f``; rather, only that the proximal method of ``f`` is given in the parameterized form :math:`\mathrm{prox}_{c f}`.

In SCICO, multiplying a :class:`.Functional` by a scalar will return a :class:`.ScaledFunctional`.  This :class:`.ScaledFunctional` retains the ``has_eval, is_smooth``, and ``has_prox`` attributes from the original :class:`.Functional`, but the proximal method is modified to accomodate the additional scalar.


Separable Functionals
*********************

A separable functional :math:`f : \mathbb{C}^N \to \mathbb{R}` can be written as the sum
of functionals :math:`f_i : \mathbb{C}^{N_i} \to \mathbb{R}` with :math:`\sum_i N_i = N`.  In particular,

    .. math::
       f(\mb{x}) = f(\mb{x}_1, \dots, \mb{x}_N) = f_1(\mb{x}_1) + \dots + f_N(\mb{x}_N)

The proximal operator of a separable :math:`f` can be written in terms of the proximal operators of the :math:`f_i` (see Theorem 6.6 of :cite:`beck-2017-first`):

    .. math::
        \mathrm{prox}_f(\mb{x}, \lambda)
        =
        \begin{bmatrix}
          \mathrm{prox}_{f_1}(\mb{x}_1, \lambda) \\
          \vdots \\
          \mathrm{prox}_{f_N}(\mb{x}_N, \lambda) \\
        \end{bmatrix}

Separable Functionals are implemented in the :class:`.SeparableFunctional` class.  Separable functionals naturally accept :class:`.BlockArray` inputs and return the prox as a :class:`.BlockArray`.



Adding New Functionals
----------------------

1. Inherit from base functional
2. Set ``has_eval``, ``is_smooth``, and ``has_prox`` flags.
3. Add ``_eval`` and ``prox`` methods, as necessary.

   ::

      class MyFunctional(scico.functional.Functional):

          has_eval = True
          is_smooth = False
          has_prox = True

          def _eval(self, x: JaxArray) -> float:
               return snp.sum(x)

          def prox(self, x: JaxArray, lam : float) -> JaxArray:
               return x - lam


Losses
------
