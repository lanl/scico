.. _optimizer:

Optimization Algorithms
=======================

ADMM
----

The Alternating Direction Method of Multipliers (ADMM) :cite:`glowinski-1975-approximation` :cite:`gabay-1976-dual`
is an algorithm for minimizing problems of the form

.. math::
   :label: eq:admm_prob

   \argmin_{\mb{x}, \mb{z}} \; f(\mb{x}) + g(\mb{z}) \; \text{such that}
   \; \acute{A} \mb{x} + \acute{B} \mb{z} = \mb{c} \;,

where :math:`f` and :math:`g` are convex (but not necessarily smooth)
functions, :math:`\acute{A}` and :math:`\acute{B}` are linear operators,
and :math:`\mb{c}` is a constant vector. (For a thorough introduction and
overview, see :cite:`boyd-2010-distributed`.)

The SCICO ADMM solver, :class:`.ADMM`, solves problems of the form

.. math::
   \argmin_{\mb{x}} \; f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}) \;,

where :math:`f` and the :math:`g_i` are instances of :class:`.Functional`,
and the :math:`C_i` are :class:`.LinearOperator`, by defining

.. math::
   g(\mb{z}) = \sum_{i=1}^N g_i(\mb{z}_i) \qquad \mb{z}_i = C_i \mb{x}

in :eq:`eq:admm_prob`, corresponding to defining

.. math::
  \acute{A} = \left( \begin{array}{c} C_0 \\ C_1 \\ C_2 \\
	   \vdots \end{array} \right)  \quad
  \acute{B} = \left( \begin{array}{cccc}
	      -I & 0 & 0 & \ldots \\
	      0 & -I & 0 & \ldots \\
	      0 &  0  & -I & \ldots \\
	      \vdots & \vdots & \vdots & \ddots
	      \end{array} \right) \quad
  \mb{z} = \left( \begin{array}{c} \mb{z}_0 \\ \mb{z}_1 \\ \mb{z}_2 \\
	   \vdots \end{array} \right)  \quad
  \mb{c} = \left( \begin{array}{c} 0 \\ 0 \\ 0 \\
	   \vdots \end{array} \right) \;.

In :class:`.ADMM`, :math:`f` is a :class:`.Functional`, typically a :class:`.Loss`, corresponding to the forward model of an imaging problem,
and the :math:`g_i` are :class:`.Functional`, typically corresponding to a
regularization term or constraint. Each of the :math:`g_i` must have a
proximal operator defined. It is also possible to set ``f = None``, which corresponds to defining :math:`f = 0`, i.e. the zero function.

The :math:`\mb{x}`-update in SCICO's ADMM formulation may admit
a specialized solver, depending on the problem structure;
SCICO has several such solvers built in and allows easy specification of new ones.
The :math:`\mb{x}`-update is

    .. math::

        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;.

By default, this problem is solved using :func:`.solver.minimize`, which wraps
SciPy's `minimize` function.
When :math:`f` takes the form :math:`\norm{\mb{A} \mb{x} - \mb{y}}^2_W`,
the user may specify that the :class:`.admm.LinearSubproblemSolver` be used,
which solves the problem using the conjugate gradient method.
As a further specialization, if :math:`\mb{A}` and all the :math:`C_i` s are circulant
(i.e., diagonalizable in a Fourier basis),
the :class:`.admm.CircularConvolveSolver` may be used to
efficiently solve the problem in the Fourier domain.
For more details of these solvers and how to specify them,
see the API reference page for :class:`scico.admm`.



PGM
---

The Proximal Gradient Method (PGM) :cite:`daubechies-2004-iterative`
:cite:`beck-2010-gradient` and Accelerated Proximal Gradient Method (AcceleratedPGM) :cite:`beck-2009-fast` are algorithms for minimizing
problems of the form

.. math::
   \argmin_{\mb{x}} f(\mb{x}) + g(\mb{x})

where :math:`g` is convex and :math:`f` is smooth and convex. The
corresponding SCICO solvers are :class:`PGM` and :class:`AcceleratedPGM`
respectively. In most cases :class:`AcceleratedPGM` is expected to provide
faster convergence. In both of these classes, :math:`f` and :math:`g` are
both of type :class:`.Functional`, where :math:`f` must be differentiable,
and :math:`g` must have a proximal operator defined.

While ADMM provides significantly more flexibility than PGM, and often
converges faster, the latter is preferred when solving the ADMM
:math:`\mb{x}`-step is computationally expensive.

.. _sec-stepsize-classes:


Step Size Options for PGM
^^^^^^^^^^^^^^^^^^^^^^^^^

The step size for the gradient descent in :class:`PGM` can be adapted via
Barzilai-Borwein methods (also called spectral methods) and iterative
line search methods.

The base class from which all algorithms for adapting the step size are
derived is :class:`PGMStepSize`. The derived classes should override the
method :meth:`update`. Note that the estimation is done in terms
of the reciprocal of the step size (:math:`L` in equations).


* :class:`BBStepSize`

  This implements the step size adaptation based on the Barzilai-Borwein
  method :cite:`barzilai-1988-stepsize`. The step size :math:`\alpha` is
  estimated as

  .. math::
     \mb{\Delta x} = \mb{x}_k - \mb{x}_{k-1} \; \\
     \mb{\Delta g} = \nabla f(\mb{x}_k) - \nabla f (\mb{x}_{k-1}) \; \\
     \alpha = \frac{\mb{\Delta x}^T \mb{\Delta g}}{\mb{\Delta g}^T
     \mb{\Delta g}} \;\;.

  Since the PGM solver uses the reciprocal of the step size, the value
  :math:`L = 1 / \alpha` is returned.


* :class:`AdaptiveBBStepSize`

  This implements the adaptive Barzilai-Borwein method as introduced in
  :cite:`zhou-2006-adaptive`. The adaptive step size rule computes

  .. math::
     \mb{\Delta x} = \mb{x}_k - \mb{x}_{k-1} \; \\
     \mb{\Delta g} = \nabla f(\mb{x}_k) - \nabla f (\mb{x}_{k-1}) \; \\
     \alpha^{\mathrm{BB1}} = \frac{\mb{\Delta x}^T \mb{\Delta x}}
     {\mb{\Delta x}^T \mb{\Delta g}} \; \\
     \alpha^{\mathrm{BB2}} = \frac{\mb{\Delta x}^T \mb{\Delta g}}
     {\mb{\Delta g}^T \mb{\Delta g}} \;\;.

  The determination of the new step size is made via the rule

  .. math::
     \alpha = \left\{ \begin{matrix} \alpha^{\mathrm{BB2}} \;, &
     \mathrm{~if~} \alpha^{\mathrm{BB2}} / \alpha^{\mathrm{BB1}}
     < \kappa \; \\
     \alpha^{\mathrm{BB1}} \;, & \mathrm{~otherwise} \end{matrix}
     \right . \;\;,

  with :math:`\kappa \in (0, 1)`.

  Since the PGM solver uses the reciprocal of the step size, the value
  :math:`L = 1 / \alpha` is returned.


* :class:`LineSearchStepSize`

  This implements the line search strategy described in :cite:`beck-2009-fast`.
  This strategy estimates :math:`L` such that
  :math:`f(\mb{x}) \leq \hat{f}_{L}(\mb{x})` is satisfied with
  :math:`\hat{f}_{L}` a quadratic approximation to :math:`f` defined as

  .. math::
     \hat{f}_{L}(\mb{x}, \mb{y}) = f(\mb{y}) + \nabla f(\mb{y})^H
     (\mb{x} - \mb{y}) + \frac{L}{2} \left\| \mb{x} - \mb{y}
     \right\|_2^2 \;\;,

  with :math:`\mb{x}` the potential new update and :math:`\mb{y}` the
  current solution or current extrapolation (if using :class:`AcceleratedPGM`).


* :class:`RobustLineSearchStepSize`

  This implements the robust line search strategy described in
  :cite:`florea-2017-robust`. This strategy estimates :math:`L` such that
  :math:`f(\mb{x}) \leq \hat{f}_{L}(\mb{x})` is satisfied with
  :math:`\hat{f}_{L}` a quadratic approximation to :math:`f` defined as

  .. math::
     \hat{f}_{L}(\mb{x}, \mb{y}) = f(\mb{y}) + \nabla f(\mb{y})^H
     (\mb{x} - \mb{y}) + \frac{L}{2} \left\| \mb{x} - \mb{y} \right\|_2^2 \;\;,

  with :math:`\mb{x}` the potential new update and :math:`\mb{y}` the
  auxiliary extrapolation state. Note that this should only be used
  with :class:`AcceleratedPGM`.
