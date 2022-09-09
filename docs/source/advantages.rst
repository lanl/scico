Why SCICO?
==========


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



Advantages of JAX-based Design
------------------------------

The vast majority of scientific computing packages in Python are based
on `NumPy <https://numpy.org/>`__ and `SciPy <https://scipy.org/>`__.
SCICO, in contrast, is based on
`JAX <https://jax.readthedocs.io/en/latest/>`__, which provides most of
the same features, but with the addition of automatic differentiation,
GPU support, and just-in-time (JIT) compilation. (The availability
of these features in SCICO is subject to some :ref:`caveats <non_jax_dep>`.) SCICO users and developers are advised to become familiar with the `differences between JAX and NumPy. <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_.

While recent advances in automatic differentiation have primarily been
driven by its important role in deep learning, it is also invaluable in
a functional minimization framework such as SCICO. The most obvious
advantage is allowing the use of gradient-based minimization methods
without the need for tedious mathematical derivation of an expression
for the gradient. Equally valuable, though, is the ability to
automatically compute the adjoint operator of a linear operator, the
manual derivation of which is often time-consuming.

GPU support and JIT compilation both offer the potential for significant
code acceleration, with the speed gains that can be obtained depending
on the algorithm/function to be executed. In many cases, a speed
improvement by an order of magnitude or more can be obtained by running
the same code on a GPU rather than a CPU, and similar speed gains can
sometimes also be obtained via JIT compilation.

The figure below shows timing results obtained on a compute server
with an Intel Xeon Gold 6230 CPU and NVIDIA GeForce RTX 2080 Ti
GPU. It is interesting to note that for :class:`.FiniteDifference` the
GPU provides no acceleration, while JIT provides more than an order of
magnitude of speed improvement on both CPU and GPU. For :class:`.DFT`
and :class:`.Convolve`, significant JIT acceleration is limited to the
GPU, which also provides significant acceleration over the CPU.


.. image:: /figures/jax-timing.png
     :align: center
     :width: 95%
     :alt: Timing results for SCICO operators on CPU and GPU with and without JIT


Related Packages
----------------

Many elements of SCICO are partially available in other packages. We
briefly review them here, highlighting some of the main differences with
SCICO.

`GlobalBioIm <https://biomedical-imaging-group.github.io/GlobalBioIm/>`__
is similar in structure to SCICO (and a major inspiration for SCICO),
providing linear operators and solvers for inverse problems in imaging.
However, it is written in MATLAB and is thus not usable in a completely
free environment. It also lacks the automatic adjoint calculation and
simple GPU support offered by SCICO.

`PyLops <https://pylops.readthedocs.io>`__ provides a linear operator
class and many built-in linear operators. These operators are compatible
with many `SciPy <https://scipy.org/>`__ solvers. GPU support is
provided via `CuPy <https://cupy.dev>`__, which has the disadvantage
that switching for a CPU to GPU requires code changes, unlike SCICO and
`JAX <https://jax.readthedocs.io/en/latest/>`__. SCICO is more focused
on computational imaging that PyLops and has several specialized
operators that PyLops does not.

`Pycsou <https://matthieumeo.github.io/pycsou/html/index>`__, like
SCICO, is a Python project inspired by GlobalBioIm. Since it is based on
PyLops, it shares the disadvantages with respect to SCICO of that
project.

`ODL <https://odlgroup.github.io/odl/>`__ provides a variety of
operators and related infrastructure for prototyping of inverse
problems. It is built on top of
`NumPy <https://numpy.org/>`__/`SciPy <https://scipy.org/>`__, and does
not support any of the advanced features of
`JAX <https://jax.readthedocs.io/en/latest/>`__.

`ProxImaL <http://www.proximal-lang.org/en/latest/>`__ is a Python
package for image optimization problems. Like SCICO and many of the
other projects listed here, problems are specified by combining objects
representing, operators, functionals, and solvers. It does not support
any of the advanced features of
`JAX <https://jax.readthedocs.io/en/latest/>`__.

`ProxMin <https://github.com/pmelchior/proxmin>`__ provides a set of
proximal optimization algorithms for minimizing non-smooth functionals.
It is built on top of
`NumPy <https://numpy.org/>`__/`SciPy <https://scipy.org/>`__, and does
not support any of the advanced features of
`JAX <https://jax.readthedocs.io/en/latest/>`__ (however, an open issue
suggests that `JAX <https://jax.readthedocs.io/en/latest/>`__
compatibility is planned).

`CVXPY <https://www.cvxpy.org>`__ provides a flexible language for
defining optimization problems and a wide selection of solvers, but has
limited support for matrix-free methods.

Other related projects that may be of interest include:

-  `ToMoBAR <https://github.com/dkazanc/ToMoBAR>`__
-  `CCPi-Regularisation Toolkit <https://github.com/vais-ral/CCPi-Regularisation-Toolkit>`__
-  `SPORCO <https://github.com/lanl/sporco>`__
-  `SigPy <https://github.com/mikgroup/sigpy>`__
-  `MIRT <https://github.com/JeffFessler/MIRT.jl>`__
-  `BART <http://mrirecon.github.io/bart/>`__
