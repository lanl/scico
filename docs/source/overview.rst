Overview
========


.. raw:: html

    <style type='text/css'>
    div.document ul blockquote {
       margin-bottom: 8px !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    div.document li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    </style>



`SCICO <https://github.com/lanl/scico>`__ is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks.

SCICO's main advantages are:

   - Operators in SCICO are defined using a simple, NumPy-like syntax, yet automatically benefit from GPU/TPU acceleration and `just-in-time compilation <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions>`__.
   - SCICO can compute the adjoint of linear operators automatically, which saves time when defining new operators.
   - SCICO's operator calculus makes code for optimization routines look like the pseudocode in scientific papers.
   - SCICO provides state-of-the-art optimization routines, including projected gradients and the Alternating Direction Method of Multipliers, with the flexibility of plug-and-play priors including BM3D and DnCNN denoisers.


If you use this library for published work, please cite :cite:`pfister-2021-scico` (see bibtex entry ``pfister-2021-scico`` in `docs/source/references.bib <https://github.com/lanl/scico/blob/main/docs/source/references.bib>`_ in the source distribution).


Anatomy of an Inverse Problem
-----------------------------

SCICO is designed to solve problems of the form

.. math::

   \argmin_{\mb{x}} \; f(\mb{x}) + \sum_{r=1}^R g_r(C_r (\mb{x})),

with :math:`\mb{x} \in  \mathbb{R}^{n}` representing the reconstruction, e.g.,  a 1D, 2D, 3D, or 3D+time image,
:math:`C_r:  \mathbb{R}^{n} \to \mathbb{R}^{m_r}` being regularization operators,
and :math:`f: \mathbb{R}^{n} \to \mathbb{R}` and :math:`g_r: \mathbb{R}^{m_r} \to \mathbb{R}` being functionals associated with the data fidelity and regularization, respectively.


In a typical computational imaging problem, we have a `forward model`
that models the data acquisition process.  We denote this forward
model by the (possibly nonlinear) operator :math:`A`.

We want to find :math:`\mb{x}` such that :math:`\mb{y} \approx A(\mb{x})`
in some appropriate sense. This discrepency is measured in our data fidelity, or `loss`, function

.. math::
   f(\mb{x}) = L(A(\mb{x}), \mb{y}) \,,

where :math:`L` is a `functional` that maps a vector (or array) into
a scalar.  A common choice is :math:`f(\mb{x}) = \norm{\mb{y} - A(\mb{x})}_2^2`.
The SCICO :class:`.Loss` object encapsulates the data
:math:`\mb{y}`, the forward model :math:`A`, and the functional
:math:`L` in a single object.  A library of functionals and losses
are implemented in :mod:`.functional` and :mod:`.loss`,
respectively.

Note that :math:`f(\mb{x})` can often be interpreted as the negative
log likelihood of the candidate :math:`\mb{x}`, given the measurements
:math:`\mb{y}` and an underlying noise model.

The functionals :math:`g_r(C_r (\mb{x}))` are regularization functionals.  The :math:`C_r` are operators,
usually linear operators.   Together, these terms encourage solutions that arex
"simple" in some sense.  A popular choice is :math:`g = \norm{ \cdot }_1` and :math:`C` to be a :class:`.FiniteDifferece`
linear operator; this combination promotes piecewise smooth solutions to the inverse problem.


Usage Examples
--------------

Usage examples are available as Python scripts and Jupyter Notebooks. Example
scripts are located in ``examples/scripts``. The corresponding Jupyter Notebooks
are provided in the ``scico-data`` submodule and symlinked to
``examples/notebooks``. They are also viewable on `GitHub
<https://github.com/lanl/scico-data/tree/main/notebooks>`_ and in the
documentation :ref:`example_notebooks`.


Related Projects
----------------

The SCICO library is inspired by the `GlobalBiolm <https://github.com/Biomedical-Imaging-Group/GlobalBioIm>`_ MATLAB package,
which provides a similar object-oriented design for solving computational imaging problems. `Pycsou <https://github.com/matthieumeo/pycsou>`_ is a similar
Python library for inverse problems that is also inspired by GlobalBioIm.

A key advantage of SCICO over these libraries is the usage of JAX, which provides automatic hardware acceleration, automatic differentiation,
and automatic adjoint calculations.  Moreover, as JAX is a machine learning library, state of the art Plug-and-Play regularizers such as DnCNN
can specified, trained, and implemented in the same software package.


Other related projects that may be of interest include:

   - `ODL <https://github.com/odlgroup/odl>`_
   - `PyLops <https://pylops.readthedocs.io/en/latest/>`_
   - `ProxImaL <https://github.com/comp-imaging/ProxImaL>`_
   - `ProxMin <https://github.com/pmelchior/proxmin>`_
   - `ToMoBAR <https://github.com/dkazanc/ToMoBAR>`_
   - `CCPi-Regularisation Toolkit <https://github.com/vais-ral/CCPi-Regularisation-Toolkit>`_
   - `SPORCO <https://github.com/lanl/sporco>`_
   - `SigPy <https://github.com/mikgroup/sigpy>`_
   - `MIRT <https://github.com/JeffFessler/MIRT.jl>`_
   - `BART <http://mrirecon.github.io/bart/>`_


Contributing
------------

Bug reports, feature requests, and general suggestions are welcome, and should be submitted via the `github issue system <https://github.com/lanl/scico/issues>`__. More substantial contributions are also welcome; please see :ref:`scico_dev_contributing`.



License
-------

SCICO is distributed as open-source software under a BSD 3-Clause License  (see the `LICENSE <https://github.com/lanl/scico/blob/master/LICENSE>`__ file for details). LANL open source approval reference C20091.

© 2020-2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.  Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government has granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
