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



`Scientific Computational Imaging Code (SCICO) <https://github.com/lanl/scico>`__ is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization algorithms that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. When solving a problem, these components are combined in a way that makes code for optimization routines look like the pseudocode in scientific papers. SCICO is built on top of `JAX <https://jax.readthedocs.io/en/latest/>`__ rather than `NumPy <https://numpy.org/>`__, enabling GPU/TPU acceleration, just-in-time compilation, and automatic gradient functionality, which is used to automatically compute the adjoints of linear operators. An example of how to solve a multi-channel tomography problem with SCICO is shown in the figure below.


.. image:: /figures/scico-tomo-overview.png
     :align: center
     :width: 95%
     :alt: Solving a multi-channel tomography problem with SCICO.

|

The SCICO source code is available from `GitHub
<https://github.com/lanl/scico>`__, and pre-built packages are
available from `PyPI <https://github.com/lanl/scico>`__. (Detailed
instructions for installing SCICO are available in :ref:`installing`.)
It has extensive `online documentation <https://scico.rtfd.io/>`__,
including :doc:`API documentation <_autosummary/scico>` and
:ref:`usage examples <example_notebooks>`, which can be run online at
`Google Colab
<https://colab.research.google.com/github/lanl/scico-data/blob/colab/notebooks/index.ipynb>`__
and `binder
<https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb>`__.


If you use this library for published work, please cite :cite:`scico-2022` (see bibtex entry ``scico-2022`` in `docs/source/references.bib <https://github.com/lanl/scico/blob/main/docs/source/references.bib>`_ in the source distribution).



Contributing
------------

Bug reports, feature requests, and general suggestions are welcome,
and should be submitted via the `GitHub issue system
<https://github.com/lanl/scico/issues>`__. More substantial
contributions are also :ref:`welcome <scico_dev_contributing>`.



License
-------

SCICO is distributed as open-source software under a BSD 3-Clause License (see the `LICENSE <https://github.com/lanl/scico/blob/master/LICENSE>`__ file for details). LANL open source approval reference C20091.

© 2020-2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.  All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration.  The Government has granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
