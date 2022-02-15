.. image:: https://img.shields.io/badge/python-3.8+-green.svg
    :target: https://www.python.org/
    :alt: Python >= 3.8

.. image:: https://img.shields.io/github/license/lanl/scico.svg
    :target: https://github.com/lanl/scico/blob/main/LICENSE
    :alt: Package License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style

.. image:: https://readthedocs.org/projects/scico/badge/?version=latest
    :target: http://scico.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/lanl/scico/actions/workflows/lint.yml/badge.svg
   :target: https://github.com/lanl/scico/actions/workflows/lint.yml
   :alt: Lint status

.. image:: https://github.com/lanl/scico/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/lanl/scico/actions/workflows/pytest.yml
   :alt: Test status

.. image:: https://codecov.io/gh/lanl/scico/branch/main/graph/badge.svg?token=wQimmjnzFf
   :target: https://codecov.io/gh/lanl/scico
   :alt: Test coverage

.. image:: https://www.codefactor.io/repository/github/lanl/scico/badge/main
   :target: https://www.codefactor.io/repository/github/lanl/scico/overview/main
   :alt: CodeFactor

.. image:: https://badge.fury.io/py/scico.svg
   :target: https://badge.fury.io/py/scico
   :alt: Current PyPI package version


Scientific Computational Imaging COde (SCICO)
=============================================

SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of `JAX <https://github.com/google/jax>`_, which provides features such as automatic gradient calculation and GPU acceleration.

`Documentation is available online <https://scico.rtfd.io/>`_. If you use this software for published work, please cite bibtex entry ``pfister-2021-scico`` in ``docs/source/references.bib``.


Installation
============

See the `online documentation <https://scico.rtfd.io/en/latest/install.html>`_ for installation instructions.


Usage Examples
==============

Usage examples are available as Python scripts and Jupyter Notebooks. Example scripts are located in ``examples/scripts``. The corresponding Jupyter Notebooks are provided in the `scico-data <https://github.com/lanl/scico-data>`_ submodule and symlinked to ``examples/notebooks``. They are also viewable on `GitHub <https://github.com/lanl/scico-data/tree/main/notebooks>`_ or `nbviewer <https://nbviewer.jupyter.org/github/lanl/scico-data/tree/main/notebooks/index.ipynb>`_.


License
=======

SCICO is distributed as open-source software under a BSD 3-Clause License (see the ``LICENSE`` file for details).

LANL open source approval reference C20091.

(c) 2020-2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government has granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
