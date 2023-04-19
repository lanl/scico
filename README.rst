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

.. image:: https://github.com/lanl/scico/actions/workflows/pytest_ubuntu.yml/badge.svg
   :target: https://github.com/lanl/scico/actions/workflows/pytest_ubuntu.yml
   :alt: Test status

.. image:: https://codecov.io/gh/lanl/scico/branch/main/graph/badge.svg?token=wQimmjnzFf
   :target: https://codecov.io/gh/lanl/scico
   :alt: Test coverage

.. image:: https://www.codefactor.io/repository/github/lanl/scico/badge/main
   :target: https://www.codefactor.io/repository/github/lanl/scico/overview/main
   :alt: CodeFactor

.. image:: https://badge.fury.io/py/scico.svg
   :target: https://badge.fury.io/py/scico
   :alt: PyPI package version

.. image:: https://static.pepy.tech/personalized-badge/scico?period=month&left_color=grey&right_color=brightgreen
   :target: https://pepy.tech/project/scico
   :alt: PyPI download statistics

.. image:: https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg
   :target: https://nbviewer.jupyter.org/github/lanl/scico-data/tree/main/notebooks/index.ipynb
   :alt: View notebooks at nbviewer

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb
   :alt: Run notebooks on binder

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/lanl/scico-data/blob/colab/notebooks/index.ipynb
   :alt: Run notebooks on google colab

.. image:: https://joss.theoj.org/papers/10.21105/joss.04722/status.svg
   :target: https://doi.org/10.21105/joss.04722
   :alt: JOSS paper



Scientific Computational Imaging Code (SCICO)
=============================================

SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of `JAX <https://github.com/google/jax>`_, which provides features such as automatic gradient calculation and GPU acceleration.

`Documentation <https://scico.rtfd.io/>`_ is available online. If you use this software for published work, please cite the corresponding `JOSS Paper <https://doi.org/10.21105/joss.04722>`_ (see bibtex entry ``balke-2022-scico`` in ``docs/source/references.bib``).


Installation
============

See the `online documentation <https://scico.rtfd.io/en/latest/install.html>`_ for installation instructions.


Usage Examples
==============

Usage examples are available as Python scripts and Jupyter Notebooks. Example scripts are located in ``examples/scripts``. The corresponding Jupyter Notebooks are provided in the `scico-data <https://github.com/lanl/scico-data>`_ submodule and symlinked to ``examples/notebooks``. They are also viewable on `GitHub <https://github.com/lanl/scico-data/tree/main/notebooks>`_ or `nbviewer <https://nbviewer.jupyter.org/github/lanl/scico-data/tree/main/notebooks/index.ipynb>`_, or can be run online by `binder <https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb>`_.


License
=======

SCICO is distributed as open-source software under a BSD 3-Clause License (see the ``LICENSE`` file for details).

LANL open source approval reference C20091.

(c) 2020-2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government has granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
