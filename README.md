[![Python \>= 3.8](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![Package License](https://img.shields.io/github/license/lanl/scico.svg)](https://github.com/lanl/scico/blob/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/scico/badge/?version=latest)](http://scico.readthedocs.io/en/latest/?badge=latest)
[![JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04722/status.svg)](https://doi.org/10.21105/joss.04722)\
[![Lint status](https://github.com/lanl/scico/actions/workflows/lint.yml/badge.svg)](https://github.com/lanl/scico/actions/workflows/lint.yml)
[![Test status](https://github.com/lanl/scico/actions/workflows/pytest_ubuntu.yml/badge.svg)](https://github.com/lanl/scico/actions/workflows/pytest_ubuntu.yml)
[![Test coverage](https://codecov.io/gh/lanl/scico/branch/main/graph/badge.svg?token=wQimmjnzFf)](https://codecov.io/gh/lanl/scico)
[![CodeFactor](https://www.codefactor.io/repository/github/lanl/scico/badge/main)](https://www.codefactor.io/repository/github/lanl/scico/overview/main)\
[![PyPI package version](https://badge.fury.io/py/scico.svg)](https://badge.fury.io/py/scico)
[![PyPI download statistics](https://static.pepy.tech/personalized-badge/scico?period=total&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/scico)
[![Conda Forge Release](https://img.shields.io/conda/vn/conda-forge/scico.svg)](https://anaconda.org/conda-forge/scico)
[![Conda Forge Downloads](https://img.shields.io/conda/dn/conda-forge/scico.svg)](https://anaconda.org/conda-forge/scico)\
[![View notebooks at nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/lanl/scico-data/tree/main/notebooks/index.ipynb)
[![Run notebooks on binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb)
[![Run notebooks on google colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lanl/scico-data/blob/colab/notebooks/index.ipynb)


# Scientific Computational Imaging Code (SCICO)

SCICO is a Python package for solving the inverse problems that arise in
scientific imaging applications. Its primary focus is providing methods
for solving ill-posed inverse problems by using an appropriate prior
model of the reconstruction space. SCICO includes a growing suite of
operators, cost functionals, regularizers, and optimization routines
that may be combined to solve a wide range of problems, and is designed
so that it is easy to add new building blocks. SCICO is built on top of
[JAX](https://github.com/google/jax), which provides features such as
automatic gradient calculation and GPU acceleration.

[Documentation](https://scico.rtfd.io/) is available online. If you use
this software for published work, please cite the corresponding [JOSS
Paper](https://doi.org/10.21105/joss.04722) (see bibtex entry
`balke-2022-scico` in `docs/source/references.bib`).


# Installation

The online documentation includes detailed
[installation instructions](https://scico.rtfd.io/en/latest/install.html).


# Usage Examples

Usage examples are available as Python scripts and Jupyter Notebooks.
Example scripts are located in `examples/scripts`. The corresponding
Jupyter Notebooks are provided in the
[scico-data](https://github.com/lanl/scico-data) submodule and symlinked
to `examples/notebooks`. They are also viewable on
[GitHub](https://github.com/lanl/scico-data/tree/main/notebooks) or
[nbviewer](https://nbviewer.jupyter.org/github/lanl/scico-data/tree/main/notebooks/index.ipynb),
and can be run online on
[binder](https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb)
or
[google colab](https://colab.research.google.com/github/lanl/scico-data/blob/colab/notebooks/index.ipynb).


# License

SCICO is distributed as open-source software under a BSD 3-Clause
License (see the `LICENSE` file for details).

LANL open source approval reference C20091.

\(c\) 2020-2026. Triad National Security, LLC. All rights reserved. This
program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National
Nuclear Security Administration. All rights in the program are reserved
by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government has
granted for itself and others acting on its behalf a nonexclusive,
paid-up, irrevocable worldwide license in this material to reproduce,
prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.
