---
title: 'Scientific Computational Imaging Code (SCICO)'
tags:
  - Python
  - computational imaging
  - scientific imaging
  - inverse problems
header-includes:
  - \usepackage{amsmath}
  - \newcommand{\mb}[1]{\mathbf{#1}}
  - \DeclareMathOperator*{\argmin}{arg\,min}
authors:
  - name: Thilo Balke
    orcid: 0000-0002-1716-5923
    affiliation: "1, 2"
  - name: Fernando Davis
    affiliation: "1, 3"
  - name: Cristina Garcia-Cardona
    orcid: 0000-0002-5641-3491
    affiliation: 1
  - name: Soumendu Majee
    orcid: 0000-0002-8442-2897
    affiliation: 4
  - name: Michael McCann
    orcid: 0000-0001-7645-252X
    affiliation: 1
  - name: Luke Pfister
    orcid: 0000-0001-7485-5966
    affiliation: 1
  - name: Brendt Wohlberg
    orcid: 0000-0002-4767-1843
    affiliation: 1
affiliations:
 - name: Los Alamos National Laboratory
   index: 1
 - name: Purdue University
   index: 2
 - name: University of Puerto Rico-Mayaguez
   index: 3
 - name: Samsung Research America
   index: 4
date: 24 August 2022
bibliography: paper.bib
---

# Summary

Scientific Computational Imaging Code (SCICO) is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of [JAX](https://jax.readthedocs.io/en/latest/) rather than [NumPy](https://numpy.org/), enabling GPU/TPU acceleration, just-in-time compilation, and automatic gradient functionality,
which is used to automatically compute the adjoints of linear operators.
An example of how to solve a multi-channel tomography problem with SCICO is shown in \autoref{fig:flow_chart}. The SCICO source code is available from [GitHub](https://github.com/lanl/scico), and pre-built packages are available from [PyPI](https://github.com/lanl/scico). It has extensive [online documentation](https://scico.rtfd.io/), including API documentation and usage examples, which can be run online at [Google Colab](https://colab.research.google.com/github/lanl/scico-data/blob/colab/notebooks/index.ipynb) and [binder](https://mybinder.org/v2/gh/lanl/scico-data/binder?labpath=notebooks%2Findex.ipynb).

Community contributions, including bug reports, feature requests, and code contributions, are welcomed at [https://github.com/lanl/scico](https://github.com/lanl/scico).

![Solving a multi-channel tomography problem with SCICO.\label{fig:flow_chart}](figures/flow_chart.pdf){ width=100% }


# Statement of Need

In traditional imaging, the burden of image formation is placed on physical components, such as a lens, with the resulting image being taken from the sensor with minimal processing. In computational imaging, in contrast, the burden of image formation is shared with or shifted to computation, with the resulting image typically being very different from the measured data. Common examples of computational imaging include demosaicing in consumer cameras, computed tomography and magnetic resonance imaging in medicine, and synthetic aperture radar in remote sensing. This is an active and growing area of research, and many of these problems have common properties that could be supported by shared implementations of solution components.

The goal of SCICO is to provide a general research tool for computational imaging, with a particular focus on scientific imaging applications, which are particularly underrepresented in the existing range of open-source packages in this area. While a number of other packages overlap somewhat in functionality with SCICO, only a few support execution of the same code on both CPU and GPU devices, and we are not aware of any that support just-in-time compilation and automatic gradient computation, which is invaluable in computational imaging. SCICO provides all three of these valuable features by being built on top of [JAX](https://jax.readthedocs.io/en/latest/) rather than [NumPy](https://numpy.org/).


# Solving Imaging Inverse Problems in SCICO

SCICO provides a set of building blocks that can be used to express a wide variety of problems and their corresponding solutions. These building blocks include operators for representing the *forward model* of an imaging problem, functionals for representing *data fidelity* and *regularization* terms, and optimization algorithms for minimizing these functionals. The [online documentation](https://scico.rtfd.io/) includes a guide to the use of these components as well as numerous example scripts demonstrating their use in practice.



## Machine Learning

SCICO includes an implementation of the DnCNN denoiser [@zhang-2017-dncnn], which can be applied to other inverse problems via the *plug-and-play priors* (PPP) [@venkatakrishnan-2013-plugandplay2; @sreehari-2016-plug; @kamilov-2022-plug] framework (see \autoref{fig:deconv}). A number of other leading machine learning methods have been implemented, and are expected to be merged into the main SCICO GitHub branch in the near future.


![Image deconvolution via PPP with DnCNN denoiser.\label{fig:deconv}](figures/deconv_ppp_dncnn.pdf){ width=100% }



## Advantages of JAX-based Design

The vast majority of scientific computing packages in Python are based on
[NumPy](https://numpy.org/) and [SciPy](https://scipy.org/). SCICO, in contrast, is based on [JAX](https://jax.readthedocs.io/en/latest/), which provides most of the same features, but with the addition of automatic differentiation, GPU support, and just-in-time (JIT) compilation. In addition to its obvious application in  gradient-based minimization methods, automatic differentiation allows automatic computation of the adjoint operator of a linear operator, the manual derivation of which is often time-consuming.


# Acknowledgments

This work was supported by the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project number 20200061DR.


# References
