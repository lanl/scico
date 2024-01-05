.. _installing:

Installing SCICO
================

SCICO requires Python version 3.8 or later. (Version 3.10 is
recommended as it is the version under which SCICO has been most
thoroughly tested.) It is supported on both Linux and MacOS, but is
not currently supported on Windows due to the limited support for
``jaxlib`` on Windows. However, Windows users can use SCICO via the
`Windows Subsystem for Linux
<https://docs.microsoft.com/en-us/windows/wsl/about>`_ (WSL). Guides
exist for using WSL with `CPU only
<https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and
with `GPU support
<https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.

While not required, installation of SCICO and its dependencies within a `Conda <https://conda.io/projects/conda/en/latest/user-guide/index.html>`_ environment
is recommended. `Scripts <https://github.com/lanl/scico/tree/main/misc/conda>`_
are provided for creating a `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installation and an environment including all primary SCICO dependencies as well as dependencies
for usage example, testing, and building the documentation.


From PyPI
---------

The simplest way to install the most recent release of SCICO from
`PyPI <https://pypi.python.org/pypi/scico/>`_ is
::

   pip install scico

which will install SCICO and its primary dependencies. If the additional
dependencies for the example scripts are also desired, it can instead be
installed using
::

   pip install scico[examples]

Note, however, that since the ``astra-toolbox`` package available from
PyPI is not straightforward to install (it has numerous build requirements
that are not specified as package dependencies), it is recommended to
first install this package via conda
::

   conda install astra-toolbox



From conda-forge
----------------

SCICO can also be installed from `conda-forge <https://anaconda.org/conda-forge/scico>`_
::

  conda install -c conda-forge "scico>0.0.5"

where the version constraint is required to avoid installation of an old
package with broken dependencies. Note, however, that installation from conda forge is only straightforward for a Python 3.10 environment on Linux x64, due
to limitations of conda packages for some of the SCICO dependencies:

* There is no conda package for the secondary dependency ``tensorstore``
  under MacOS.
* In a Python 3.9 environment, a version of secondary dependency ``etils``
  that does not support Python 3.9 will be installed. This can be rectified
  by
  ::

     conda install etils=1.5.1
* Conda packages for dependency ``svmbir`` are not currently available for
  Python versions greater than 3.10. If an attempt is made to install SCICO
  via conda forge, an older package with some missing dependencies for the
  example scripts will be installed. If required, these dependencies
  (including ``svmbir``, which can be installed using ``pip``) will have to
  be manually installed.

The most recent SCICO conda forge package also includes dependencies for
the example scripts, except for ``bm3d``, ``bm4d``, and
``colour_demosaicing``, for which conda packages are not available. These
can be installed from PyPI
::

  pip install bm3d bm4d colour_demosaicing



From GitHub
-----------

SCICO can be downloaded from the `GitHub repo
<https://github.com/lanl/scico>`_. Note that, since the SCICO repo has
a submodule, it should be cloned via the command
::

   git clone --recurse-submodules git@github.com:lanl/scico.git

Install using the commands
::

   cd scico
   pip install -r requirements.txt
   pip install -e .



GPU Support
-----------

The instructions above install a CPU-only version of SCICO. To install
a version with GPU support:

1. Follow the CPU-only instructions, above

2. Install the version of jaxlib with GPU support, as described in the `JAX installation
   instructions  <https://jax.readthedocs.io/en/latest/installation.html>`_.
   In the simplest case, the appropriate command is
   ::

      pip install --upgrade "jax[cuda11]"

   for CUDA 11, or
   ::

      pip install --upgrade "jax[cuda12]"

   for CUDA 12, but it may be necessary to explicitly specify the
   ``jaxlib`` version if the most recent release is not yet supported
   by SCICO (as specified in the ``requirements.txt`` file), or if
   using a version of CUDA older than 11.4, or CuDNN older than 8.2,
   in which case the command would be of the form
   ::

      pip install --upgrade "jaxlib==0.4.2+cuda11.cudnn82" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   with appropriate substitution of ``jaxlib``, CUDA, and CuDNN version
   numbers.



Additional Dependencies
-----------------------

See :ref:`example_depend` for instructions on installing dependencies
related to the examples.


For Developers
--------------

See :ref:`scico_dev_contributing` for instructions on installing a
version of SCICO suitable for development.
