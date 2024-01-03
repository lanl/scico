.. _installing:

Installing SCICO
================

SCICO requires Python version 3.8 or later. (Version 3.10 is
recommended as it is the version under which SCICO has been most
thoroughly tested.) It is supported on both Linux and macOS, but is
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
are provided for creating a `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installation and an environment including all SCICO dependencies and optional support
packages.


From PyPI
---------

The simplest way to install the most recent release of SCICO from
`PyPI <https://pypi.python.org/pypi/scico/>`_ is

::

   pip install scico


From conda-forge
----------------

SCICO can also be installed from `conda-forge <https://anaconda.org/conda-forge/scico>`_

::

  conda install -c conda-forge scico

Note, however, that some additional action will typically be required to address
problems in the conda packages for secondary dependencies. For example, on Linux x64
and Python 3.10,

::

  conda install "chex>=0.1.7" "optax>=0.1.5"

is required to avoid bugs in the earlier versions that are installed by default, and
Python 3.9 also requires

::

  conda install etils==1.5.1

since some more recent versions are not compatible with Python versions earlier
than 3.10.


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
   in which case the command would be of the form ::

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
