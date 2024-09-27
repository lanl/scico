.. _installing:

Installing SCICO
================

SCICO requires Python version 3.8 or later. (Version 3.12 is
recommended as it is the version under which SCICO is tested in GitHub
continuous integration, and since the most recent versions of JAX require
version 3.10 or later.) SCICO is supported on both Linux and
MacOS, but is not currently supported on Windows due to the limited
support for ``jaxlib`` on Windows. However, Windows users can use
SCICO via the `Windows Subsystem for Linux
<https://docs.microsoft.com/en-us/windows/wsl/about>`_ (WSL). Guides
exist for using WSL with
`CPU only <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_
and with
`GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.

While not required, installation of SCICO and its dependencies within a
`Conda <https://conda.io/projects/conda/en/latest/user-guide/index.html>`_
environment is recommended.
`Scripts <https://github.com/lanl/scico/tree/main/misc/conda>`_
are provided for creating a
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
installation and an environment including all primary SCICO dependencies
as well as dependencies for usage example, testing, and building the
documentation.


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
package with broken dependencies.

Note, however, that installation from conda forge is only possible on a Linux
platform since there is no conda package for the secondary dependency
``tensorstore`` under MacOS. There are also complications on Linux platforms
with Python versions 3.9 or earlier due to the automatic installation of a
version of secondary dependency ``etils`` that does not support Python versions
earlier than 3.10. This can be rectified by
::

  conda install etils=1.5.1

The most recent SCICO conda forge package also includes dependencies for
the example scripts, except for ``bm3d``, ``bm4d``, and
``colour_demosaicing``, for which conda packages are not available. These
can be installed from PyPI
::

  pip install bm3d bm4d colour_demosaicing



From GitHub
-----------

The development version of SCICO can be downloaded from the `GitHub repo
<https://github.com/lanl/scico>`_. Note that, since the SCICO repo has
a submodule, it should be cloned via the command
::

   git clone --recurse-submodules git@github.com:lanl/scico.git

Install using the commands
::

   cd scico
   pip install -r requirements.txt
   pip install -e .


If a clone of the SCICO repository is not needed, it is simpler to
install directly using ``pip``
::

   pip install git+https://github.com/lanl/scico



GPU Support
-----------

The instructions above install a CPU-only version of SCICO. To install
a version with GPU support:

1. Follow the CPU-only instructions, above

2. Install the version of jaxlib with GPU support, as described in the `JAX installation
   instructions  <https://jax.readthedocs.io/en/latest/installation.html>`_.
   In the simplest case, the appropriate command is
   ::

      pip install --upgrade "jax[cuda12]"

   for CUDA 12, but it may be necessary to explicitly specify the
   ``jaxlib`` version if the most recent release is not yet supported
   by SCICO (as specified in the ``requirements.txt`` file).


The script
`misc/gpu/envinfo.py <https://github.com/lanl/scico/blob/main/misc/gpu/envinfo.py>`_
in the source distribution is provided as an aid to debugging GPU support
issues. The script
`misc/gpu/availgpu.py <https://github.com/lanl/scico/blob/main/misc/gpu/availgpu.py>`_
can be used to automatically recommend a setting of the CUDA_VISIBLE_DEVICES
environment variable that excludes GPUs that are already in use.



Additional Dependencies
-----------------------

See :ref:`example_depend` for instructions on installing dependencies
related to the examples.


For Developers
--------------

See :ref:`scico_dev_contributing` for instructions on installing a
version of SCICO suitable for development.
