Installing SCICO
================

SCICO requires Python version 3.8 or later. It has been tested on Linux and macOS, but is not currently supported on Windows due to the limited support for ``jaxlib`` on Windows. However, Windows users can use SCICO via the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_. Guides exist for using WSL with `CPU only <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and with
`GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.


From PyPI
---------

.. warning::
   Installation from PyPI is not currently recommended since the current
   package available from PyPI is a pre-release testing version. The
   instructions directly below are intended for post-release documentation.

The simplest way to install the most recent release of SCICO from
`PyPI <https://pypi.python.org/pypi/scico/>`_ is

   ::

      pip install scico


From GitHub
-----------

SCICO can be downloaded from the `GitHub repo <https://github.com/lanl/scico>`_. Note that, since the SCICO repo has a submodule, it should be cloned via the command

::

   git clone --recurse-submodules git@github.com:lanl/scico.git

Install using the commands

::

   cd scico
   pip install -r requirements.txt
   pip install -e .



GPU Support
-----------

The instructions above install a CPU-only version of SCICO. To install a version with GPU support:

1. Follow the CPU only instructions, above

2. Identify which version of jaxlib was installed

   ::

      pip list | grep jaxlib

3. Install the same version of jaxlib, but with GPU support.
   For help with this, see `JAX with GPU support <https://github.com/google/jax#installation>`_.
   The command will be something like

   ::

      pip install --upgrade "jaxlib==0.1.70+cuda110" -f https://storage.googleapis.com/jax-releases/jax_releases.html



Additional Dependencies
-----------------------

We use the `ASTRA Toolbox <https://www.astra-toolbox.com/>`_ for tomographic projectors. We currently require the development version of ASTRA, as suggested by the package maintainers.

The development version of ASTRA can be installed using conda:

::

   conda install -c astra-toolbox/label/dev astra-toolbox

Alternatively, it can be `built from source <https://www.astra-toolbox.com/docs/install.html#for-python>`_.

We also support the `Super-Voxel Model-Based Iterative Reconstruction <https://svmbir.readthedocs.io/en/latest/>`_ package as an alternative tomographic projector. Since this package can be installed via ``pip``, it is
included in the list of package dependencies (``requirements.txt``), and need
not be separately installed.


For Developers
--------------

For installing a version of SCICO suitable for development,
see the instructions in :ref:`scico_dev_contributing`.


Building Documentation
----------------------

The documentation can be built from the respository root directory by doing

::

   python setup.py build_sphinx

Alternatively:

1. Navigate to the docs directory ``docs/``

2. Install dependencies

   ::

      pip install -r docs_requirements.txt

3. Build documentation

   ::

      make html
