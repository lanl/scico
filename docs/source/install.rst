Installing SCICO
================

SCICO requires Python version 3.8 or later. It has been tested on Linux and macOS, but is not currently supported on Windows due to the limited support for ``jaxlib`` on Windows. However, Windows users can use SCICO via the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_. Guides exist for using WSL with `CPU only <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and with
`GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.


From PyPI
---------

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

      pip install --upgrade "jaxlib==0.3.0+cuda11.cudnn805" -f https://storage.googleapis.com/jax-releases/jax_releases.html



Additional Dependencies
-----------------------

For instructions on installing dependencies related to the examples please see :ref:`example_dependencies`.


For Developers
--------------

For installing a version of SCICO suitable for development,
see the instructions in :ref:`scico_dev_contributing`.
