************
Installation
************


Installing SCICO
================

..
   The simplest way to install the most recent release of SCICO from
   `PyPI <https://pypi.python.org/pypi/scico/>`_ is

   ::

       pip install scico



Installing Dependencies
-----------------------

For detailed instructions on how to install a CPU-only version see `Installing a Development Version`_.



GPU Enabled
###########

By default, ``pip install -r requirements.txt`` will install a CPU-only version of SCICO. To install a version with GPU support:

1. Follow the CPU Only instructions, above

2. Identify which version of jaxlib was installed

   ::

      pip list | grep jaxlib

3. Install the same version of jaxlib, but with GPU support.
   For help with this, see `JAX with GPU support <https://github.com/google/jax#installation>`_.
   The command will be something like

   ::

      pip install --upgrade "jaxlib==0.1.70+cuda110" -f https://storage.googleapis.com/jax-releases/jax_releases.html



Additional Dependencies for Tomography
######################################

We use the `ASTRA Toolbox <https://www.astra-toolbox.com/>`_ for tomographic projectors. We currently require the development version of ASTRA, as suggested by the package maintainers.

The development version of ASTRA can be installed using conda:

::

   conda install -c astra-toolbox/label/dev astra-toolbox

Alternatively, it can be `built from source <https://www.astra-toolbox.com/docs/install.html#for-python>`_.

We also support the `Super-Voxel Model-Based Iterative Reconstruction <https://svmbir.readthedocs.io/en/latest/>`_ package as an alternative tomographic projector. Since this package can be installed via pip, it is
included in the list of package dependencies (`requirements.txt`), and need
not be separately installed.



SCICO on Windows
----------------

We do not support using SCICO on Windows. Our advice for users on Windows is to use Linux inside a virtual machine. Microsoft's `WSL <https://docs.microsoft.com/en-us/windows/wsl/about>`_ is one such solution. Guides exist for using WSL with the `CPU only <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and with `GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.



For Developers
--------------

The SCICO project uses the `Black <https://black.readthedocs.io/en/stable/>`_
and `isort <https://pypi.org/project/isort/>`_ code formatting utilities.
You can set up a `pre-commit hook <https://pre-commit.com>`_ to ensure any modified code passes format check before it is committed to the development repo.

In the cloned repository root directory, set up the pre-commit hook:

::

   pre-commit install



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
