Installing SCICO
================

..
   The simplest way to install the most recent release of SCICO from
   `PyPI <https://pypi.python.org/pypi/scico/>`_ is

   ::

       pip install scico


Once it has been released, installation of SCICO via ``pip`` and ``conda`` will be supported. At the moment, SCICO should be downloaded from the `GitHub repo <https://github.com/lanl/scico>`_. Note that, since the SCICO repo has a submodule, it should be cloned via the command

::

   git clone --recurse-submodules git@github.com:lanl/scico.git

Once the dependencies have been installed, as described below, install SCICO itself in editable form

::

   cd scico
   pip install -e .


Installing Dependencies
=======================

Automatic Installation
----------------------

The recommended way to install SCICO and its dependencies is via `conda <https://docs.conda.io/en/latest/>`_ using the scripts in ``misc/conda``:

  - ``install_conda.sh``: install ``miniconda`` (needed if conda is not already installed on your system)
  - ``conda_env.sh``: install a ``conda`` environment with all SCICO dependencies


Manual Installation
-------------------

SCICO and its dependencies may also be installed manually.  Installation depends on whether SCICO is intended to run on CPUs only, or whether GPU support is also required.

CPU Only
########

1. Clone the repository

2. Navigate to the repository root directory

   ::

      cd scico

3. Install dependencies

   ::

      pip install -r requirements.txt



GPU Enabled
###########

By default, ``pip install -r requirements.txt`` will install a CPU-only version of SCICO. To install a version with GPU support:

1. Clone the repository

2. Navigate to the repository root directory

   ::

      cd scico

3. Install `JAX with GPU support <https://github.com/google/jax#installation>`_.

4. Install remaining dependencies

   ::

      pip install -r requirements.txt


Additional Dependencies for Tomography
######################################

We use the `ASTRA Toolbox <https://www.astra-toolbox.com/>`_ for tomographic projectors. We currently require the development version of ASTRA, as suggested by the package maintainers.

The development version of ASTRA can be installed using conda:

::

   conda install -c astra-toolbox/label/dev astra-toolbox

Alternatively, it can be `built from source <https://www.astra-toolbox.com/docs/install.html#for-python>`_.

We also support the `Super-Voxel Model-Based Iterative Reconstruction <https://svmbir.readthedocs.io/en/latest/>`_ package as an alternative tomographic projector.

This package can be installed using pip:

::

   pip install svmbir


SCICO on Windows
----------------

We do not support using SCICO on Windows. Our advice for users on Windows is to use Linux inside a virtual machine. Microsoft's `WSL <https://docs.microsoft.com/en-us/windows/wsl/about>`_ is one such solution. Guides exist for using WSL with the `CPU only <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and with `GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.


For Developers
==============

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
