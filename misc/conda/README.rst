Conda Installation Scripts
==========================

These scripts are intended to faciliate the installation of `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ and an environment with all SCICO requirements:

- ``install_conda.sh``:  Install miniconda
- ``make_conda_env.sh``:  Create a conda environment with all SCICO requirements

For usage details, run the scripts with the ``-h`` flag, e.g. ``./install_conda.sh -h``.


Example Usage
-------------

To install miniconda in ``/opt/conda`` do

::

   ./install_conda.sh -y /opt/conda


To create a conda environment called ``scico`` with Python version 3.12 and without GPU support

::

   ./make_conda_env.sh -y -p 3.12 -e scico


To include GPU support, follow the `jax installation instructions <https://github.com/google/jax#pip-installation-gpu-cuda>`__ after
running this script and activating the environment created by it.


Caveats
-------

These scripts should function correctly out-of-the-box on a standard Linux installation. (If you find that this is not the case, please create a GitHub issue, providing details of the Linux variant and version.)

While these scripts are supported under OSX (MacOS), there are some caveats:

- Required utilities ``realpath`` and ``gsed`` (GNU sed) must be installed via MacPorts or some other 3rd party package management system.
- Installation of jaxlib with GPU capabilities is not supported.
- While ``make_conda_env.sh`` installs ``matplotlib``, it does not attempt to resolve the `additional complications <https://matplotlib.org/faq/osx_framework.html>`_ in using a conda installed matplotlib under OSX.
