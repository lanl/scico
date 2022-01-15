SCICO Usage Examples
====================

This directory contains usage examples for the SCICO package. The primary form of these examples is the Python scripts in the directory ``scripts``. A corresponding set of Jupyter notebooks, in the directory ``notebooks``, is auto-generated from these usage example scripts.


Building Notebooks
------------------

The scripts for building Jupyter notebooks from the source example scripts are currently only supported under Linux. All scripts described below should be run from this directory, i.e. ``[repo root]/examples``.


Running on a GPU
^^^^^^^^^^^^^^^^

Since some of the examples require a considerable amount of memory (``deconv_microscopy_tv_admm.py`` and ``deconv_microscopy_allchn_tv_admm.py`` in particular), it is recommended to set the following environment variables prior to building the notebooks:

::

  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export XLA_PYTHON_CLIENT_PREALLOCATE=false


Running on a CPU
^^^^^^^^^^^^^^^^

If a GPU is not available, or if the available GPU does not have sufficient memory to build the notebooks, set the environment variable

::

  JAX_PLATFORM_NAME=cpu

to run on the CPU instead.


Building Specific Examples
--------------------------

To build or rebuild notebooks for specific examples, the example script names can be specified on the command line, e.g.

::

  python makenotebooks.py ct_astra_pcg.py ct_astra_tv_admm.py

When rebuilding notebooks for examples that themselves make use of ``ray``
for parallelization (e.g. ``deconv_microscopy_allchn_tv_admm.py``), it is recommended to specify serial notebook execution, as in

::

  python makenotebooks.py --no-ray deconv_microscopy_allchn_tv_admm.py


Building All Examples
---------------------

By default, ``makenotebooks.py`` only rebuilds notebooks that are out of date with respect to their corresponding example scripts, as determined by their respective file timestamps. However, timestamps for files retrieved from version control may not be meaningful for this purpose. To rebuild all examples, the following commands (assuming that GPUs are available) are recommended:

::

  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  touch scripts/*.py

  python makenotebooks.py --no-ray deconv_microscopy_tv_admm.py deconv_microscopy_allchn_tv_admm.py

  python makenotebooks.py


Updating Notebooks in the Repo
------------------------------

The recommended procedure for rebuilding notebooks for inclusion in the ``data`` submodule is:

1. Add and commit the modified script(s).

2. Rebuild the notebooks as described above.

2. Add and commit the updated notebooks following the submodule handling procedure described in the developer docs.


Adding a New Notebook
---------------------

The procedure for adding a adding a new notebook is:

1. Add an entry for the source file in ``scripts/index.rst``. Note that a script that is not listed in this index will not be converted into a notebook.

2. Run ``makeindex.py`` to update the example scripts README file, the notebook index file, and the examples index in the docs.

3. Build the corresponding notebook following the instructions above.

4. Add and commit the new script, the ``scripts/index.rst`` script index file, the auto-generated ``scripts/README.rst`` file and ``docs/source/examples.rst`` index file, and the new or updated notebooks and the auto-generated ``notebooks/index.ipynb`` file in the notebooks directory, following the submodule handling procedure as described in the developer docs.



Management Utilities
--------------------

A number of files in this directory assist in the mangement of the usage examples:

`examples_requirements.txt <examples_requirements.txt>`_
   Requirements file (as used by ``pip``) listing additional dependencies for running the usage example scripts.

`notebooks_requirements.txt <notebooks_requirements.txt>`_
   Requirements file (as used by ``pip``) listing additional dependencies for building the Jupyter notebooks from the usage example scripts.

`makenotebooks.py <makenotebooks.py>`_
   Auto-generate Jupyter notebooks from the example scripts.

`makeindex.py <makeindex.py>`_
   Auto-generate the docs example index ``docs/source/examples.rst`` from the example scripts index ``scripts/index.rst``.

`scriptcheck.sh <scriptcheck.sh>`_
   Run all example scripts with a reduced number of iterations as a rapid check that they are functioning correctly.
