SCICO Usage Examples
====================

This directory contains usage examples for the SCICO package. The primary form of these examples is the Python scripts in the directory ``scripts``. A corresponding set of Jupyter notebooks, in the directory ``notebooks``, is auto-generated from these usage example scripts.


Building Notebooks
------------------

The scripts for building Jupyter notebooks from the source example scripts are currently only supported under Linux. All scripts described below should be run from this directory, i.e. ``[repo root]/examples``.


The procedure for adding a adding a new notebook to the documentation is:

1. Add an entry for the source file in ``scripts/index.rst``. Note that a script that is not listed in this index will not be converted into a notebook.

2. Run ``makeindex.py`` to update the example scripts README file, the notebook index file, and the examples index in the docs.

3. Run ``makejnb.py`` to build the new notebook, as well as any other notebooks that are out of date with respect to their source scripts, as determined by the respective file timestamps.

4. Add and commit the new script, the ``scripts/index.rst`` script index file, the auto-generated ``scripts/README.rst`` file and ``docs/source/examples.rst`` index file, and the new or updated notebooks and the auto-generated ``notebooks/index.ipynb`` file in the notebooks directory (following the submodule handling procedure as described in the developer docs).


The procedure for rebuilding notebook(s) after the source file(s) have been modified is:

1. Run ``makejnb.py`` to build the new notebook, as well as any other notebooks that are out of date with respect to their source scripts, as determined by the respective file timestamps. Note that timestamps for files retrieved from version control may not be meaningful for this purpose. In such cases, ``touch`` the relevant source scripts to force updating on the next run of ``makejnb.py``.

2. Add and commit the modified script(s), and the updated notebooks (following the submodule handling procedure as described in the developer docs).



Management Utilities
--------------------

A number of files in this directory assist in the mangement of the usage examples:

`examples_requirements.txt <examples_requirements.txt>`_
   Requirements file (as used by ``pip``) listing additional dependencies for running the usage example scripts.

`notebooks_requirements.txt <examples_requirements.txt>`_
   Requirements file (as used by ``pip``) listing additional dependencies for building the Jupyter notebooks from the usage example scripts.

`makejnb.py <makejnb.py>`_
   An alternative to the makefile for updating the auto-generated Jupyter notebooks. Notebooks are executed in parallel using the ``ray`` package.

`makeindex.py <makeindex.py>`_
   Auto-generate the docs example index ``docs/source/examples.rst`` from the example scripts index ``scripts/index.rst``.

`Makefile <Makefile>`_
   A makefile allowing use of the command ``make`` to update auto-generated Jupyter notebooks. Run as ``make no-execute=true`` to update the notebooks without executing them. Use of `makejnb.py` rather than this makefile is recommended.

`pytojnb.sh <pytojnb.sh>`_
   Low-level python to Jupyter notebook conversion script. Used by both the makefile and `makejnb.py <makejnb.py>`_.

`scriptcheck.sh <scriptcheck.sh>`_
   Run all example scripts with a reduced number of iterations as a rapid check that they are functioning correctly.
