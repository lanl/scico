SCICO Usage Examples
====================

This directory contains usage examples for the SCICO package. The primary form of these examples is the Python scripts in the directory ``scripts``. A corresponding set of Jupyter notebooks, in the directory ``notebooks``, is auto-generated from these usage example scripts.


Management Utilities
--------------------

A number of additional files in this directory assist in the mangement of the usage examples:

`examples_requirements.txt <examples_requirements.txt>`_
   Requirements file (as used by ``pip``) listing additional dependencies for the usage examples.

`Makefile <Makefile>`_
   A makefile allowing use of the command ``make`` to update auto-generated Jupyter notebooks. Run as ``make no-execute=true`` to update the notebooks without executing them.

`makejnb.py <makejnb.py>`_
   An alternative to the makefile for updating the auto-generated Jupyter notebooks. Requires package ``ray`` to be installed. Notebooks are executed in parallel.

`pytojnb <pytojnb>`_
   Low-level python to Jupyter notebook conversion script. Used by both the makefile and `makejnb.py <makejnb.py>`_.

`scriptcheck <scriptcheck>`_
   Run all example scripts with a reduced number of iterations as a rapid check that they are functioning correctly.
