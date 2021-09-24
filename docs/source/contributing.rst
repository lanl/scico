.. _scico_dev_contributing:

Contributing
============

.. raw:: html

    <style type='text/css'>
    div.document ul blockquote {
       margin-bottom: 8px !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    div.document li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    </style>


Contributions to SCICO are welcome. Before starting work, please contact the maintainers, either via email or the GitHub issue system, to discuss the relevance of your contribution and the most appropriate location within the existing package structure.



.. _installing_dev:

Installing a Development Version
--------------------------------


.. todo::

   At time of public release, this should be updated to a forking tutorial (see
   `the jax example <https://jax.readthedocs.io/en/latest/contributing.html#contributing-code-using-pull-requests>`_)


1. Create a conda environment using Python >= 3.8.

::

   conda create -n scico python=3.8


2. Activate the conda virtual environment:

::

   conda activate scico

3. Clone the SCICO repository:

::

   git clone https://github.com/lanl/scico.git --recurse-submodules


4. Navigate to the cloned repository:

::

    cd scico

5. Install dependencies:

::

  pip install -r requirements.txt  # Installs basic requirements
  pip install -r docs/docs_requirements.txt # Installs documentation requirements
  pip install -r examples/examples_requirements.txt # Installs example requirements
  pip install -e .  # Installs SCICO from the current directory in editable mode.

6. Set up ``black`` and ``isort`` pre-commit hooks

::

  pre-commit install  # Sets up git pre-commit hooks

7. If desired, tests can be run on the installed version:

::

   pytest --pyargs scico


Contributing Code
-----------------

- New features / bugs / documentation are *always* developed in separate branches.
- Branches should be named in the form `<username>/<brief-description>`,
  where `<brief-description>` provides a highly condensed description of the purpose of the branch (e.g. `address_todo`), and may include an issue number if appropriate (e.g. `fix_223`).

|

A feature development workflow might look like this:

1. Follow the instructions in `Installing a Development Version`_.

2. Sync with the upstream repository:

::

   git pull --rebase origin main --recurse-submodules

3. Create a branch to develop from:

::

   git checkout -b name-of-change

4. Make your desired changes.

5. Run the test suite:

::

   pytest

You can limit the test suite to a specific file:

::

   pytest scico/test/test_blockarray.py

6. When you are finished making changes, create a new commit:

::

   git add file1.py git add file2.py
   git commit -m "A good commit message"


NOTE:  If you have added or modified an example script, see `Adding Usage Examples`_

7. Sync with the upstream repository:

::

   git pull --rebase origin main --recurse-submodules


8. Push your development upstream:

::

   git push --set-upstream origin name-of-change

9.  Create a new pull request to the ``main`` branch; see `the GitHub instructions <https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_

10. Delete the branch after it has been merged.


Adding Usage Examples
---------------------

New usage examples should adhere to the same general structure as the existing examples to ensure that the mechanism for automatically generating corresponding Jupyter notebooks functions correctly. In particular:

1. The initial lines of the script should consist of a comment block, followed by a blank line, followed by a multiline string with an RST heading on the first line, e.g.

::

  #!/usr/bin/env python
  # -*- coding: utf-8 -*-
  # This file is part of the SCICO package. Details of the copyright
  # and user license can be found in the 'LICENSE.txt' file distributed
  # with the package.

  """
  Script Title
  ============

  Script description.
  """

2. The final line of the script is an ``input`` statement intended to avoid the script terminating immediately, thereby closing all figures:

::

  input("\nWaiting for input to close figures and exit")

3. Citations are included using the standard `Sphinx <https://www.sphinx-doc.org/en/master/>`__ ``:cite:`cite-key``` syntax, where ``cite-key`` is the key of an entry in ``docs/source/references.bib``.

4. Cross-references to other components of the documentation are included using the syntax described in the `nbsphinx documentation <https://nbsphinx.readthedocs.io/en/0.3.5/markdown-cells.html#Links-to-*.rst-Files-(and-Other-Sphinx-Source-Files)>`__.

5. External links are included using Markdown syntax ``[link text](url)``.


Adding new examples
^^^^^^^^^^^^^^^^^^^

The following steps show how to add a new example, ``new_example.py``, to the packaged usage
examples. We assume the SCICO repository has been cloned to ``scico/``.

Note that the ``.py`` scripts are included in ``scico/examples/scripts``, while the compiled
Jupyter Notebooks are located in the scico-data submodule, which is symlinked to ``scico/data``.
When adding a new usage example, both the scico and scico-data repositories must be updated and
kept in sync.

.. warning::
   Ensure that all binary data (including raw data, images, ``.ipynb`` files) are added to scico-data, not the base ``scico`` repo.



1. Add the ``new_example.py`` script to the ``scico/examples/scripts`` directory.

2. Add the basename of the script (i.e., without the pathname or ``.py`` extension; in this case,
   ``new_example``) to ``examples/notebooks/examples.rst``.

3. Convert your new example to a Jupyter notebook by navigating the ``scico/examples`` directory and performing

::

   make notebooks/new_example.ipynb

Alternatively, all examples can be run by calling

::

   make

from ``scico/examples``.

4.  Navigate to the ``data`` directory and add/commit the new Jupyter Notebook

::

   cd scico/data
   git add notebooks/new_example.ipynb
   git commit -m "Add new usage example"

5.  Return to the base SCICO repository, ensure the ``main`` branch is checked out, add/commit the new script and updated submodule:

::

   cd ..  # pwd now `scico` repo root
   git add data
   git add examples/scripts/new_filename.py
   git commit -m "Add usage example and update data module"

6.  Push both repositories:

::

  git submodule foreach --recursive 'git push' && git push


Adding New Data
---------------

The following steps show how to add new data, ``new_data.npz``, to the packaged data. We assume the SCICO repository has been cloned to ``scico/``.

Note that the data is located in the scico-data submodule, which is symlinked to ``scico/data``.
When adding new data, both the scico and scico-data repositories must be updated and
kept in sync.


1. Add the ``new_data.npz`` file to the ``scico/data`` directory.

2.  Navigate to the ``data`` directory and add/commit the new data file

::

   cd scico/data
   git add new_data.npz
   git commit -m "Add new data file"

3.  Return to the base SCICO repository, ensure the ``main`` branch is checked out, add/commit the new data and update submodule:

::

   cd ..  # pwd now `scico` repo root
   git checkout main
   git add data
   git commit -m "Add data and update data module"

4.  Push both repositories:

::

  git submodule foreach --recursive 'git push' && git push


Tests
=====

All functions and classes should have corresponding `pytest` unit tests.


Running Tests
-------------


To be able to run the tests, install `pytest` and, optionally, `pytest-runner`

::

    conda install pytest pytest-runner

The tests can be run by

::

    pytest

or

::

    python setup.py test


Type Checking
-------------

In the future, we will require all code to pass `mypy` type checking.  This is not currently enforced.

Install ``mypy``:

::

   conda install mypy

To run the type checker on the ``scico`` module:

::

   mypy -p scico




Building Documentation
======================

To build a local copy of the docs, from the repo root directory, do

::

  python setup.py build_sphinx




Test Coverage
-------------

Test coverage is a measure of the fraction of the package code that is exercised by the tests. While this should not be the primary criterion in designing tests, it is a useful tool for finding obvious areas of omission.

To be able to check test coverage, install `coverage`

::

    conda install coverage

A coverage report can be obtained by

::

    coverage run
    coverage report
