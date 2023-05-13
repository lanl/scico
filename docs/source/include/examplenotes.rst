.. _example_depend:

Example Dependencies
--------------------

Some examples use additional dependencies, which are listed in `examples_requirements.txt <https://github.com/lanl/scico/blob/main/examples/examples_requirements.txt>`_.
The additional requirements should be installed via pip, with the exception of ``astra-toolbox``,
which should be installed via conda:

::

   conda install -c astra-toolbox astra-toolbox
   pip install -r examples/examples_requirements.txt  # Installs other example requirements

The dependencies can also be installed individually as required.

Note that ``astra-toolbox`` should be installed on a host with one or more CUDA GPUs to ensure
that the version with GPU support is installed.


Run Time
--------

Most of these examples have been constructed with sufficiently small test problems to
allow them to run to completion within 5 minutes or less on a reasonable workstation.
Note, however, that it was not feasible to construct meaningful examples of the training
of some of the deep learning algorithms that complete within a relatively short time;
the examples "CT Training and Reconstructions with MoDL" and "CT Training and
Reconstructions with ODP" in particular are much slower, and can require multiple hours
to run on a workstation with multiple GPUs.

|
