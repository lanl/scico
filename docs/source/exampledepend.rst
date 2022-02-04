.. _example_dependencies:

Example Dependencies
--------------------

Some examples use additional dependencies, which are listed in `examples_requirements.txt <https://github.com/lanl/scico/blob/main/examples/examples_requirements.txt>`_.
The additional requirements should be installed via pip, with the exception of ``astra-toolbox``,
which should be installed via conda:

   ::

      conda install -c astra-toolbox astra-toolbox
      pip install -r examples/examples_requirements.txt # Installs other example requirements

The dependencies can also be installed individually as required.
