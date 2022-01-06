.. _example_notebooks:

Usage Examples
==============

.. toctree::
   :maxdepth: 1

.. _example_dependencies:

Example Dependencies
------------------------

Some examples may use additional dependecies. These dependencies are listed in `examples_requirements.txt <https://github.com/lanl/scico/blob/main/examples/examples_requirements.txt>`_.
Pip should be used to install these extra requirements except astra which should be installed via conda:

   ::

      conda install -c astra-toolbox astra-toolbox
      pip install -r examples/examples_requirements.txt # Installs other example requirements

The dependencies can also be installed one by one as required.

Organized by Application
------------------------

.. toctree::
   :maxdepth: 1


Computed Tomography
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_astra_pcg
   examples/ct_astra_tv_admm
   examples/ct_astra_weighted_tv_admm
   examples/ct_svmbir_ppp_bm3d_admm_cg
   examples/ct_svmbir_ppp_bm3d_admm_prox
   examples/ct_svmbir_tv_multi


Deconvolution
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/deconv_circ_tv_admm
   examples/deconv_microscopy_tv_admm
   examples/deconv_microscopy_allchn_tv_admm
   examples/deconv_ppp_bm3d_admm
   examples/deconv_ppp_bm3d_pgm
   examples/deconv_ppp_dncnn_admm
   examples/deconv_tv_admm


Sparse Coding
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/sparsecode_admm
   examples/sparsecode_pgm
   examples/sparsecode_poisson_blkarr_pgm
   examples/sparsecode_poisson_pgm


Miscellaneous
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/demosaic_ppp_bm3d_admm
   examples/denoise_tv_iso_admm
   examples/denoise_tv_iso_pgm
   examples/denoise_tv_iso_multi



Organized by Regularization
---------------------------

.. toctree::
   :maxdepth: 1

Plug and Play Priors
^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_svmbir_ppp_bm3d_admm_cg
   examples/ct_svmbir_ppp_bm3d_admm_prox
   examples/deconv_ppp_bm3d_admm
   examples/deconv_ppp_bm3d_pgm
   examples/deconv_ppp_dncnn_admm
   examples/demosaic_ppp_bm3d_admm


Total Variation
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_astra_tv_admm
   examples/ct_astra_weighted_tv_admm
   examples/ct_svmbir_tv_multi
   examples/deconv_circ_tv_admm
   examples/deconv_microscopy_tv_admm
   examples/deconv_microscopy_allchn_tv_admm
   examples/deconv_tv_admm
   examples/denoise_tv_iso_admm
   examples/denoise_tv_iso_pgm
   examples/denoise_tv_iso_multi


Sparsity
^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/sparsecode_admm
   examples/sparsecode_pgm
   examples/sparsecode_poisson_blkarr_pgm
   examples/sparsecode_poisson_pgm



Organized by Optimization Algorithm
-----------------------------------

.. toctree::
   :maxdepth: 1

ADMM
^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_astra_tv_admm
   examples/ct_astra_weighted_tv_admm
   examples/ct_svmbir_ppp_bm3d_admm_cg
   examples/ct_svmbir_ppp_bm3d_admm_prox
   examples/ct_svmbir_tv_multi
   examples/deconv_circ_tv_admm
   examples/deconv_microscopy_tv_admm
   examples/deconv_microscopy_allchn_tv_admm
   examples/deconv_ppp_bm3d_admm
   examples/deconv_ppp_dncnn_admm
   examples/deconv_tv_admm
   examples/demosaic_ppp_bm3d_admm
   examples/denoise_tv_iso_admm
   examples/denoise_tv_iso_multi
   examples/sparsecode_admm


Linearized ADMM
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_svmbir_tv_multi
   examples/denoise_tv_iso_multi


PDHG
^^^^

.. toctree::
   :maxdepth: 1

   examples/ct_svmbir_tv_multi
   examples/denoise_tv_iso_multi


PGM
^^^

.. toctree::
   :maxdepth: 1

   examples/deconv_ppp_bm3d_pgm
   examples/denoise_tv_iso_pgm
   examples/sparsecode_pgm
   examples/sparsecode_poisson_blkarr_pgm
   examples/sparsecode_poisson_pgm


PCG
^^^

.. toctree::
   :maxdepth: 1

   examples/ct_astra_pcg
