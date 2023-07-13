Usage Examples
==============


Organized by Application
------------------------


Computed Tomography
^^^^^^^^^^^^^^^^^^^

   - ct_abel_tv_admm.py
   - ct_abel_tv_admm_tune.py
   - ct_astra_noreg_pcg.py
   - ct_astra_3d_tv_admm.py
   - ct_astra_tv_admm.py
   - ct_astra_weighted_tv_admm.py
   - ct_svmbir_tv_multi.py
   - ct_svmbir_ppp_bm3d_admm_cg.py
   - ct_svmbir_ppp_bm3d_admm_prox.py
   - ct_fan_svmbir_ppp_bm3d_admm_prox.py
   - ct_astra_modl_train_foam2.py
   - ct_astra_odp_train_foam2.py
   - ct_astra_unet_train_foam2.py


Deconvolution
^^^^^^^^^^^^^

   - deconv_circ_tv_admm.py
   - deconv_tv_admm.py
   - deconv_tv_padmm.py
   - deconv_tv_admm_tune.py
   - deconv_microscopy_tv_admm.py
   - deconv_microscopy_allchn_tv_admm.py
   - deconv_ppp_bm3d_admm.py
   - deconv_ppp_bm3d_pgm.py
   - deconv_ppp_dncnn_admm.py
   - deconv_ppp_dncnn_padmm.py
   - deconv_ppp_bm4d_admm.py
   - deconv_modl_train_foam1.py
   - deconv_odp_train_foam1.py


Sparse Coding
^^^^^^^^^^^^^

   - sparsecode_admm.py
   - sparsecode_conv_admm.py
   - sparsecode_conv_md_admm.py
   - sparsecode_pgm.py
   - sparsecode_poisson_pgm.py


Miscellaneous
^^^^^^^^^^^^^

   - demosaic_ppp_bm3d_admm.py
   - superres_ppp_dncnn_admm.py
   - denoise_l1tv_admm.py
   - denoise_tv_admm.py
   - denoise_tv_pgm.py
   - denoise_tv_multi.py
   - denoise_cplx_tv_nlpadmm.py
   - denoise_cplx_tv_pdhg.py
   - denoise_dncnn_universal.py
   - diffusercam_tv_admm.py
   - video_rpca_admm.py
   - ct_astra_datagen_foam2.py
   - deconv_datagen_bsds.py
   - deconv_datagen_foam1.py
   - denoise_datagen_bsds.py


Organized by Regularization
---------------------------

Plug and Play Priors
^^^^^^^^^^^^^^^^^^^^

   - ct_svmbir_ppp_bm3d_admm_cg.py
   - ct_svmbir_ppp_bm3d_admm_prox.py
   - ct_fan_svmbir_ppp_bm3d_admm_prox.py
   - deconv_ppp_bm3d_admm.py
   - deconv_ppp_bm3d_pgm.py
   - deconv_ppp_dncnn_admm.py
   - deconv_ppp_dncnn_padmm.py
   - deconv_ppp_bm4d_admm.py
   - demosaic_ppp_bm3d_admm.py
   - superres_ppp_dncnn_admm.py


Total Variation
^^^^^^^^^^^^^^^

   - ct_abel_tv_admm.py
   - ct_abel_tv_admm_tune.py
   - ct_astra_tv_admm.py
   - ct_astra_3d_tv_admm.py
   - ct_astra_weighted_tv_admm.py
   - ct_svmbir_tv_multi.py
   - deconv_circ_tv_admm.py
   - deconv_tv_admm.py
   - deconv_tv_admm_tune.py
   - deconv_tv_padmm.py
   - deconv_microscopy_tv_admm.py
   - deconv_microscopy_allchn_tv_admm.py
   - denoise_l1tv_admm.py
   - denoise_tv_admm.py
   - denoise_tv_pgm.py
   - denoise_tv_multi.py
   - denoise_cplx_tv_nlpadmm.py
   - denoise_cplx_tv_pdhg.py
   - diffusercam_tv_admm.py



Sparsity
^^^^^^^^

   - diffusercam_tv_admm.py
   - sparsecode_admm.py
   - sparsecode_conv_admm.py
   - sparsecode_conv_md_admm.py
   - sparsecode_pgm.py
   - sparsecode_poisson_pgm.py
   - video_rpca_admm.py


Machine Learning
^^^^^^^^^^^^^^^^

   - ct_astra_datagen_foam2.py
   - ct_astra_modl_train_foam2.py
   - ct_astra_odp_train_foam2.py
   - ct_astra_unet_train_foam2.py
   - deconv_datagen_bsds.py
   - deconv_datagen_foam1.py
   - deconv_modl_train_foam1.py
   - deconv_odp_train_foam1.py
   - denoise_datagen_bsds.py
   - denoise_dncnn_train_bsds.py
   - denoise_dncnn_universal.py


Organized by Optimization Algorithm
-----------------------------------

ADMM
^^^^

   - ct_abel_tv_admm.py
   - ct_abel_tv_admm_tune.py
   - ct_astra_tv_admm.py
   - ct_astra_3d_tv_admm.py
   - ct_astra_weighted_tv_admm.py
   - ct_svmbir_tv_multi.py
   - ct_svmbir_ppp_bm3d_admm_cg.py
   - ct_svmbir_ppp_bm3d_admm_prox.py
   - ct_fan_svmbir_ppp_bm3d_admm_prox.py
   - deconv_circ_tv_admm.py
   - deconv_tv_admm.py
   - deconv_tv_admm_tune.py
   - deconv_microscopy_tv_admm.py
   - deconv_microscopy_allchn_tv_admm.py
   - deconv_ppp_bm3d_admm.py
   - deconv_ppp_dncnn_admm.py
   - deconv_ppp_bm4d_admm.py
   - diffusercam_tv_admm.py
   - sparsecode_admm.py
   - sparsecode_conv_admm.py
   - sparsecode_conv_md_admm.py
   - demosaic_ppp_bm3d_admm.py
   - superres_ppp_dncnn_admm.py
   - denoise_l1tv_admm.py
   - denoise_tv_admm.py
   - denoise_tv_multi.py
   - video_rpca_admm.py


Linearized ADMM
^^^^^^^^^^^^^^^

    - ct_svmbir_tv_multi.py
    - denoise_tv_multi.py


Proximal ADMM
^^^^^^^^^^^^^

    - deconv_tv_padmm.py
    - denoise_tv_multi.py
    - denoise_cplx_tv_nlpadmm.py
    - deconv_ppp_dncnn_padmm.py


Non-linear Proximal ADMM
^^^^^^^^^^^^^^^^^^^^^^^^

    - denoise_cplx_tv_nlpadmm.py


PDHG
^^^^

    - ct_svmbir_tv_multi.py
    - denoise_tv_multi.py
    - denoise_cplx_tv_pdhg.py


PGM
^^^

   - deconv_ppp_bm3d_pgm.py
   - sparsecode_pgm.py
   - sparsecode_poisson_pgm.py
   - denoise_tv_pgm.py


PCG
^^^

   - ct_astra_noreg_pcg.py
