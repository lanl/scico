Usage Examples
==============


Organized by Application
------------------------


Computed Tomography
^^^^^^^^^^^^^^^^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      Regularized Abel Inversion
   `ct_astra_pcg.py <ct_astra_pcg.py>`_
      CT with Preconditioned Conjugate Gradient
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      Few-View CT (ADMM w/ Total Variation)
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      Low-Dose CT (ADMM w/ Total Variation)
   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+CG)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+Prox)
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      CT Reconstruction with TV Regularization


Deconvolution
^^^^^^^^^^^^^

   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation and Circulant Blur)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ BM3D)
   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      Image Deconvolution (PGM Plug-and-Play Priors w/ BM3D)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ DnCNN)
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning


Sparse Coding
^^^^^^^^^^^^^

   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (Accelerated PGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM w/ adaptive PGMStepSize)


Miscellaneous
^^^^^^^^^^^^^

   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      Image Demosaicing (ADMM Plug-and-Play Priors w/ BM3D)
   `denoise_tv_iso_admm.py <denoise_tv_iso_admm.py>`_
      Isotropic Total Variation (ADMM)
   `denoise_tv_iso_pgm.py <denoise_tv_iso_pgm.py>`_
      Isotropic Total Variation (Accelerated PGM)
   `denoise_tv_iso_multi.py <denoise_tv_iso_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      Image Superresolution (ADMM Plug-and-Play Priors w/ DnCNN)
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA



Organized by Regularization
---------------------------

Plug and Play Priors
^^^^^^^^^^^^^^^^^^^^

   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+CG)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+Prox)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ BM3D)
   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      Image Deconvolution (PGM Plug-and-Play Priors w/ BM3D)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ DnCNN)
   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      Image Demosaicing (ADMM Plug-and-Play Priors w/ BM3D)
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      Image Superresolution (ADMM Plug-and-Play Priors w/ DnCNN)


Total Variation
^^^^^^^^^^^^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      Regularized Abel Inversion
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      Few-View CT (ADMM w/ Total Variation)
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      Low-Dose CT (ADMM w/ Total Variation)
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      CT Reconstruction with TV Regularization
   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation and Circulant Blur)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning
   `denoise_tv_iso_admm.py <denoise_tv_iso_admm.py>`_
      Isotropic Total Variation (ADMM)
   `denoise_tv_iso_pgm.py <denoise_tv_iso_pgm.py>`_
      Isotropic Total Variation (Accelerated PGM)
   `denoise_tv_iso_multi.py <denoise_tv_iso_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising


Sparsity
^^^^^^^^

   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (Accelerated PGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM w/ adaptive PGMStepSize)
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA



Organized by Optimization Algorithm
-----------------------------------

ADMM
^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      Regularized Abel Inversion
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      Few-View CT (ADMM w/ Total Variation)
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      Low-Dose CT (ADMM w/ Total Variation)
   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+CG)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      CT Reconstruction (ADMM Plug-and-Play Priors w/ BM3D, SVMBIR+Prox)
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      CT Reconstruction with TV Regularization
   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation and Circulant Blur)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ BM3D)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      Image Deconvolution (ADMM Plug-and-Play Priors w/ DnCNN)
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution (ADMM w/ Total Variation)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning
   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      Image Demosaicing (ADMM Plug-and-Play Priors w/ BM3D)
   `denoise_tv_iso_admm.py <denoise_tv_iso_admm.py>`_
      Isotropic Total Variation (ADMM)
   `denoise_tv_iso_multi.py <denoise_tv_iso_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-negative Basis Pursuit DeNoising (ADMM)
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      Image Superresolution (ADMM Plug-and-Play Priors w/ DnCNN)
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA


Linearized ADMM
^^^^^^^^^^^^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       CT Reconstruction with TV Regularization
    `denoise_tv_iso_multi.py <denoise_tv_iso_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising


PDHG
^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       CT Reconstruction with TV Regularization
    `denoise_tv_iso_multi.py <denoise_tv_iso_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising


PGM
^^^

   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      Image Deconvolution (PGM Plug-and-Play Priors w/ BM3D)
   `denoise_tv_iso_pgm.py <denoise_tv_iso_pgm.py>`_
      Isotropic Total Variation (Accelerated PGM)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (Accelerated PGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM w/ adaptive PGMStepSize)


PCG
^^^

   `ct_astra_pcg.py <ct_astra_pcg.py>`_
      CT with Preconditioned Conjugate Gradient
