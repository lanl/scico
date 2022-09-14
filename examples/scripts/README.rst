Usage Examples
==============


Organized by Application
------------------------


Computed Tomography
^^^^^^^^^^^^^^^^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      TV-Regularized Abel Inversion
   `ct_astra_noreg_pcg.py <ct_astra_noreg_pcg.py>`_
      CT Reconstruction with CG and PCG
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      TV-Regularized Low-Dose CT Reconstruction
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      TV-Regularized CT Reconstruction (Multiple Algorithms)
   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with CG Subproblem Solver)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with Fast SVMBIR Prox)
   `ct_fan_svmbir_ppp_bm3d_admm_prox.py <ct_fan_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) Fan-Beam CT Reconstruction


Deconvolution
^^^^^^^^^^^^^

   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Circulant Blur Image Deconvolution with TV Regularization
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution with TV Regularization
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Deconvolution (ADMM Solver)
   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution
   `deconv_ppp_bm4d_admm.py <deconv_ppp_bm4d_admm.py>`_
      PPP (with BM4D) Volume Deconvolution


Sparse Coding
^^^^^^^^^^^^^

   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)


Miscellaneous
^^^^^^^^^^^^^

   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Demosaicing
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Superresolution
   `denoise_l1tv_admm.py <denoise_l1tv_admm.py>`_
      ℓ1 Total Variation Denoising
   `denoise_tv_admm.py <denoise_tv_admm.py>`_
      Total Variation Denoising (ADMM)
   `denoise_tv_pgm.py <denoise_tv_pgm.py>`_
      Total Variation Denoising with Constraint (APGM)
   `denoise_tv_multi.py <denoise_tv_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
      Complex Total Variation Denoising
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA



Organized by Regularization
---------------------------

Plug and Play Priors
^^^^^^^^^^^^^^^^^^^^

   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with CG Subproblem Solver)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with Fast SVMBIR Prox)
   `ct_fan_svmbir_ppp_bm3d_admm_prox.py <ct_fan_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) Fan-Beam CT Reconstruction
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Deconvolution (ADMM Solver)
   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution
   `deconv_ppp_bm4d_admm.py <deconv_ppp_bm4d_admm.py>`_
      PPP (with BM4D) Volume Deconvolution
   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Demosaicing
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Superresolution


Total Variation
^^^^^^^^^^^^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      TV-Regularized Abel Inversion
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      TV-Regularized Low-Dose CT Reconstruction
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      TV-Regularized CT Reconstruction (Multiple Algorithms)
   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Circulant Blur Image Deconvolution with TV Regularization
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution with TV Regularization
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `denoise_l1tv_admm.py <denoise_l1tv_admm.py>`_
      ℓ1 Total Variation Denoising
   `denoise_tv_admm.py <denoise_tv_admm.py>`_
      Total Variation Denoising (ADMM)
   `denoise_tv_pgm.py <denoise_tv_pgm.py>`_
      Total Variation Denoising with Constraint (APGM)
   `denoise_tv_multi.py <denoise_tv_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
      Complex Total Variation Denoising


Sparsity
^^^^^^^^

   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA



Organized by Optimization Algorithm
-----------------------------------

ADMM
^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      TV-Regularized Abel Inversion
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      TV-Regularized Low-Dose CT Reconstruction
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      TV-Regularized CT Reconstruction (Multiple Algorithms)
   `ct_svmbir_ppp_bm3d_admm_cg.py <ct_svmbir_ppp_bm3d_admm_cg.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with CG Subproblem Solver)
   `ct_svmbir_ppp_bm3d_admm_prox.py <ct_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) CT Reconstruction (ADMM with Fast SVMBIR Prox)
   `ct_fan_svmbir_ppp_bm3d_admm_prox.py <ct_fan_svmbir_ppp_bm3d_admm_prox.py>`_
      PPP (with BM3D) Fan-Beam CT Reconstruction
   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Circulant Blur Image Deconvolution with TV Regularization
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution with TV Regularization
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Image Deconvolution Parameter Tuning
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Deconvolution (ADMM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution
   `deconv_ppp_bm4d_admm.py <deconv_ppp_bm4d_admm.py>`_
      PPP (with BM4D) Volume Deconvolution
   `sparsecode_admm.py <sparsecode_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Demosaicing
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Superresolution
   `denoise_l1tv_admm.py <denoise_l1tv_admm.py>`_
      ℓ1 Total Variation Denoising
   `denoise_tv_admm.py <denoise_tv_admm.py>`_
      Total Variation Denoising (ADMM)
   `denoise_tv_multi.py <denoise_tv_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA


Linearized ADMM
^^^^^^^^^^^^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       TV-Regularized CT Reconstruction (Multiple Algorithms)
    `denoise_tv_multi.py <denoise_tv_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising


PDHG
^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       TV-Regularized CT Reconstruction (Multiple Algorithms)
    `denoise_tv_multi.py <denoise_tv_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising
    `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
       Complex Total Variation Denoising


PGM
^^^

   `deconv_ppp_bm3d_pgm.py <deconv_ppp_bm3d_pgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `sparsecode_pgm.py <sparsecode_pgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_pgm.py <sparsecode_poisson_pgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)
   `denoise_tv_pgm.py <denoise_tv_pgm.py>`_
      Total Variation Denoising with Constraint (APGM)


PCG
^^^

   `ct_astra_noreg_pcg.py <ct_astra_noreg_pcg.py>`_
      CT Reconstruction with CG and PCG
