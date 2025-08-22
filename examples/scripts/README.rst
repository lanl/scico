Usage Examples
==============


Organized by Application
------------------------


Computed Tomography
^^^^^^^^^^^^^^^^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      TV-Regularized Abel Inversion
   `ct_abel_tv_admm_tune.py <ct_abel_tv_admm_tune.py>`_
      Parameter Tuning for TV-Regularized Abel Inversion
   `ct_symcone_tv_padmm.py <ct_symcone_tv_padmm.py>`_
      TV-Regularized Cone Beam CT for Symmetric Objects
   `ct_astra_noreg_pcg.py <ct_astra_noreg_pcg.py>`_
      CT Reconstruction with CG and PCG
   `ct_astra_3d_tv_admm.py <ct_astra_3d_tv_admm.py>`_
      3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver)
   `ct_astra_3d_tv_padmm.py <ct_astra_3d_tv_padmm.py>`_
      3D TV-Regularized Sparse-View CT Reconstruction (Proximal ADMM Solver)
   `ct_tv_admm.py <ct_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Integrated Projector)
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (ASTRA Projector)
   `ct_multi_tv_admm.py <ct_multi_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Multiple Projectors)
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
   `ct_modl_train_foam2.py <ct_modl_train_foam2.py>`_
      CT Training and Reconstruction with MoDL
   `ct_odp_train_foam2.py <ct_odp_train_foam2.py>`_
      CT Training and Reconstruction with ODP
   `ct_unet_train_foam2.py <ct_unet_train_foam2.py>`_
      CT Training and Reconstructions with UNet
   `ct_projector_comparison_2d.py <ct_projector_comparison_2d.py>`_
      2D X-ray Transform Comparison
   `ct_projector_comparison_3d.py <ct_projector_comparison_3d.py>`_
      3D X-ray Transform Comparison

Deconvolution
^^^^^^^^^^^^^

   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Circulant Blur Image Deconvolution with TV Regularization
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_tv_padmm.py <deconv_tv_padmm.py>`_
      Image Deconvolution with TV Regularization (Proximal ADMM Solver)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Parameter Tuning for Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Deconvolution (ADMM Solver)
   `deconv_ppp_bm3d_apgm.py <deconv_ppp_bm3d_apgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution (ADMM Solver)
   `deconv_ppp_dncnn_padmm.py <deconv_ppp_dncnn_padmm.py>`_
      PPP (with DnCNN) Image Deconvolution (Proximal ADMM Solver)
   `deconv_ppp_bm4d_admm.py <deconv_ppp_bm4d_admm.py>`_
      PPP (with BM4D) Volume Deconvolution
   `deconv_modl_train_foam1.py <deconv_modl_train_foam1.py>`_
      Deconvolution Training and Reconstructions with MoDL
   `deconv_odp_train_foam1.py <deconv_odp_train_foam1.py>`_
      Deconvolution Training and Reconstructions with ODP


Sparse Coding
^^^^^^^^^^^^^

   `sparsecode_nn_admm.py <sparsecode_nn_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_nn_apgm.py <sparsecode_nn_apgm.py>`_
      Non-Negative Basis Pursuit DeNoising (APGM)
   `sparsecode_conv_admm.py <sparsecode_conv_admm.py>`_
      Convolutional Sparse Coding (ADMM)
   `sparsecode_conv_md_admm.py <sparsecode_conv_md_admm.py>`_
      Convolutional Sparse Coding with Mask Decoupling (ADMM)
   `sparsecode_apgm.py <sparsecode_apgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_apgm.py <sparsecode_poisson_apgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)


Miscellaneous
^^^^^^^^^^^^^

   `demosaic_ppp_bm3d_admm.py <demosaic_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Demosaicing
   `superres_ppp_dncnn_admm.py <superres_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Superresolution
   `denoise_l1tv_admm.py <denoise_l1tv_admm.py>`_
      ℓ1 Total Variation Denoising
   `denoise_ptv_pdhg.py <denoise_ptv_pdhg.py>`_
      Polar Total Variation Denoising (PDHG)
   `denoise_tv_admm.py <denoise_tv_admm.py>`_
      Total Variation Denoising (ADMM)
   `denoise_tv_apgm.py <denoise_tv_apgm.py>`_
      Total Variation Denoising with Constraint (APGM)
   `denoise_tv_multi.py <denoise_tv_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `denoise_approx_tv_multi.py <denoise_approx_tv_multi.py>`_
      Denoising with Approximate Total Variation Proximal Operator
   `denoise_cplx_tv_nlpadmm.py <denoise_cplx_tv_nlpadmm.py>`_
      Complex Total Variation Denoising with NLPADMM Solver
   `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
      Complex Total Variation Denoising with PDHG Solver
   `denoise_dncnn_universal.py <denoise_dncnn_universal.py>`_
      Comparison of DnCNN Variants for Image Denoising
   `diffusercam_tv_admm.py <diffusercam_tv_admm.py>`_
      TV-Regularized 3D DiffuserCam Reconstruction
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA
   `ct_datagen_foam2.py <ct_datagen_foam2.py>`_
      CT Data Generation for NN Training
   `deconv_datagen_bsds.py <deconv_datagen_bsds.py>`_
      Blurred Data Generation (Natural Images) for NN Training
   `deconv_datagen_foam1.py <deconv_datagen_foam1.py>`_
      Blurred Data Generation (Foams) for NN Training
   `denoise_datagen_bsds.py <denoise_datagen_bsds.py>`_
      Noisy Data Generation for NN Training


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
   `deconv_ppp_bm3d_apgm.py <deconv_ppp_bm3d_apgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution (ADMM Solver)
   `deconv_ppp_dncnn_padmm.py <deconv_ppp_dncnn_padmm.py>`_
      PPP (with DnCNN) Image Deconvolution (Proximal ADMM Solver)
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
   `ct_abel_tv_admm_tune.py <ct_abel_tv_admm_tune.py>`_
      Parameter Tuning for TV-Regularized Abel Inversion
   `ct_symcone_tv_padmm.py <ct_symcone_tv_padmm.py>`_
      TV-Regularized Cone Beam CT for Symmetric Objects
   `ct_tv_admm.py <ct_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Integrated Projector)
   `ct_multi_tv_admm.py <ct_multi_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Multiple Projectors)
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (ASTRA Projector)
   `ct_astra_3d_tv_admm.py <ct_astra_3d_tv_admm.py>`_
      3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver)
   `ct_astra_3d_tv_padmm.py <ct_astra_3d_tv_padmm.py>`_
      3D TV-Regularized Sparse-View CT Reconstruction (Proximal ADMM Solver)
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      TV-Regularized Low-Dose CT Reconstruction
   `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
      TV-Regularized CT Reconstruction (Multiple Algorithms)
   `deconv_circ_tv_admm.py <deconv_circ_tv_admm.py>`_
      Circulant Blur Image Deconvolution with TV Regularization
   `deconv_tv_admm.py <deconv_tv_admm.py>`_
      Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Parameter Tuning for Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_tv_padmm.py <deconv_tv_padmm.py>`_
      Image Deconvolution with TV Regularization (Proximal ADMM Solver)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `denoise_l1tv_admm.py <denoise_l1tv_admm.py>`_
      ℓ1 Total Variation Denoising
   `denoise_ptv_pdhg.py <denoise_ptv_pdhg.py>`_
      Polar Total Variation Denoising (PDHG)
   `denoise_tv_admm.py <denoise_tv_admm.py>`_
      Total Variation Denoising (ADMM)
   `denoise_tv_apgm.py <denoise_tv_apgm.py>`_
      Total Variation Denoising with Constraint (APGM)
   `denoise_tv_multi.py <denoise_tv_multi.py>`_
      Comparison of Optimization Algorithms for Total Variation Denoising
   `denoise_approx_tv_multi.py <denoise_approx_tv_multi.py>`_
      Denoising with Approximate Total Variation Proximal Operator
   `denoise_cplx_tv_nlpadmm.py <denoise_cplx_tv_nlpadmm.py>`_
      Complex Total Variation Denoising with NLPADMM Solver
   `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
      Complex Total Variation Denoising with PDHG Solver
   `diffusercam_tv_admm.py <diffusercam_tv_admm.py>`_
      TV-Regularized 3D DiffuserCam Reconstruction



Sparsity
^^^^^^^^

   `diffusercam_tv_admm.py <diffusercam_tv_admm.py>`_
      TV-Regularized 3D DiffuserCam Reconstruction
   `sparsecode_nn_admm.py <sparsecode_nn_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_nn_apgm.py <sparsecode_nn_apgm.py>`_
      Non-Negative Basis Pursuit DeNoising (APGM)
   `sparsecode_conv_admm.py <sparsecode_conv_admm.py>`_
      Convolutional Sparse Coding (ADMM)
   `sparsecode_conv_md_admm.py <sparsecode_conv_md_admm.py>`_
      Convolutional Sparse Coding with Mask Decoupling (ADMM)
   `sparsecode_apgm.py <sparsecode_apgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_apgm.py <sparsecode_poisson_apgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA


Machine Learning
^^^^^^^^^^^^^^^^

   `ct_datagen_foam2.py <ct_datagen_foam2.py>`_
      CT Data Generation for NN Training
   `ct_modl_train_foam2.py <ct_modl_train_foam2.py>`_
      CT Training and Reconstruction with MoDL
   `ct_odp_train_foam2.py <ct_odp_train_foam2.py>`_
      CT Training and Reconstruction with ODP
   `ct_unet_train_foam2.py <ct_unet_train_foam2.py>`_
      CT Training and Reconstructions with UNet
   `deconv_datagen_bsds.py <deconv_datagen_bsds.py>`_
      Blurred Data Generation (Natural Images) for NN Training
   `deconv_datagen_foam1.py <deconv_datagen_foam1.py>`_
      Blurred Data Generation (Foams) for NN Training
   `deconv_modl_train_foam1.py <deconv_modl_train_foam1.py>`_
      Deconvolution Training and Reconstructions with MoDL
   `deconv_odp_train_foam1.py <deconv_odp_train_foam1.py>`_
      Deconvolution Training and Reconstructions with ODP
   `denoise_datagen_bsds.py <denoise_datagen_bsds.py>`_
      Noisy Data Generation for NN Training
   `denoise_dncnn_train_bsds.py <denoise_dncnn_train_bsds.py>`_
      Training of DnCNN for Denoising
   `denoise_dncnn_universal.py <denoise_dncnn_universal.py>`_
      Comparison of DnCNN Variants for Image Denoising


Organized by Optimization Algorithm
-----------------------------------

ADMM
^^^^

   `ct_abel_tv_admm.py <ct_abel_tv_admm.py>`_
      TV-Regularized Abel Inversion
   `ct_abel_tv_admm_tune.py <ct_abel_tv_admm_tune.py>`_
      Parameter Tuning for TV-Regularized Abel Inversion
   `ct_symcone_tv_padmm.py <ct_symcone_tv_padmm.py>`_
      TV-Regularized Cone Beam CT for Symmetric Objects
   `ct_astra_tv_admm.py <ct_astra_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (ASTRA Projector)
   `ct_tv_admm.py <ct_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Integrated Projector)
   `ct_astra_3d_tv_admm.py <ct_astra_3d_tv_admm.py>`_
      3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver)
   `ct_astra_weighted_tv_admm.py <ct_astra_weighted_tv_admm.py>`_
      TV-Regularized Low-Dose CT Reconstruction
   `ct_multi_tv_admm.py <ct_multi_tv_admm.py>`_
      TV-Regularized Sparse-View CT Reconstruction (Multiple Projectors)
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
      Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_tv_admm_tune.py <deconv_tv_admm_tune.py>`_
      Parameter Tuning for Image Deconvolution with TV Regularization (ADMM Solver)
   `deconv_microscopy_tv_admm.py <deconv_microscopy_tv_admm.py>`_
      Deconvolution Microscopy (Single Channel)
   `deconv_microscopy_allchn_tv_admm.py <deconv_microscopy_allchn_tv_admm.py>`_
      Deconvolution Microscopy (All Channels)
   `deconv_ppp_bm3d_admm.py <deconv_ppp_bm3d_admm.py>`_
      PPP (with BM3D) Image Deconvolution (ADMM Solver)
   `deconv_ppp_dncnn_admm.py <deconv_ppp_dncnn_admm.py>`_
      PPP (with DnCNN) Image Deconvolution (ADMM Solver)
   `deconv_ppp_bm4d_admm.py <deconv_ppp_bm4d_admm.py>`_
      PPP (with BM4D) Volume Deconvolution
   `diffusercam_tv_admm.py <diffusercam_tv_admm.py>`_
      TV-Regularized 3D DiffuserCam Reconstruction
   `sparsecode_nn_admm.py <sparsecode_nn_admm.py>`_
      Non-Negative Basis Pursuit DeNoising (ADMM)
   `sparsecode_conv_admm.py <sparsecode_conv_admm.py>`_
      Convolutional Sparse Coding (ADMM)
   `sparsecode_conv_md_admm.py <sparsecode_conv_md_admm.py>`_
      Convolutional Sparse Coding with Mask Decoupling (ADMM)
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
   `denoise_approx_tv_multi.py <denoise_approx_tv_multi.py>`_
      Denoising with Approximate Total Variation Proximal Operator
   `video_rpca_admm.py <video_rpca_admm.py>`_
      Video Decomposition via Robust PCA


Linearized ADMM
^^^^^^^^^^^^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       TV-Regularized CT Reconstruction (Multiple Algorithms)
    `denoise_tv_multi.py <denoise_tv_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising


Proximal ADMM
^^^^^^^^^^^^^

    `ct_astra_3d_tv_padmm.py <ct_astra_3d_tv_padmm.py>`_
       3D TV-Regularized Sparse-View CT Reconstruction (Proximal ADMM Solver)
    `deconv_tv_padmm.py <deconv_tv_padmm.py>`_
       Image Deconvolution with TV Regularization (Proximal ADMM Solver)
    `denoise_tv_multi.py <denoise_tv_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising
    `deconv_ppp_dncnn_padmm.py <deconv_ppp_dncnn_padmm.py>`_
       PPP (with DnCNN) Image Deconvolution (Proximal ADMM Solver)


Non-linear Proximal ADMM
^^^^^^^^^^^^^^^^^^^^^^^^

    `denoise_cplx_tv_nlpadmm.py <denoise_cplx_tv_nlpadmm.py>`_
       Complex Total Variation Denoising with NLPADMM Solver


PDHG
^^^^

    `ct_svmbir_tv_multi.py <ct_svmbir_tv_multi.py>`_
       TV-Regularized CT Reconstruction (Multiple Algorithms)
    `denoise_ptv_pdhg.py <denoise_ptv_pdhg.py>`_
       Polar Total Variation Denoising (PDHG)
    `denoise_tv_multi.py <denoise_tv_multi.py>`_
       Comparison of Optimization Algorithms for Total Variation Denoising
    `denoise_cplx_tv_pdhg.py <denoise_cplx_tv_pdhg.py>`_
       Complex Total Variation Denoising with PDHG Solver


PGM
^^^

   `deconv_ppp_bm3d_apgm.py <deconv_ppp_bm3d_apgm.py>`_
      PPP (with BM3D) Image Deconvolution (APGM Solver)
   `sparsecode_apgm.py <sparsecode_apgm.py>`_
      Basis Pursuit DeNoising (APGM)
   `sparsecode_nn_apgm.py <sparsecode_nn_apgm.py>`_
      Non-Negative Basis Pursuit DeNoising (APGM)
   `sparsecode_poisson_apgm.py <sparsecode_poisson_apgm.py>`_
      Non-negative Poisson Loss Reconstruction (APGM)
   `denoise_tv_apgm.py <denoise_tv_apgm.py>`_
      Total Variation Denoising with Constraint (APGM)
   `denoise_approx_tv_multi.py <denoise_approx_tv_multi.py>`_
      Denoising with Approximate Total Variation Proximal Operator


PCG
^^^

   `ct_astra_noreg_pcg.py <ct_astra_noreg_pcg.py>`_
      CT Reconstruction with CG and PCG
