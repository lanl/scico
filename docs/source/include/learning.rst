Learned Models
==============

In SCICO, neural network models are used to represent imaging problems and provide different modes of data-driven regularization.
The models are implemented in `Flax <https://flax.readthedocs.io/>`_, and constitute a representative sample of frequently used networks.


FlaxMap
-------

SCICO interfaces with the implemented models via :class:`.FlaxMap`. This provides a standardized access to all trained models via the model definiton and the learned parameters. Further specialized functionality, such as learned denoisers, are built on top of :class:`.FlaxMap`. The specific models that have been implemented are described below.



DnCNN
-----

The denoiser convolutional neural network model (DnCNN) :cite:`zhang-2017-dncnn`, implemented as :class:`.DnCNNNet`, is used to denoise images that have been corrupted with additive Gaussian noise.



ODP
---

The unrolled optimization with deep priors (ODP) :cite:`diamond-2018-odp`, implemented as :class:`.ODPNet`, is used to solve inverse problems in imaging by adapting classical iterative methods into an end-to-end framework that incorporates deep networks as well as knowledge of the image formation model.

The framework aims to solve the optimization problem

.. math::
   \argmin_{\mb{x}} \; f(A \mb{x}, \mb{y}) + r(\mb{x}) \;,

where :math:`A` represents a linear forward model and :math:`r` a regularization function encoding prior information, by unrolling the iterative solution method into a network where each iteration corresponds to a different stage in the ODP network. Different iterative solutions produce different unrolled optimization algorithms which, in turn, produce different ODP networks. The ones implemented in SCICO are described below.


Proximal Map
^^^^^^^^^^^^

This algorithm corresponds to solving

.. math::
   :label: eq:odp_prox

   \argmin_{\mb{x}} \; \alpha_k \, f(A \mb{x}, \mb{y}) + \frac{1}{2} \| \mb{x} - \mb{x}^k - \mb{x}^{k+1/2} \|_2^2 \;,

with :math:`k` corresponding to the index of the iteration, which translates to an index of the stage of the network, :math:`f(A \mb{x}, \mb{y})` a fidelity term, usually an :math:`\ell_2` norm, and :math:`\mb{x}^{k+1/2}` a regularization representing :math:`\mathrm{prox}_r (\mb{x}^k)` and usually implemented as a convolutional neural network (CNN). This proximal map representation is used when minimization problem :eq:`eq:odp_prox` can be solved in a computationally efficient manner.

:class:`.ODPProxDnBlock` uses this formulation to solve a denoising problem, which, according to :cite:`diamond-2018-odp`, can be solved by

.. math::
   \mb{x}^{k+1} = (\alpha_k \, \mb{y} + \mb{x}^k + \mb{x}^{k+1/2}) \, / \, (\alpha_k + 1) \;,

where :math:`A` corresponds to the identity operator and is therefore omitted, :math:`\mb{y}` is the noisy signal, :math:`\alpha_k > 0` is a learned stage-wise parameter weighting the contribution of the fidelity term and :math:`\mb{x}^k + \mb{x}^{k+1/2}` is the regularization, usually represented by a residual CNN.


:class:`.ODPProxDblrBlock` uses this formulation to solve a deblurring problem, which, according to :cite:`diamond-2018-odp`, can be solved by

.. math::
   \mb{x}^{k+1} = \mathcal{F}^{-1} \mathrm{diag} (\alpha_k | \mathcal{F}(K)|^2 + 1 )^{-1} \mathcal{F} \, (\alpha_k K^T * \mb{y} + \mb{x}^k + \mb{x}^{k+1/2}) \;,

where :math:`A` is the blurring operator, :math:`K` is the blurring kernel, :math:`\mb{y}` is the blurred signal, :math:`\mathcal{F}` is the DFT, :math:`\alpha_k > 0` is a learned  stage-wise parameter weighting the contribution of the fidelity term and :math:`\mb{x}^k + \mb{x}^{k+1/2}` is the regularization represented by a residual CNN.


Gradient Descent
^^^^^^^^^^^^^^^^

When the solution of the optimization problem in :eq:`eq:odp_prox` can not be simply represented by an analytical step, a formulation based on a gradient descent iteration is preferred. This yields

.. math::
   \mb{x}^{k+1} = \mb{x}^k + \mb{x}^{k+1/2} - \alpha_k \, A^T \nabla_x \, f(A \mb{x}^k, \mb{y}) \;,

where :math:`\mb{x}^{k+1/2}` represents :math:`\nabla r(\mb{x}^k)`.

:class:`.ODPGrDescBlock` uses this formulation to solve a generic problem with :math:`\ell_2` fidelity as

.. math::
   \mb{x}^{k+1} = \mb{x}^k + \mb{x}^{k+1/2} - \alpha_k \, A^T (A \mb{x} - \mb{y}) \;,

with :math:`\mb{y}` the measured signal and :math:`\mb{x} + \mb{x}^{k+1/2}` a residual CNN.


MoDL
----

The model-based deep learning (MoDL) :cite:`aggarwal-2019-modl`, implemented as :class:`.MoDLNet`, is used to solve inverse problems in imaging also by adapting classical iterative methods into an end-to-end deep learning framework, but, in contrast to ODP, it solves the optimization problem

.. math::
   \argmin_{\mb{x}} \; \| A \mb{x} - \mb{y}\|_2^2 + \lambda \, \| \mb{x} - \mathrm{D}_w(\mb{x})\|_2^2 \;,

by directly computing the update

.. math::
   \mb{x}^{k+1} = (A^T A + \lambda \, I)^{-1} (A^T \mb{y} + \lambda \, \mb{z}^k) \;,

via conjugate gradient. The regularization :math:`\mb{z}^k = \mathrm{D}_w(\mb{x}^{k})` incorporates prior information, usually in the form of a denoiser model. In this case, the denoiser :math:`\mathrm{D}_w` is shared between all the stages of the network requiring relatively less memory than other unrolling methods. This also allows for deploying a different number of iterations in testing than the ones used in training.
