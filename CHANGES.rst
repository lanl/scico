===================
SCICO Release Notes
===================


Version 0.0.7   (2025-12-09)
----------------------------

• New module ``scico.trace`` for tracing function/method calls.
• New generic functional ``functional.ComposedFunctional`` representing
  a functional composed with an orthogonal linear operator.
• New optimizer methods ``save_state`` and ``load_state`` supporting
  algorithm state checkpointing.
• New classes for creating a volume from an image by symmetry, and
  for cone beam X-ray transform of a cylindrically symmetric object
  in module ``linop.xray.symcone``.
• New utility functions for CT reconstruction preprocessing added in
  module ``linop.xray``.
• Moved ``linop.abel`` module to ``linop.xray.abel``.
• Make ``orbax-checkpoint`` dependency optional due to absence of recent
  conda-forge packages.
• Support ``jaxlib`` and ``jax`` versions 0.4.13 to 0.8.1.
• Support ``flax`` versions 0.8.0 to 0.12.0



Version 0.0.6   (2024-10-25)
----------------------------

• Significant changes to ``linop.xray.astra`` API.
• Rename integrated 2D X-ray transform class to
  ``linop.xray.XRayTransform2D`` and add filtered back projection method
  ``fbp``.
• New integrated 3D X-ray transform via ``linop.xray.XRayTransform3D``.
• New functional ``functional.IsotropicTVNorm`` and faster implementation
  of ``functional.AnisotropicTVNorm``.
• New linear operators ``linop.ProjectedGradient``, ``linop.PolarGradient``,
  ``linop.CylindricalGradient``, and ``linop.SphericalGradient``.
• Rename ``scico.numpy.util.parse_axes`` to
  ``scico.numpy.util.normalize_axes``.
• Rename ``scico.flax.save_weights`` and ``scico.flax.load_weights`` to
  ``scico.flax.save_variables`` and ``scico.flax.load_variables``
  respectively.
• Support ``jaxlib`` and ``jax`` versions 0.4.13 to 0.4.35.
• Support ``flax`` versions 0.8.0 to 0.10.0.



Version 0.0.5   (2023-12-18)
----------------------------

• New functionals ``functional.AnisotropicTVNorm`` and
  ``functional.ProximalAverage`` with proximal operator approximations.
• New integrated Radon/X-ray transform ``linop.XRayTransform``.
• New operators ``operator.DiagonalStack`` and ``operator.VerticalStack``.
• Rename modules ``radon_astra`` and ``radon_svmbir`` to ``xray.astra`` and
  ``xray.svmbir`` respectively, and rename ``TomographicProjector`` classes
  to ``XRayTransform``.
• Rename ``AbelProjector`` to ``AbelTransform``.
• Rename ``solver.ATADSolver`` to ``solver.MatrixATADSolver``.
• Rename some ``__init__`` parameters of ``linop.DiagonalStack`` and
  ``linop.VerticalStack``.
• Support ``jaxlib`` and ``jax`` versions 0.4.3 to 0.4.23.
• Support ``flax`` versions up to 0.7.5.
• Use ``orbax`` for checkpointing ``flax`` models.



Version 0.0.4   (2023-08-03)
----------------------------

• Add new ``Function`` class for representing array-to-array mappings with
  more than one input.
• Add new methods and a function for computing Jacobian-vector products for
  ``Operator`` objects.
• Add new proximal ADMM solvers.
• Add new ADMM subproblem solvers for problems involving a sum-of-convolutions
  operator.
• Extend support for other ML models including UNet, ODP and MoDL.
• Add functionality for training Flax-based ML models and for data generation.
• Enable diagnostics for ML training loops.
• Support ``jaxlib`` and ``jax`` versions 0.4.3 to 0.4.14.
• Change required packages and version numbers, including more recent version
  for ``flax``.
• Drop support for Python 3.7.
• Add support for 3D tomographic projection with the ASTRA Toolbox.



Version 0.0.3   (2022-09-21)
----------------------------

• Change required packages and version numbers, including more recent version
  requirements for ``numpy``, ``scipy``, ``svmbir``, and ``ray``.
• Package ``bm4d`` removed from main requirements list due to issue #342.
• Support ``jaxlib`` versions 0.3.0 to 0.3.15 and ``jax`` versions
  0.3.0 to 0.3.17.
• Rename linear operators in ``radon_astra`` and ``radon_svmbir`` modules
  to ``TomographicProjector``.
• Add support for fan beam CT in ``radon_svmbir`` module.
• Add function ``linop.linop_from_function`` for constructing linear
  operators from functions.
• Enable addition operator for functionals.
• Completely new implementation of ``BlockArray`` class.
• Additional solvers in ``scico.solver``.
• New Huber norm (``HuberNorm``) and set distance functionals (``SetDistance``
  and ``SquaredSetDistance``).
• New loss functions ``loss.SquaredL2AbsLoss`` and
  ``loss.SquaredL2SquaredAbsLoss`` for phase retrieval problems.
• Add interface to BM4D denoiser.
• Change interfaces of ``linop.FiniteDifference`` and ``linop.DFT``.
• Change filenames of some example scripts (and corresponding notebooks).
• Add support for Python 3.7.
• New ``DiagonalStack`` linear operator.
• Add support for non-linear operators to ``optimize.PDHG`` optimizer class.
• Various bug fixes.



Version 0.0.2   (2022-02-14)
----------------------------

• Additional optimization algorithms: Linearized ADMM and PDHG.
• Additional Abel transform and array slicing linear operators.
• Additional nuclear norm functional.
• New module ``scico.ray.tune`` providing a simplified interface to Ray Tune.
• Move optimization algorithms into ``optimize`` subpackage.
• Additional iteration stats columns for iterative ADMM subproblem solvers.
• Renamed "Primal Rsdl" to "Prml Rsdl" in displayed iteration stats.
• Move some functions from ``util`` and ``math`` modules to new ``array``
  module.
• Bump pinned ``jaxlib`` and ``jax`` versions to 0.3.0.


Version 0.0.1   (2021-11-24)
----------------------------

• Initial release.
