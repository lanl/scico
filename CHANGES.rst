===================
SCICO Release Notes
===================


Version 0.0.3   (2022-09-21)
----------------------------

• Change required packages and version numbers, including more recent version
  requirements for `numpy`, `scipy`, `svmbir`, and `ray`.
• Package `bm4d` removed from main requirements list due to issue #342.
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
