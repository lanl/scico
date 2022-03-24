===================
SCICO Release Notes
===================


Version 0.0.3   (unreleased)
----------------------------

• Support for ``jaxlib`` versions 0.3.0 to 0.3.2 and ``jax`` versions
  0.3.0 to 0.3.4.
• Rename linear operators in ``radon_astra`` and ``radon_svmbir`` modules
  to ``TomographicProjector``.
• Add support for fan beam CT in ``radon_svmbir`` module.
• Add function ``linop.linop_from_function`` for constructing linear
  operators from functions.
• Add support for addition of functionals.
• Additional solvers in ``scico.solver``.
• New Huber norm and set distance functionals.



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
