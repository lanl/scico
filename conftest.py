"""
Configure pytest.
"""

import os

import numpy as np

import pytest

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # suppress ray warning
try:
    import ray  # noqa: F401
except ImportError:
    have_ray = False
else:
    have_ray = True
    ray.init(num_cpus=1)  # call required to be here: see ray-project/ray#44087

import jax.numpy as jnp

import scico.numpy as snp


def pytest_sessionstart(session):
    """Initialize before start of test session."""
    # placeholder: currently unused


def pytest_sessionfinish(session, exitstatus):
    """Clean up after end of test session."""
    if have_ray:
        ray.shutdown()


@pytest.fixture(autouse=True)
def add_modules(doctest_namespace):
    """Add common modules for use in docstring examples.

    Necessary because `np` is used in doc strings for jax functions
    (e.g. `linear_transpose`) that get pulled into `scico/__init__.py`.
    Also allow `snp` and `jnp` to be used without explicitly importing.
    """
    doctest_namespace["np"] = np
    doctest_namespace["snp"] = snp
    doctest_namespace["jnp"] = jnp
