"""
Configure the environment in which doctests run. This is necessary
because `np` is used in doc strings for jax functions
(e.g. `linear_transpose`) that get pulled into `scico/__init__.py`.

Also allows `snp` to be used without explicitly importing
"""

import numpy as np

import pytest

import scico.numpy as snp


@pytest.fixture(autouse=True)
def add_modules(doctest_namespace):
    """Add common modules for use in docstring examples."""
    doctest_namespace["np"] = np
    doctest_namespace["snp"] = snp
