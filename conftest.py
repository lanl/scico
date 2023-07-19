"""
Configure the environment in which doctests run. This is necessary
because `np` is used in doc strings for jax functions
(e.g. `linear_transpose`) that get pulled into `scico/__init__.py`.

Also allow `snp` to be used without explicitly importing, and add
`level` parameter.
"""

import numpy as np

import pytest

import scico.numpy as snp


@pytest.fixture(autouse=True)
def add_modules(doctest_namespace):
    """Add common modules for use in docstring examples."""
    doctest_namespace["np"] = np
    doctest_namespace["snp"] = snp


def pytest_addoption(parser, pluginmanager):
    """Add --level pytest option.

    Level definitions:
      1  Critical tests only
      2  Skip tests that do have a significant impact on coverage
      3  All tests
    """
    parser.addoption(
        "--level", action="store", default=3, type=int, help="Set test level to be run"
    )
