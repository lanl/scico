"""
Configure the --level pytest option and its functionality.
"""

import pytest


def pytest_addoption(parser, pluginmanager):
    """Add --level pytest option.

    Level definitions:
      1  Critical tests only
      2  Skip tests that do have a significant impact on coverage
      3  All standard tests
      4  Run all tests, including those marked as slow to run
    """
    parser.addoption(
        "--level", action="store", default=3, type=int, help="Set test level to be run"
    )


def pytest_configure(config):
    """Add marker description."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests depending on selected testing level."""
    if config.getoption("--level") >= 4:
        # don't skip tests at level 4 or higher
        return
    level_skip = pytest.mark.skip(reason="test not appropriate for selected level")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(level_skip)
