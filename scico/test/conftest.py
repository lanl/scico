"""
Configure the --level pytest option and the x64 marker.
"""

import os

import jax

import pytest

# Make a request for a 64-bit dtype that is not available an error rather than
# a warning. Such a request indicates a test that needs 64-bit mode but does
# not enable it, which is otherwise easy to miss: the test continues to pass,
# because its assertions are self-consistent, while silently checking 32-bit
# behaviour instead. Skipped if the environment already selects a mode, so that
# the check can be turned off with JAX_EXPLICIT_X64_DTYPES=warn.
if "JAX_EXPLICIT_X64_DTYPES" not in os.environ:
    jax.config.update("jax_explicit_x64_dtypes", "error")


def pytest_addoption(parser, pluginmanager):
    """Add --level pytest option.

    Level definitions:
      1  Critical tests only
      2  Skip tests that have a significant impact on coverage
      3  All standard tests
      4  Run all tests, including those marked as slow to run
    """
    parser.addoption(
        "--level", action="store", default=3, type=int, help="Set test level to be run"
    )


def pytest_configure(config):
    """Add marker descriptions."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "x64: run the tests in the marked module with 64-bit mode enabled"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests depending on selected testing level."""
    if config.getoption("--level") >= 4:
        # don't skip tests at level 4 or higher
        return
    level_skip = pytest.mark.skip(reason="test not appropriate for selected level")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(level_skip)


@pytest.fixture(scope="module", autouse=True)
def enable_x64(request):
    """Enable 64-bit mode for the tests in modules marked with "x64".

    Some tests require 64-bit mode, most commonly because they check the
    dtype of an operator output. Enabling it via `jax.config.update` at
    module level would be a global change that persists for the remainder
    of the session, so that a test could behave differently when run on its
    own than when run after such a module. Enabling it via this fixture
    confines it to the modules that ask for it, leaving 32-bit defaults, as
    seen by users of the package, in place everywhere else.

    The fixture is autouse so that it is entered before any other fixture
    of the same scope, including module scope fixtures that construct the
    arrays on which the tests operate.
    """
    if request.node.get_closest_marker("x64") is None:
        yield
    else:
        with jax.enable_x64(True):
            yield
