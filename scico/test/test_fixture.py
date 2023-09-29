import jax

import pytest


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    print("Before: ", jax.config.read("jax_enable_x64"))

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    print("After: ", jax.config.read("jax_enable_x64"))
