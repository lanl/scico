Specialized Pytest Usage
========================

These scripts support specialized ``pytest`` usage:

- ``pytest_cov.sh``: This script runs ``scico`` unit tests using the ``pytest-cov`` plugin for test coverage analysis.
- ``pytest_fast.sh``: This script runs ``pytest`` tests in parallel using the ``pytest-xdist`` plugin. Some tests (those that do not function correctly when run in parallel) are run separately.
- ``pytest_time.sh``: This script runs each ``scico`` unit test module and lists them all in order of decreasing run time.

All of these scripts must be run from the repository root directory.
