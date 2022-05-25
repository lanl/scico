#!/usr/bin/env bash

# This script runs scico unit tests using the pytest-cov plugin for test
# coverage analysis. It must be run from the repository root directory.

plugin="pytest-cov"
if ! pytest -VV | grep -o $plugin > /dev/null; then
    echo Required pytest plugin $plugin not installed
    exit 1
fi

pytest --cov=scico --cov-report html

echo "To view the report, open htmlcov/index.html in a web browser."

exit 0
