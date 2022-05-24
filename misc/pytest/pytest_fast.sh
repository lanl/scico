#!/usr/bin/env bash

# This script runs pytest tests in parallel using the pytest-xdist plugin.
# Some tests that do not function correctly when run in parallel are run
# separately. It must be run from the repository root directory.

plugin="pytest-xdist"
if ! pytest -VV | grep -o $plugin > /dev/null; then
    echo Required pytest plugin $plugin not installed
    exit 1
fi

pytest --deselect scico/test/test_ray_tune.py \
       --deselect scico/test/functional/test_core.py -x -n 2
pytest -x scico/test/test_ray_tune.py scico/test/functional/test_core.py

exit 0
