# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Support functions for determining the package version."""

import os
import re
from ast import parse
from subprocess import PIPE, Popen
from typing import Any, Optional, Tuple, Union


def root_init_path() -> str:  # pragma: no cover
    """Get the path to the package root `__init__.py` file.

    Returns:
       Path to the package root `__init__.py` file.
    """
    return os.path.join(os.path.dirname(__file__), "__init__.py")


def variable_assign_value(path: str, var: str) -> Any:
    """Get variable initialization value from a Python file.

    Args:
        path: Path of Python file.
        var: Name of variable.

    Returns:
        Value to which variable `var` is initialized.

    Raises:
        RuntimeError: If the statement initializing variable `var` is not
           found.
    """
    with open(path) as f:
        try:
            # See https://stackoverflow.com/a/30471662
            value_obj = parse(next(filter(lambda line: line.startswith(var), f))).body[0].value  # type: ignore
            value = value_obj.value  # type: ignore
        except StopIteration:
            raise RuntimeError(f"Could not find initialization of variable {var}")
    return value


def init_variable_assign_value(var: str) -> Any:  # pragma: no cover
    """Get variable initialization value from package `__init__.py` file.

    Args:
        var: Name of variable.

    Returns:
        Value to which variable `var` is initialized.

    Raises:
        RuntimeError: If the statement initializing variable `var` is not
           found.
    """
    return variable_assign_value(root_init_path(), var)


def current_git_hash() -> Optional[str]:  # nosec  pragma: no cover
    """Get current short git hash.

    Returns:
       Short git hash of current commit, or ``None`` if no git repo found.
    """
    process = Popen(["git", "rev-parse", "--short", "HEAD"], shell=False, stdout=PIPE, stderr=PIPE)
    git_hash: Optional[str] = process.communicate()[0].strip().decode("utf-8")
    if git_hash == "":
        git_hash = None
    return git_hash


def package_version(split: bool = False) -> Union[str, Tuple[str, str]]:  # pragma: no cover
    """Get current package version.

    Args:
        split: Flag indicating whether to return the package version as a
           single string or split into a tuple of components.

    Returns:
        Package version string or tuple of strings.
    """
    version = init_variable_assign_value("__version__")
    # don't extend purely numeric version numbers, possibly ending with post<n>
    if re.match(r"^[0-9\.]+(post[0-9]+)?$", version):
        git_hash = None
    else:
        git_hash = current_git_hash()
    if git_hash:
        git_hash = "+" + git_hash
    else:
        git_hash = ""
    if split:
        version = (version, git_hash)
    else:
        version = version + git_hash
    return version
