#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCICO package configuration."""

import importlib.util
import os
import os.path
import re
import site
import sys

from setuptools import find_packages, setup

# Import module scico._version without executing __init__.py
spec = importlib.util.spec_from_file_location("_version", os.path.join("scico", "_version.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["_version"] = module
spec.loader.exec_module(module)
from _version import init_variable_assign_value, package_version

name = "scico"
version = package_version()
packages = find_packages()

longdesc = """
SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of JAX, which provides features such as automatic gradient calculation and GPU acceleration.
"""

# Set install_requires from requirements.txt file
with open("requirements.txt") as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines]

# Check that jaxlib version requirements in __init__.py and requirements.txt match
jaxlib_ver = init_variable_assign_value("jaxlib_ver_req")
jaxlib_req_str = list(filter(lambda s: s.startswith("jaxlib"), install_requires))[0]
m = re.match("jaxlib[=<>]+([\d\.]+)", jaxlib_req_str)
if not m:
    raise ValueError(f"Could not extract jaxlib version number from specification {jaxlib_req_str}")
req_jaxlib_ver = m[1]
if jaxlib_ver != req_jaxlib_ver:
    raise ValueError(
        f"Version requirements for jaxlib in __init__.py ({jaxlib_ver}) and "
        f"requirements.txt ({req_jaxlib_ver}) do not match"
    )

# Check that jax version requirements in __init__.py and requirements.txt match
jax_ver = init_variable_assign_value("jax_ver_req")
jax_req_str = list(
    filter(lambda s: s.startswith("jax") and not s.startswith("jaxlib"), install_requires)
)[0]
m = re.match("jax[=<>]+([\d\.]+)", jax_req_str)
if not m:
    raise ValueError(f"Could not extract jax version number from specification {jax_req_str}")
req_jax_ver = m[1]
if jax_ver != req_jax_ver:
    raise ValueError(
        f"Version requirements for jax in __init__.py ({jax_ver}) and "
        f"requirements.txt ({req_jax_ver}) do not match"
    )

python_requires = ">=3.7"
tests_require = ["pytest", "pytest-runner"]

extra_require_files = [
    "dev_requirements.txt",
    os.path.join("docs", "docs_requirements.txt"),
    os.path.join("examples", "examples_requirements.txt"),
    os.path.join("examples", "notebooks_requirements.txt"),
]
extras_require = {"tests": tests_require}
for require_file in extra_require_files:
    extras_label = os.path.basename(require_file).partition("_")[0]
    with open(require_file) as f:
        lines = f.readlines()
    extras_require[extras_label] = [line.strip() for line in lines if line[0:2] != "-r"]

# PEP517 workaround, see https://www.scivision.dev/python-pip-devel-user-install/
site.ENABLE_USER_SITE = True

setup(
    name=name,
    version=version,
    description="Scientific Computational Imaging COde: A Python "
    "package for scientific imaging problems",
    long_description=longdesc,
    keywords=[
        "Computational Imaging",
        "Scientific Imaging",
        "Inverse Problems",
        "Plug-and-Play Priors",
        "Total Variation",
        "Optimization",
        "ADMM",
        "Linearized ADMM",
        "PDHG",
        "PGM",
    ],
    platforms="Any",
    license="BSD",
    url="https://github.com/lanl/scico",
    author="SCICO Developers",
    author_email="brendt@ieee.org",  # Temporary
    packages=packages,
    package_data={"scico": ["data/*/*.png", "data/*/*.npz"]},
    include_package_data=True,
    python_requires=python_requires,
    tests_require=tests_require,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
)
