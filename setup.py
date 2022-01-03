#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCICO package configuration."""

import os
import os.path
from ast import parse

from setuptools import find_packages, setup

name = "scico"


def get_init_variable_value(var):
    """
    Get version number from scico/__init__.py
         See http://stackoverflow.com/questions/2058802
    """

    with open(os.path.join(name, "__init__.py")) as f:
        try:
            value = parse(next(filter(lambda line: line.startswith(var), f))).body[0].value.s
        except StopIteration:
            raise RuntimeError(f"Could not find initialization of variable {var}")
    return value


version = get_init_variable_value("__version_")

packages = find_packages()

longdesc = """
SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of JAX, which provides features such as automatic gradient calculation and GPU acceleration.
"""

# Set install_requires from requirements.txt file
with open(os.path.join("requirements.txt")) as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines]

# Check that jaxlib version requirements in __init__.py and requirements.txt match
jaxlib_ver = get_init_variable_value("jaxlib_ver_req")
req_jaxlib_ver = list(filter(lambda s: s.startswith("jaxlib"), install_requires))[0].split("==")[1]
if jaxlib_ver != req_jaxlib_ver:
    raise ValueError(
        f"Version requirements for jaxlib in __init__.py ({jaxlib_ver}) and "
        f"requirements.txt ({req_jaxlib_ver}) do not match"
    )

# Check that jax version requirements in __init__.py and requirements.txt match
jax_ver = get_init_variable_value("jax_ver_req")
req_jax_ver = list(
    filter(lambda s: s.startswith("jax") and not s.startswith("jaxlib"), install_requires)
)[0].split("==")[1]
if jax_ver != req_jax_ver:
    raise ValueError(
        f"Version requirements for jax in __init__.py ({jax_ver}) and "
        f"requirements.txt ({req_jax_ver}) do not match"
    )

tests_require = ["pytest", "pytest-runner"]
python_requires = ">=3.8"


setup(
    name=name,
    version=version,
    description="Scientific Computational Imaging COde: A Python "
    "package for scientific imaging problems",
    long_description=longdesc,
    keywords=["Computational Imaging", "Inverse Problems", "Optimization", "ADMM", "PGM"],
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
    extras_require={"tests": tests_require},
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 3 - Alpha",
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
