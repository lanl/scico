"""SCICO package configuration."""

import importlib.util
import os
import os.path
import site
import sys

from setuptools import find_namespace_packages, setup

# Import module scico._version without executing __init__.py
spec = importlib.util.spec_from_file_location("_version", os.path.join("scico", "_version.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["_version"] = module
spec.loader.exec_module(module)
from _version import package_version

name = "scico"
version = package_version()
# Add argument exclude=["test", "test.*"] to exclude test subpackage
packages = find_namespace_packages(where="scico")
packages = ["scico"] + [f"scico.{m}" for m in packages]


longdesc = """
SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of JAX, which provides features such as automatic gradient calculation and GPU acceleration.
"""

# Set install_requires from requirements.txt file
with open("requirements.txt") as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines]

python_requires = ">=3.8"
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
    license="BSD-3-Clause",
    url="https://github.com/lanl/scico",
    author="SCICO Developers",
    author_email="brendt@ieee.org",  # Temporary
    packages=packages,
    package_data={"scico": ["data/*/*.png", "data/*/*.npz"]},
    include_package_data=True,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
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
