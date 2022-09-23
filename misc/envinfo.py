#!/usr/bin/env python

# Print host and environment information. Useful for determining whether
# a Python host has available GPUs, and if so, whether the JAX installation
# is able to make use of them.

# pylint: disable=missing-module-docstring

import sys

missing = []

try:
    import psutil

    have_psutil = True
except ImportError:
    have_psutil = False
    missing.append("psutil")

try:
    import GPUtil

    have_gputil = True
except ImportError:
    have_gputil = False
    missing.append("gputil")

import jax

import jaxlib

try:
    import scico

    have_scico = True
except ImportError:
    scico = None
    have_scico = False
    missing.append("scico")


if missing:
    print("Some output not available due to missing modules: " + ", ".join(missing))

pyver = ".".join([f"{v}" for v in sys.version_info[0:3]])
print(f"Python version: {pyver}")
print("Packages:")
packages = [jaxlib, jax, scico]
for p in packages:
    if hasattr(p, "__version__") and hasattr(p, "__name__"):
        v = getattr(p, "__version__")
        n = getattr(p, "__name__")
        print(f"    {n:15s} {v}")

if have_psutil:
    print(f"Number of CPU cores: {psutil.cpu_count(logical=False)}")

if have_gputil:
    if GPUtil.getAvailable():
        print("GPUs:")
        for gpu in GPUtil.getGPUs():
            print(f"    {gpu.id:2d}  {gpu.name:10s}  {gpu.memoryTotal} kB RAM")
    else:
        print("No GPUs available")

sys.stderr = open("/dev/null")  # suppress annoying jax warning
numdev = jax.device_count()
if jax.devices()[0].device_kind == "cpu":
    print("No GPUs available to JAX (JAX device is CPU)")
else:
    print(f"Number of GPUs available to JAX: {jax.device_count()}")
