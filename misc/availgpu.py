#!/usr/bin/env python

# Determine which GPUs available for use and recommend CUDA_VISIBLE_DEVICES
# setting if any are already in use.

# pylint: disable=missing-module-docstring


import GPUtil

print("GPU utlizitation")
GPUtil.showUtilization()

devIDs = GPUtil.getAvailable(
    order="first", limit=65536, maxLoad=0.1, maxMemory=0.1, includeNan=False
)

Ngpu = len(GPUtil.getGPUs())
if len(devIDs) == Ngpu:
    print(f"All {Ngpu} GPUs available for use")
else:
    print(f"Only {len(devIDs)} of {Ngpu} GPUs available for use")
    print("To avoid attempting to use GPUs already in use, run the command")
    print(f"    export CUDA_VISIBLE_DEVICES={','.join(map(str, devIDs))}")
