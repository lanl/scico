#!/usr/bin/env python

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    import ray
except ImportError:
    raise RuntimeError("The ray package is required to run this script")
import os

# source scripts
scripts = list(Path("scripts").glob("*py"))
notebooks = []
# construct list of scripts that are have no corresponding notebook or
# are more recent than corresponding notebook
for s in scripts:
    nb = Path("notebooks") / (s.stem + ".ipynb")
    if not nb.is_file() or s.stat().st_mtime > nb.stat().st_mtime:
        # make notebook file
        os.popen(f"./pytojnb.sh {s} {nb}")
        # add it to the list for execution
        notebooks.append(nb)

ray.init()

ngpu = 0
cr = ray.cluster_resources()
if "GPU" in cr:
    ngpu = cr["GPU"]
if ngpu == 0:
    print("Warning: host has no GPUs")
else:
    ngpu = 0
    ar = ray.available_resources()
    if "GPU" in cr:
        ngpu = int(ar["GPU"])
    if ngpu < 2:
        print("Warning: host has fewer than two GPUs available")
    print(f"Executing on {ngpu} GPUs")

# function to execute each notebook with one gpu each
@ray.remote(num_gpus=1)
def run_nb(fname):
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor()
    try:
        out = ep.preprocess(nb)
        with open(fname, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except CellExecutionError:
        raise Exception(f"Error executing the notebook {fname}")
    print(f"{fname} done")


# run all; blocking
ray.get([run_nb.remote(_) for _ in notebooks])
