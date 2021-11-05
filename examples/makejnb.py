#!/usr/bin/env python

# Extract a list of Python scripts from "scripts/index.rst" and
# create/update and execute any Jupyter notebooks that are out
# of date with respect to their source Python scripts. If script
# names specified on command line, process them instead.
# Run as
#     python makejnb.py [script_name_1 [script_name_2 [...]]]

import os
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    import ray
except ImportError:
    raise RuntimeError("The ray package is required to run this script")


if sys.argv[1:]:
    # Script names specified on command line
    scriptnames = [os.path.basename(s) for s in sys.argv[1:]]
else:
    # Read script names from index file
    scriptnames = []
    srcidx = "scripts/index.rst"
    with open(srcidx, "r") as idxfile:
        for line in idxfile:
            m = re.match(r"(\s+)- ([^\s]+.py)", line)
            if m:
                scriptnames.append(m.group(2))

# Ensure list entries are unique
scriptnames = list(set(scriptnames))

# Construct script paths
scripts = [Path("scripts") / Path(s) for s in scriptnames]


# Construct list of notebooks that are out of date with respect to the corresponding
# script, or that have not yet been constructed from the corresponding script, and
# construct/update each of them
notebooks = []
for s in scripts:
    nb = Path("notebooks") / (s.stem + ".ipynb")
    if not nb.is_file() or s.stat().st_mtime > nb.stat().st_mtime:
        # Make notebook file
        os.popen(f"./pytojnb.sh {s} {nb}")
        # Add it to the list for execution
        notebooks.append(nb)
if sys.argv[1:]:
    # If scripts specified on command line, add all corresonding notebooks to the
    # list for execution
    notebooks = [Path("notebooks") / (s.stem + ".ipynb") for s in scripts]

ray.init()

nproc = len(notebooks)
ngpu = 0
ar = ray.available_resources()
ncpu = max(int(ar["CPU"]) // nproc, 1)
if "GPU" in ar:
    ngpu = max(int(ar["GPU"]) // nproc, 1)
print(f"Running on {ncpu} CPUs and {ngpu} GPUs per process")

# Function to execute each notebook with one GPU
@ray.remote(num_cpus=ncpu, num_gpus=ngpu)
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
