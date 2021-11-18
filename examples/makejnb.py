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

from py2jn.tools import py_string_to_notebook, write_notebook

try:
    import ray
except ImportError:
    raise RuntimeError("The ray package is required to run this script")



def py_file_to_string(src):
    """Preprocess example script file and return result as a string."""

    with open(src, "r") as srcfile:
        # Drop header comment
        for line in srcfile:
            if line[0] != "#":
                break  # assume first non-comment line is a newline that can be dropped
        # Insert notebook plot config after last import
        lines = []
        import_seen = False
        for line in srcfile:
            line = re.sub('^r"""', '"""', line)  # remove r from r"""
            line = re.sub(":cite:`([^`]+)`", r'<cite data-cite="\1"\/>', line)  # fix cite format
            if import_seen:
                # Once an import statement has been seen, break on encountering a line that
                # is neither an import statement not a newline
                if not re.match(r"(^import|^from|^\n$)", line):
                    lines.append(line)
                    break
            else:
                # Set flag indicating that an import statement has been seen once one has
                # been encountered
                if re.match("^(import|from)", line):
                    import_seen = True
            lines.append(line)
        # Backtrack through list of lines to find last import statement
        n = 1
        for line in lines[-2::-1]:
            if re.match("^(import|from)", line):
                break
            else:
                n += 1
        # Insert notebook plotting config directly after last import statement
        lines.insert(-n, "plot.config_notebook_plotting()\n")

        # Process remainder of source file
        for line in srcfile:
            if re.match("^input", line):  # end processing when input statement encountered
                break
            line = re.sub('^r"""', '"""', line)  # remove r from r"""
            line = re.sub(":cite:\`([^`]+)\`", r'<cite data-cite="\1"/>', line)  # fix cite format
            lines.append(line)

        # Backtrack through list of lines to remove trailing newlines
        n = 0
        for line in lines[::-1]:
            if re.match("^\n$", line):
                n += 1
            else:
                break
        lines = lines[0:-n]

        return "".join(lines)


def script_to_notebook(src, dst):
    """Convert a Python example script into a Jupyter notebook."""

    str = py_file_to_string(src)
    nb = py_string_to_notebook(str)
    write_notebook(nb, dst)




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
        script_to_notebook(s, nb)
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
