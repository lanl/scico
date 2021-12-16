#!/usr/bin/env python

# Extract a list of Python scripts from "scripts/index.rst" and
# create/update and execute any Jupyter notebooks that are out
# of date with respect to their source Python scripts. If script
# names specified on command line, process them instead.
# Run
#     python makenotebooks.py -h
# for usage details.

import argparse
import os
import re
import signal
from pathlib import Path
from timeit import default_timer as timer

import nbformat
import psutil
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from py2jn.tools import py_string_to_notebook, write_notebook

have_ray = True
try:
    import ray
except ImportError:
    have_ray = False


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
            line = re.sub(":cite:`([^`]+)`", r'<cite data-cite="\1"/>', line)  # fix cite format
            if import_seen:
                # Once an import statement has been seen, break on encountering a line that
                # is neither an import statement nor a newline, nor a component of an import
                # statement extended over multiple lines, nor components of a try/except
                # construction (note that handling of these final two cases is probably not
                # very robust).
                if not re.match(r"(^import|^from|^\n$|^\W+[^\W]|^\)$|^try:$|^except)", line):
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

    s = py_file_to_string(src)
    nb = py_string_to_notebook(s)
    write_notebook(nb, dst)


def execute_notebook(fname):
    """Execute the specified notebook file."""

    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor()
    try:
        t0 = timer()
        out = ep.preprocess(nb)
        t1 = timer()
        with open(fname, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except CellExecutionError:
        print(f"ERROR executing {fname}")
        return False
    print(f"{fname} done in {(t1 - t0):.1e} s")
    return True


argparser = argparse.ArgumentParser(
    description="Convert Python example scripts to Jupyter notebooks."
)
argparser.add_argument(
    "--all",
    action="store_true",
    help="Process all notebooks, without  checking timestamps. "
    "Has no effect when files to process are explicitly specified.",
)
argparser.add_argument(
    "--no-exec", action="store_true", help="Create/update notebooks but don't execute them"
)
argparser.add_argument(
    "--no-ray",
    action="store_true",
    help="Execute notebooks serially, without the use of ray parallelization",
)
argparser.add_argument("filename", nargs="*", help="Optional Python example script filenames")
args = argparser.parse_args()

# Raise error if ray needed but not present
if not have_ray and not args.no_ray:
    raise RuntimeError("The ray package is required to run this script")


if args.filename:
    # Script names specified on command line
    scriptnames = [os.path.basename(s) for s in args.filename]
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
scriptnames = sorted(list(set(scriptnames)))

# Creat list of selected scripts and corresponding notebooks.
scripts = []
notebooks = []
for s in scriptnames:
    sb = Path(s).stem
    spath = Path("scripts") / Path(sb + ".py")
    npath = Path("notebooks") / Path(sb + ".ipynb")
    # If scripts specified on command line or --all flag specified, convert all scripts.
    # Otherwise, only convert scripts that have a newer timestamp than their corresponding
    # notebooks, or that have not previously been converted (i.e. corresponding notebook
    # file does not exist).
    if (
        args.all
        or args.filename
        or not npath.is_file()
        or spath.stat().st_mtime > npath.stat().st_mtime
    ):
        # Add to the list of selected scripts
        scripts.append(spath)
        # Add to the list of selected notebooks
        notebooks.append(npath)

# Display status information
print(f"Processing scripts {', '.join([os.path.basename(s) for s in scriptnames])}")

# Convert selected scripts to corresponding notebooks and create list of new/modified notebooks.
for spath in scripts:
    npath = Path("notebooks") / Path(spath.stem + ".ipynb")
    # Make notebook file
    script_to_notebook(spath, npath)

# Run relevant notebooks if no excecution flag not specified and notebooks list is not empty
if not args.no_exec and notebooks:
    nproc = len(notebooks)

    # Execute notebooks serially if requested to avoid use of ray, or if only one notebook
    if args.no_ray or nproc < 2:

        for nbfile in notebooks:
            execute_notebook(nbfile)

    # Execute notebooks in parallel using ray
    else:
        ray.init()

        ngpu = 0
        ar = ray.available_resources()
        ncpu = max(int(ar["CPU"]) // nproc, 1)
        if "GPU" in ar:
            ngpu = max(int(ar["GPU"]) // nproc, 1)
        print(f"Running on {ncpu} CPUs and {ngpu} GPUs per process")

        # Function to execute each notebook with available resources suitably divided
        @ray.remote(num_cpus=ncpu, num_gpus=ngpu)
        def ray_run_nb(fname):
            execute_notebook(fname)

        # Execute relevant notebooks in parallel
        try:
            objrefs = [ray_run_nb.remote(nbfile) for nbfile in notebooks]
            ray.wait(objrefs, num_returns=len(objrefs))
        except KeyboardInterrupt:
            print("\nTerminating on keyboard interrupt")
            for ref in objrefs:
                ray.cancel(ref, force=True)
            ray.shutdown()
            # Clean up sub-processes not ended by ray.cancel
            process = psutil.Process()
            children = process.children(recursive=True)
            for child in children:
                os.kill(child.pid, signal.SIGTERM)
