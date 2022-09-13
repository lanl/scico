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
import sys
from pathlib import Path

import psutil
from jnb import execute_notebook, script_to_notebook

examples_dir = Path(__file__).resolve().parent  # absolute path to ../scico/examples/

have_ray = True
try:
    import ray
except ImportError:
    have_ray = False


def script_uses_ray(fname):
    """Determine whether a script uses ray."""

    with open(fname, "r") as f:
        text = f.read()
    return bool(re.search("^import ray", text, re.MULTILINE)) or bool(
        re.search("^import scico.ray", text, re.MULTILINE)
    )


def script_path(sname):
    """Get script path from script name."""

    return examples_dir / "scripts" / Path(sname)


def notebook_path(sname):
    """Get notebook path from script path."""

    return examples_dir / "notebooks" / Path(Path(sname).stem + ".ipynb")


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
    "--no-exec", action="store_true", help="Create/update notebooks but don't execute them."
)
argparser.add_argument(
    "--no-ray",
    action="store_true",
    help="Execute notebooks serially, without the use of ray parallelization.",
)
argparser.add_argument(
    "--verbose",
    action="store_true",
    help="Verbose operation.",
)
argparser.add_argument(
    "--test",
    action="store_true",
    help="Show actions that would be taken but don't do anything.",
)
argparser.add_argument("filename", nargs="*", help="Optional Python example script filenames")
args = argparser.parse_args()


# Raise error if ray needed but not present
if not have_ray and not args.no_ray:
    raise RuntimeError("The ray package is required to run this script, try --no-ray")


if args.filename:
    # Script names specified on command line
    scriptnames = [os.path.basename(s) for s in args.filename]
else:
    # Read script names from index file
    scriptnames = []
    srcidx = examples_dir / "scripts" / "index.rst"
    with open(srcidx, "r") as idxfile:
        for line in idxfile:
            m = re.match(r"(\s+)- ([^\s]+.py)", line)
            if m:
                scriptnames.append(m.group(2))

# Ensure list entries are unique
scriptnames = sorted(list(set(scriptnames)))

# Create list of selected scripts.
scripts = []
for s in scriptnames:
    spath = script_path(s)
    npath = notebook_path(s)
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
        scripts.append(s)

if not scripts:
    if args.verbose:
        print("No scripts require conversion")
    sys.exit(0)

# Display status information
if args.verbose:
    print(f"Processing scripts {', '.join(scripts)}")

# Convert selected scripts to corresponding notebooks and determine which can be run in parallel
serial_scripts = []
parallel_scripts = []
for s in scripts:
    spath = script_path(s)
    npath = notebook_path(s)
    # Determine how script should be executed
    if script_uses_ray(spath):
        serial_scripts.append(s)
    else:
        parallel_scripts.append(s)
    # Make notebook file
    if args.verbose or args.test:
        print(f"Converting script {s} to notebook")
    if not args.test:
        script_to_notebook(spath, npath)

if args.no_exec:
    if args.verbose:
        print("Notebooks will not be executed")
    sys.exit(0)


# If ray disabled or not worth using, run all serially
if args.no_ray or len(parallel_scripts) < 2:
    serial_scripts.extend(parallel_scripts)
    parallel_scripts = []

# Execute notebooks corresponding to serial_scripts
for s in serial_scripts:
    npath = notebook_path(s)
    if args.verbose or args.test:
        print(f"Executing notebook corresponding to script {s}")
    if not args.test:
        execute_notebook(npath)


# Execute notebooks corresponding to parallel_scripts
if parallel_scripts:
    if args.verbose or args.test:
        print(
            f"Notebooks corresponding to scripts {', '.join(parallel_scripts)} will "
            "be executed in parallel"
        )

    # Execute notebooks in parallel using ray
    nproc = len(parallel_scripts)
    ray.init()

    ngpu = 0
    ar = ray.available_resources()
    ncpu = max(int(ar["CPU"]) // nproc, 1)
    if "GPU" in ar:
        ngpu = max(int(ar["GPU"]) // nproc, 1)
    if args.verbose or args.test:
        print(f"    Running on {ncpu} CPUs and {ngpu} GPUs per process")

    # Function to execute each notebook with available resources suitably divided
    @ray.remote(num_cpus=ncpu, num_gpus=ngpu)
    def ray_run_nb(fname):
        execute_notebook(fname)

    if not args.test:
        # Execute relevant notebooks in parallel
        try:
            notebooks = [notebook_path(s) for s in parallel_scripts]
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
