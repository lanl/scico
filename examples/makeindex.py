#!/usr/bin/env python

# Construct an index README file and a docs example index file from
# source index file "scripts/index.rst".
# Run as
#     python makeindex.py


import re
from pathlib import Path

src = "scripts/index.rst"

# Make dict mapping script names to docstring header titles
titles = {}
scripts = list(Path("scripts").glob("*py"))
for s in scripts:
    prevline = None
    with open(s, "r") as sfile:
        for line in sfile:
            if line[0:3] == "===":
                titles[s.name] = prevline.rstrip()
                break
            else:
                prevline = line


# Build README in scripts directory
dst = "scripts/README.rst"
with open(dst, "w") as dstfile:
    with open(src, "r") as srcfile:
        for line in srcfile:
            # Detect lines containing script filenames
            m = re.match(r"(\s+)- ([^\s]+.py)", line)
            if m:
                prespace = m.group(1)
                name = m.group(2)
                title = titles[name]
                print(
                    "%s`%s <%s>`_\n%s   %s" % (prespace, name, name, prespace, title), file=dstfile
                )
            else:
                print(line, end="", file=dstfile)


# Build examples index for docs
dst = "../docs/source/examples.rst"
prfx = "examples/"
with open(dst, "w") as dstfile:
    print(".. _example_notebooks:\n", file=dstfile)
    with open(src, "r") as srcfile:
        for line in srcfile:
            # Detect lines containing script filenames
            m = re.match(r"(\s+)- ([^\s]+).py", line)
            if m:
                print(m.group(1) + prfx + m.group(2), file=dstfile)
            else:
                print(line, end="", file=dstfile)
                # Add toctree statements after section headings
                if line[0:3] == line[0] * 3 and line[0] in ["-", "="]:
                    print("\n.. toctree::\n   :maxdepth: 1", file=dstfile)
