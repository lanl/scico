#!/usr/bin/env python

# Construct an index README file and a docs example index file from
# source index file "scripts/index.rst".
# Run as
#     python makeindex.py


import re
from pathlib import Path

import py2jn
import pypandoc

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


# Build notebooks index file in notebooks directory
dst = "notebooks/index.ipynb"
rst_text = ""
with open(src, "r") as srcfile:
    for line in srcfile:
        # Detect lines containing script filenames
        m = re.match(r"(\s+)- ([^\s]+).py", line)
        if m:
            prespace = m.group(1)
            name = m.group(2)
            title = titles[name + ".py"]
            rst_text += "%s- `%s <%s.ipynb>`_\n" % (prespace, title, name)
        else:
            rst_text += line
# Convert text from rst to markdown
md_format = "markdown_github+tex_math_dollars+fenced_code_attributes"
md_text = pypandoc.convert_text(rst_text, md_format, format="rst", extra_args=["--atx-headers"])
md_text = '"""' + md_text + '"""'
# Convert from python to notebook format and write notebook
nb = py2jn.py_string_to_notebook(md_text)
py2jn.tools.write_notebook(nb, dst, nbver=4)


# Build examples index for docs
dst = "../docs/source/examples.rst"
prfx = "examples/"
with open(dst, "w") as dstfile:
    print(".. _example_notebooks:\n", file=dstfile)
    with open(src, "r") as srcfile:
        for line in srcfile:
            # Add toctree and include statements after main heading
            if line[0:3] == "===":
                print(line, end="", file=dstfile)
                print("\n.. toctree::\n   :maxdepth: 1", file=dstfile)
                print("\n.. include:: exampledepend.rst", file=dstfile)
                continue
            # Detect lines containing script filenames
            m = re.match(r"(\s+)- ([^\s]+).py", line)
            if m:
                print("   " + prfx + m.group(2), file=dstfile)
            else:
                print(line, end="", file=dstfile)
                # Add toctree statement after section headings
                if line[0:3] == line[0] * 3 and line[0] in ["=", "-", "^"]:
                    print("\n.. toctree::\n   :maxdepth: 1", file=dstfile)
