#!/usr/bin/env python

import re

src = "scripts/README.rst"
dst = "../docs/source/examples.rst"
prfx = "examples/"

with open(dst, "w") as dstfile:
    print(".. _example_notebooks:\n", file=dstfile)
    with open(src, "r") as srcfile:
        for line in srcfile:
            m = re.match(r"(\s+)- ([^\s]+).py", line)
            if m:
                print(m.group(1) + prfx + m.group(2), file=dstfile)
            else:
                print(line, end="", file=dstfile)
                if line[0:3] == line[0] * 3 and line[0] in ["-", "="]:
                    print("\n.. toctree::\n   :maxdepth: 1", file=dstfile)
