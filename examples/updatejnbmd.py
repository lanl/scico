#!/usr/bin/env python

# Update markdown cells in notebooks from corresponding scripts without
# the need to re-execute the notebook. Only applicable if the changes to
# the script since generation of the corresponding notebook only affect
# markdown cells.
# Run as
#     python updatejnbmd.py

import glob
import os

from jnb import (
    py_file_to_string,
    read_notebook,
    replace_markdown_cells,
    same_notebook_code,
    same_notebook_markdown,
)
from py2jn.tools import py_string_to_notebook, write_notebook

for src in glob.glob(os.path.join("scripts", "*.py")):
    dst = os.path.join("notebooks", os.path.splitext(os.path.basename(src))[0] + ".ipynb")
    if os.path.exists(dst):
        srcnb = py_string_to_notebook(py_file_to_string(src), nbver=4)
        dstnb = read_notebook(dst)
        if not same_notebook_code(srcnb, dstnb):
            print(f"Non-markup changes in {src}")
            continue
        if not same_notebook_markdown(srcnb, dstnb):
            print(f"Updating markdown in {dst}")
            replace_markdown_cells(srcnb, dstnb)
            write_notebook(dstnb, dst)
