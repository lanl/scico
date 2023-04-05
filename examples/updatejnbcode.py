#!/usr/bin/env python

# Update code cells in notebooks from corresponding scripts without
# the need to re-execute the notebook. NB: use with caution!
# Run as
#     python updatejnbcode.py <script-name.py>

import os
import sys

from jnb import py_file_to_string, read_notebook
from py2jn.tools import py_string_to_notebook, write_notebook


def replace_code_cells(src, dst):
    """Overwrite code cells in notebook object `dst` with corresponding
    cells in notebook object `src`.
    """

    if "cells" in src:
        srccell = src["cells"]
    else:
        srccell = src["worksheets"][0]["cells"]
    if "cells" in dst:
        dstcell = dst["cells"]
    else:
        dstcell = dst["worksheets"][0]["cells"]

    # It is an error to attempt replacement if src and dst have different
    # numbers of cells
    if len(srccell) != len(dstcell):
        raise ValueError("Notebooks do not have the same number of cells.")

    # Iterate over cells in src
    for n in range(len(srccell)):
        # It is an error to attempt replacement if any corresponding pair
        # of cells have different type
        if srccell[n]["cell_type"] != dstcell[n]["cell_type"]:
            raise ValueError("Cell number %d of different type in src and dst.")
        # If current src cell is a code cell, copy the src cell to the dst cell
        if srccell[n]["cell_type"] == "code":
            dstcell[n]["source"] = srccell[n]["source"]


src = sys.argv[1]
dst = os.path.join("notebooks", os.path.splitext(os.path.basename(src))[0] + ".ipynb")
print(f"Updating code cells in {dst} from {src}")
if os.path.exists(dst):
    srcnb = py_string_to_notebook(py_file_to_string(src), nbver=4)
    dstnb = read_notebook(dst)
    replace_code_cells(srcnb, dstnb)
    write_notebook(dstnb, dst)
