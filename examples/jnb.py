# -*- coding: utf-8 -*-
# Copyright (C) 2022 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Support functions for manipulating Jupyter notebooks."""


import re
from timeit import default_timer as timer

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from py2jn.tools import py_string_to_notebook, write_notebook


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
                # statement extended over multiple lines, nor an os.environ statement, nor
                # components of a try/except construction (note that handling of these final
                # two cases is probably not very robust).
                if not re.match(
                    r"(^import|^from|^\n$|^\W+[^\W]|^\)$|^os.environ|^try:$|^except)", line
                ):
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
            if re.match("^input\(", line):  # end processing when input statement encountered
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


def read_notebook(fname):
    """Read a notebook from the specified notebook file."""

    try:
        nb = nbformat.read(fname, as_version=4)
    except (AttributeError, nbformat.reader.NotJSONError):
        raise RuntimeError("Error reading notebook file %s." % fname)
    return nb


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


def notebook_executed(nbfn):
    """Determine whether the notebook at `nbfn` has been executed."""

    try:
        nb = nbformat.read(nbfn, as_version=4)
    except (AttributeError, nbformat.reader.NotJSONError):
        raise RuntimeError("Error reading notebook file %s." % pth)
    cells = nb["worksheets"][0]["cells"]
    for n in range(len(nb["cells"])):
        if cells[n].cell_type == "code" and cells[n].execution_count is None:
            return False
    return True


def same_notebook_code(nb1, nb2):
    """Return ``True`` if the code cells of notebook objects `nb1` and `nb2`
    are all the same.
    """

    if "cells" in nb1:
        nb1c = nb1["cells"]
    else:
        nb1c = nb1["worksheets"][0]["cells"]
    if "cells" in nb2:
        nb2c = nb2["cells"]
    else:
        nb2c = nb2["worksheets"][0]["cells"]

    # Notebooks do not match if the number of cells differ
    if len(nb1c) != len(nb2c):
        return False

    # Iterate over cells in nb1
    for n in range(len(nb1c)):
        # Notebooks do not match if corresponding cells have different
        # types
        if nb1c[n]["cell_type"] != nb2c[n]["cell_type"]:
            return False
        # Notebooks do not match if source of corresponding code cells
        # differ
        if nb1c[n]["cell_type"] == "code" and nb1c[n]["source"] != nb2c[n]["source"]:
            return False

    return True


def same_notebook_markdown(nb1, nb2):
    """Return ``True`` if the markdown cells of notebook objects `nb1`
    and `nb2` are all the same.
    """

    if "cells" in nb1:
        nb1c = nb1["cells"]
    else:
        nb1c = nb1["worksheets"][0]["cells"]
    if "cells" in nb2:
        nb2c = nb2["cells"]
    else:
        nb2c = nb2["worksheets"][0]["cells"]

    # Notebooks do not match if the number of cells differ
    if len(nb1c) != len(nb2c):
        return False

    # Iterate over cells in nb1
    for n in range(len(nb1c)):
        # Notebooks do not match if corresponding cells have different
        # types
        if nb1c[n]["cell_type"] != nb2c[n]["cell_type"]:
            return False
        # Notebooks do not match if source of corresponding code cells
        # differ
        if nb1c[n]["cell_type"] == "markdown" and nb1c[n]["source"] != nb2c[n]["source"]:
            return False

    return True


def replace_markdown_cells(src, dst):
    """Overwrite markdown cells in notebook object `dst` with corresponding
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

    # It is an error to attempt markdown replacement if src and dst
    # have different numbers of cells
    if len(srccell) != len(dstcell):
        raise ValueError("Notebooks do not have the same number of cells.")

    # Iterate over cells in src
    for n in range(len(srccell)):
        # It is an error to attempt markdown replacement if any
        # corresponding pair of cells have different type
        if srccell[n]["cell_type"] != dstcell[n]["cell_type"]:
            raise ValueError("Cell number %d of different type in src and dst.")
        # If current src cell is a markdown cell, copy the src cell to
        # the dst cell
        if srccell[n]["cell_type"] == "markdown":
            dstcell[n]["source"] = srccell[n]["source"]
