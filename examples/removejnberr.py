#!/usr/bin/env python

# Remove output to stderr in notebooks. NB: use with caution!
# Run as
#     python removejnberr.py

import glob
import os

from jnb import read_notebook, remove_error_output
from py2jn.tools import write_notebook

for src in glob.glob(os.path.join("notebooks", "*.ipynb")):
    nb = read_notebook(src)
    modflg = remove_error_output(nb)
    if modflg:
        print(f"Removing output to stderr from {src}")
        write_notebook(nb, src)
