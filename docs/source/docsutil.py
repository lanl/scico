# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.


"""Utilities for building docs."""

import importlib
import inspect
import os
import pkgutil
import sys
from glob import glob
from runpy import run_path


def run_conf_files(vardict=None, path=None):
    """Execute Python files in conf directory.

    Args:
        vardict: Dictionary into which variable names should be inserted.
            Defaults to empty dict.
        path: Path to conf directory. Defaults to path to this module.

    Returns:
        A dict populated with variables defined during execution of the
        configuration files.
    """
    if vardict is None:
        vardict = {}
    if path is None:
        path = os.path.dirname(__file__)

    files = os.path.join(path, "conf", "*.py")
    for f in sorted(glob(files)):
        conf = run_path(f, init_globals=vardict)
        for k, v in conf.items():
            if len(k) >= 4 and k[0:2] == "__" and k[-2:] == "__":  # ignore __<name>__ variables
                continue
            vardict[k] = v
    return vardict


def package_classes(package):
    """Get a list of classes in a package.

    Return a list of qualified names of classes in the specified
    package. Classes in modules with names beginning with an "_" are
    omitted, as are classes whose internal module name record is not
    the same as the module in which they are found (i.e. indicating
    that they have been imported from elsewhere).

    Args:
        package: Reference to package for which classes are to be listed
          (not package name string).

    Returns:
        A list of qualified names of classes in the specified package.
    """

    classes = []
    # Iterate over modules in package
    for importer, modname, _ in pkgutil.walk_packages(
        path=package.__path__, prefix=(package.__name__ + "."), onerror=lambda x: None
    ):
        # Skip modules whose names begin with a "_"
        if modname.split(".")[-1][0] == "_":
            continue
        importlib.import_module(modname)
        # Iterate over module members
        for name, obj in inspect.getmembers(sys.modules[modname]):
            if inspect.isclass(obj):
                # Get internal module name of class for comparison with working module name
                try:
                    objmodname = getattr(sys.modules[modname], obj.__name__).__module__
                except Exception:
                    objmodname = None
                if objmodname == modname:
                    classes.append(modname + "." + obj.__name__)

    return classes


def get_text_indentation(text, skiplines=0):
    """Compute the leading whitespace indentation in a block of text.

    Args:
        text: A block of text as a string.

    Returns:
        Indentation length.
    """
    min_indent = len(text)
    lines = text.splitlines()
    if len(lines) > skiplines:
        lines = lines[skiplines:]
    else:
        return None
    for line in lines:
        if len(line) > 0:
            indent = len(line) - len(line.lstrip())
            if indent < min_indent:
                min_indent = indent
    return min_indent


def add_text_indentation(text, indent):
    """Insert leading whitespace into a block of text.

    Args:
        text: A block of text as a string.
        indent: Number of leading spaces to insert on each line.

    Returns:
        Text with additional indentation.
    """
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if len(line) > 0:
            lines[n] = (" " * indent) + line
    return "\n".join(lines)


def insert_inheritance_diagram(clsqname, parts=None, default_nparts=2):
    """Insert an inheritance diagram into a class docstring.

    No action is taken for classes without a base clase, and for classes
    without a docstring.

    Args:
        clsqname: Qualified name (i.e. including module name path) of class.
        parts: A dict mapping qualified class names to custom values for
          the ":parts:" directive.
        default_nparts: Default value for the ":parts:" directive.
    """

    # Extract module name and class name from qualified class name
    clspth = clsqname.split(".")
    modname = ".".join(clspth[0:-1])
    clsname = clspth[-1]
    # Get reference to class
    cls = getattr(sys.modules[modname], clsname)
    # Return immediately if class has no base classes
    if getattr(cls, "__bases__") == (object,):
        return
    # Get current docstring
    docstr = getattr(cls, "__doc__")
    # Return immediately if class has no docstring
    if docstr is None:
        return
    # Use class-specific parts or default parts directive value
    if parts and clsqname in parts:
        nparts = parts[clsqname]
    else:
        nparts = default_nparts
    # Split docstring into individual lines
    lines = docstr.splitlines()
    # Return immediately if there are no lines
    if not lines:
        return
    # Cut leading whitespace lines
    n = 0
    for n, line in enumerate(lines):
        if line != "":
            break
    lines = lines[n:]
    # Define inheritance diagram insertion text
    idstr = f"""

    .. inheritance-diagram:: {clsname}
       :parts: {nparts}


    """
    docstr_indent = get_text_indentation(docstr, skiplines=1)
    if docstr_indent is not None and docstr_indent > 4:
        idstr = add_text_indentation(idstr, docstr_indent - 4)
    # Insert inheritance diagram after summary line and whitespace line following it
    lines.insert(2, idstr)
    # Construct new docstring and attach it to the class
    extdocstr = "\n".join(lines)
    setattr(cls, "__doc__", extdocstr)
