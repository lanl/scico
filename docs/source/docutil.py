# -*- coding: utf-8 -*-
# Copyright (C) 2021 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.


"""Utilities for building docs."""


import importlib
import inspect
import pkgutil
import sys


def package_classes(package):
    """Get a list of classes in a package.

    Return a list of qualified names of classes in the specified package. Classes in modules
    with names beginning with an "_" are omitted, as are classes whose internal module name
    record is not the same as the module in which they are found (i.e. indicating that they
    have been imported from elsewhere).

    Args:
        package: Reference to package for which classes are to be listed (not package name
          string)
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


def insert_inheritance_diagram(clsqname):
    """Insert an inheritance diagram into a class docstring.

    No action is taken for classes without a base clase, and for classes without a docstring.

    Args:
        clsqname: Qualified name (i.e. including module name path) of class
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
    idstr = (
        """

        .. inheritance-diagram:: %s
           :parts: 2


    """
        % clsname
    )
    # Insert inheritance diagram after summary line and whitespace line following it
    lines.insert(2, idstr)
    # Construct new docstring and attach it to the class
    extdocstr = "\n".join(lines)
    setattr(cls, "__doc__", extdocstr)
