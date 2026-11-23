# -*- coding: utf-8 -*-

import os
import sys

confpath = os.path.dirname(__file__)
sys.path.append(confpath)
rootpath = os.path.realpath(os.path.join(confpath, "..", ".."))
sys.path.append(rootpath)

from docsutil import insert_inheritance_diagram, package_classes, run_conf_files

# Process settings in files in conf directory
_vardict = run_conf_files(vardict={"confpath": confpath, "rootpath": rootpath})
for _k, _v in _vardict.items():
    globals()[_k] = _v
del _vardict, _k, _v


# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0.0"

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# Output file base name for HTML help builder.
htmlhelp_basename = "SCICOdoc"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**tests**", "**README.rst", "include"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Include TODOs
todo_include_todos = True


def class_inherit_diagrams(_):
    # Insert inheritance diagrams for classes that have base classes
    import scico

    custom_parts = {"scico.ray.tune.Tuner": 4}
    clslst = package_classes(scico)
    for cls in clslst:
        insert_inheritance_diagram(cls, parts=custom_parts)


def setup(app):
    app.connect("builder-inited", class_inherit_diagrams)

    # Required due to bug in sphinx-autodoc-typehints 3.17.6 (see tox-dev/sphinx-autodoc-typehints#1143)
    # Guard sphinx_autodoc_typehints against objects whose unwrapped callable has __qualname__ = None
    for listener in list(app.events.listeners.get("autodoc-process-docstring", [])):
        if listener.handler.__module__.startswith("sphinx_autodoc_typehints"):
            orig_handler = listener.handler
            priority = listener.priority
            app.disconnect(listener.id)

            def _guarded_handler(app, what, name, obj, options, lines, _orig=orig_handler):
                try:
                    return _orig(app, what, name, obj, options, lines)
                except AttributeError as exc:
                    if "'NoneType' object has no attribute 'split'" in str(exc):
                        return
                    raise

            app.connect("autodoc-process-docstring", _guarded_handler, priority=priority)
