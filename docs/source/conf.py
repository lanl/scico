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


def process_docstring(app, what, name, obj, options, lines):
    # Don't show docs for inherited members in classes in scico.flax.
    # This is primarily useful for silencing warnings due to problems in
    # the current release of flax, but is arguably also useful in avoiding
    # extensive documentation of methods that are likely to be of limited
    # interest to users of the scico.flax classes.
    #
    # Note: this event handler currently has no effect since inclusion of
    #   inherited members is currently globally disabled (see
    #   "inherited-members" in autodoc_default_options), but is left in
    #   place in case a decision is ever made to revert the global setting.
    #
    # See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    # for documentation of the autodoc-process-docstring event used here.
    if what == "class" and "scico.flax." in name:
        options["inherited-members"] = False


def setup(app):
    app.connect("builder-inited", class_inherit_diagrams)
    app.connect("autodoc-process-docstring", process_docstring)
