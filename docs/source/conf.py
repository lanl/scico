# -*- coding: utf-8 -*-

import os
import re
import sys
from inspect import getmembers, isfunction
from typing import Optional, Sequence, Union  # needed for typehints_formatter hack

from sphinx.ext.napoleon.docstring import GoogleDocstring

confpath = os.path.dirname(__file__)
sys.path.append(confpath)
rootpath = os.path.join(confpath, "..", "..")
sys.path.append(rootpath)

from docutil import insert_inheritance_diagram, package_classes

from scico._version import package_version
from scico.typing import (  # needed for typehints_formatter hack
    ArrayIndex,
    AxisIndex,
    DType,
)


## See
##   https://github.com/sphinx-doc/sphinx/issues/2115
##   https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
##
# first, we define new methods for any new sections and add them to the class
def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


confpath = os.path.dirname(__file__)
sys.path.append(confpath)

on_rtd = os.environ.get("READTHEDOCS") == "True"


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
rootpath = os.path.abspath("../..")
sys.path.insert(0, rootpath)

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "nbsphinx",
]

bibtex_bibfiles = ["references.bib"]

nbsphinx_execute = "never"
nbsphinx_prolog = """
.. raw:: html

    <style>
    .nbinput .prompt, .nboutput .prompt {
        display: none;
    }
    div.highlight {
        background-color: #f9f9f4;
    }
    p {
        margin-bottom: 0.8em;
        margin-top: 0.8em;
    }
    </style>
"""


# See
#  https://stackoverflow.com/questions/2701998#62613202
#  https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
autosummary_generate = True


if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
else:
    extensions.append("sphinx.ext.mathjax")
    # To use local copy of MathJax for offline use, set MATHJAX_URI to
    #    file:///[path-to-mathjax-repo-root]/es5/tex-mml-chtml.js
    if os.environ.get("MATHJAX_URI"):
        mathjax_path = os.environ.get("MATHJAX_URI")

mathjax3_config = {
    "tex": {
        "macros": {
            "mb": [r"\mathbf{#1}", 1],
            "mbs": [r"\boldsymbol{#1}", 1],
            "mbb": [r"\mathbb{#1}", 1],
            "norm": [r"\lVert #1 \rVert", 1],
            "abs": [r"\left| #1 \right|", 1],
            "argmin": [r"\mathop{\mathrm{argmin}}"],
            "sign": [r"\mathop{\mathrm{sign}}"],
            "prox": [r"\mathrm{prox}"],
            "det": [r"\mathrm{det}"],
            "exp": [r"\mathrm{exp}"],
            "loss": [r"\mathop{\mathrm{loss}}"],
            "kp": [r"k_{\|}"],
            "rp": [r"r_{\|}"],
        }
    }
}

latex_macros = []
for k, v in mathjax3_config["tex"]["macros"].items():
    if len(v) == 1:
        latex_macros.append(r"\newcommand{\%s}{%s}" % (k, v[0]))
    else:
        latex_macros.append(r"\newcommand{\%s}[1]{%s}" % (k, v[0]))

imgmath_latex_preamble = "\n".join(latex_macros)


# See https://stackoverflow.com/questions/5599254
autoclass_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "SCICO"
copyright = "2020-2023, SCICO Developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = package_version()
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "tmp",
    "*.tmp.*",
    "*.tmp",
    "examples",
    "include",
]


# napoleon_include_init_with_doc = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# See https://github.com/sphinx-doc/sphinx/issues/9119
# napoleon_custom_sections = [("Returns", "params_style")]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
# html_theme = "python_docs_theme"
html_theme = "furo"

html_theme_options = {
    "top_of_page_button": None,
    # "sidebar_hide_name": True,
}

if html_theme == "python_docs_theme":
    html_sidebars = {
        "**": ["globaltoc.html", "sourcelink.html", "searchbox.html"],
    }


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/scico.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if on_rtd:
    html_static_path = []
else:
    html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "SCICOdoc"

# Include TODOs
todo_include_todos = True


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "scico.tex", "SCICO Documentation", "The SCICO Developers", "manual"),
]

latex_engine = "xelatex"

# latex_use_xindy = False

latex_elements = {"preamble": "\n".join(latex_macros)}


# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "svmbir": ("https://svmbir.readthedocs.io/en/latest/", None),
}
# Added timeout due to periodic scipy.org down time
# intersphinx_timeout = 30


graphviz_output_format = "svg"
inheritance_graph_attrs = dict(rankdir="LR", fontsize=9, ratio="compress", bgcolor="transparent")
inheritance_edge_attrs = dict(
    color='"#2962ffff"',
)
inheritance_node_attrs = dict(
    shape="box",
    fontsize=9,
    height=0.4,
    margin='"0.08, 0.03"',
    style='"rounded,filled"',
    color='"#2962ffff"',
    fontcolor='"#2962ffff"',
    fillcolor='"#f0f0f8b0"',
)


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "scico", "SCICO Documentation", ["SCICO Developers"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "SCICO",
        "SCICO Documentation",
        "SCICO Developers",
        "SCICO",
        "Scientific Computational Imaging COde (SCICO)",
        "Miscellaneous",
    ),
]


if on_rtd:
    print("Building on ReadTheDocs\n")
    print("  current working directory: {}".format(os.path.abspath(os.curdir)))
    print("  rootpath: %s" % rootpath)
    print("  confpath: %s" % confpath)

    import numpy as np

    print("NumPy version: %s" % np.__version__)
    import matplotlib

    matplotlib.use("agg")


autodoc_default_options = {
    "member-order": "bysource",
    "inherited-members": False,
    "ignore-module-all": False,
    "show-inheritance": True,
    "members": True,
    "special-members": "__call__",
}
autodoc_docstring_signature = True
autoclass_content = "both"


# An explanation for this nasty hack, the primary purpose of which is to avoid
# the very long definition of the scico.typing.DType appearing explicitly in the
# docs. This is handled correctly by sphinx.ext.autodoc in some circumstances,
# but only when sphinx_autodoc_typehints is not included in the extension list,
# and the appearance of the type hints (e.g. whether links to definitions are
# included) seems to depend on whether "from __future__ import annotations" was
# used in the module being documented, which is not ideal from a consistency
# perspective. (It's also worth noting that sphinx.ext.autodoc provides some
# configurability for type aliases via the autodoc_type_aliases sphinx
# configuration option.) The alternative is to include sphinx_autodoc_typehints,
# which gives a consistent appearance to the type hints, but the
# autodoc_type_aliases configuration option is ignored, and type aliases are
# always expanded. This hack avoids expansion for the type aliases with the
# longest definitions by definining a custom function for formatting the
# type hints, using an option provided by sphinx_autodoc_typehints. For
# more information, see
#   https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
#   https://github.com/tox-dev/sphinx-autodoc-typehints/issues/284
#   https://github.com/tox-dev/sphinx-autodoc-typehints/blob/main/README.md
def typehints_formatter_function(annotation, config):
    markup = {
        DType: ":obj:`~scico.typing.DType`",
        # Compound types involving DType must be added here to avoid their DType
        # component being expanded in the docs.
        Optional[DType]: ":obj:`~typing.Optional`\ [\ :obj:`~scico.typing.DType`\ ]",
        Union[DType, Sequence[DType]]: (
            ":obj:`~typing.Union`\ [\ :obj:`~scico.typing.DType`\ , "
            ":obj:`~typing.Sequence`\ [\ :obj:`~scico.typing.DType`\ ]]"
        ),
        AxisIndex: ":obj:`~scico.typing.AxisIndex`",
        ArrayIndex: ":obj:`~scico.typing.ArrayIndex`",
    }
    if annotation in markup:
        return markup[annotation]
    else:
        return None


typehints_formatter = typehints_formatter_function


# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports
autodoc_mock_imports = ["astra", "svmbir", "ray"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**tests**", "**spi**", "**README.rst", "include"]


# Rewrite module names for certain functions imported into scico.numpy so that they are
# included in the docs for that module. While a bit messy to do so here rather than in a
# function run via app.connect, it is necessary (for some yet to be identified reason)
# to do it here to ensure that the relevant API docs include a table of functions.
import scico.numpy

for module in (scico.numpy, scico.numpy.fft, scico.numpy.linalg, scico.numpy.testing):
    for _, f in getmembers(module, isfunction):
        # Rewrite module name so that function is included in docs
        f.__module__ = module.__name__
        f.__doc__ = re.sub(
            r"^:func:`([\w_]+)` wrapped to operate",
            r":obj:`jax.numpy.\1` wrapped to operate",
            str(f.__doc__),
            flags=re.M,
        )
        modname = ".".join(module.__name__.split(".")[1:])
        f.__doc__ = re.sub(
            r"^LAX-backend implementation of :func:`([\w_]+)`.",
            r"LAX-backend implementation of :obj:`%s.\1`." % modname,
            str(f.__doc__),
            flags=re.M,
        )
        # Improve formatting of jax.numpy warning
        f.__doc__ = re.sub(
            r"^\*\*\* This function is not yet implemented by jax.numpy, and will "
            "raise NotImplementedError \*\*\*",
            "**WARNING**: This function is not yet implemented by jax.numpy, "
            " and will raise :exc:`NotImplementedError`.",
            f.__doc__,
            flags=re.M,
        )
        # Remove cross-references to section NEP35
        f.__doc__ = re.sub(":ref:`NEP 35 <NEP35>`", "NEP 35", f.__doc__, re.M)
        # Remove cross-reference to numpydoc style references section
        f.__doc__ = re.sub(r" \[(\d+)\]_", "", f.__doc__, flags=re.M)
        # Remove entire numpydoc references section
        f.__doc__ = re.sub(r"References\n----------\n.*\n", "", f.__doc__, flags=re.DOTALL)


# Remove spurious two-space indentation of entire docstring
scico.numpy.vectorize.__doc__ = re.sub("^  ", "", scico.numpy.vectorize.__doc__, flags=re.M)

# Fix various docstring formatting errors
scico.numpy.testing.break_cycles.__doc__ = re.sub(
    "calling gc.collect$",
    "calling gc.collect.\n\n",
    scico.numpy.testing.break_cycles.__doc__,
    flags=re.M,
)
scico.numpy.testing.break_cycles.__doc__ = re.sub(
    " __del__\) inside", "__del__\) inside", scico.numpy.testing.break_cycles.__doc__, flags=re.M
)
scico.numpy.testing.assert_raises_regex.__doc__ = re.sub(
    "\*args,\n.*\*\*kwargs",
    "*args, **kwargs",
    scico.numpy.testing.assert_raises_regex.__doc__,
    flags=re.M,
)
scico.numpy.BlockArray.global_shards.__doc__ = re.sub(
    "`Shard`s", "`Shard`\ s", scico.numpy.BlockArray.global_shards.__doc__, flags=re.M
)


# Similar processing for scico.scipy
import scico.scipy

ssp_func = getmembers(scico.scipy.special, isfunction)
for _, f in ssp_func:
    if f.__module__[0:11] == "scico.scipy" or f.__module__[0:14] == "jax._src.scipy":
        # Rewrite module name so that function is included in docs
        f.__module__ = "scico.scipy.special"
        # Attempt to fix incorrect cross-reference
        f.__doc__ = re.sub(
            r"^:func:`([\w_]+)` wrapped to operate",
            r":obj:`jax.scipy.special.\1` wrapped to operate",
            str(f.__doc__),
            flags=re.M,
        )
        modname = "scipy.special"
        f.__doc__ = re.sub(
            r"^LAX-backend implementation of :func:`([\w_]+)`.",
            r"LAX-backend implementation of :obj:`%s.\1`." % modname,
            str(f.__doc__),
            flags=re.M,
        )
        # Remove cross-reference to numpydoc style references section
        f.__doc__ = re.sub(r"(^|\ )\[(\d+)\]_", "", f.__doc__, flags=re.M)
        # Remove entire numpydoc references section
        f.__doc__ = re.sub(r"References\n----------\n.*\n", "", f.__doc__, flags=re.DOTALL)
        # Remove problematic citation
        f.__doc__ = re.sub("See \[dlmf\]_ for details.", "", f.__doc__, re.M)
        f.__doc__ = re.sub("\[dlmf\]_", "NIST DLMF", f.__doc__, re.M)

# Fix indentation problems
scico.scipy.special.sph_harm.__doc__ = re.sub(
    "^Computes the", "  Computes the", scico.scipy.special.sph_harm.__doc__, flags=re.M
)


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

    app.add_css_file("scico.css")
    app.add_css_file(
        "http://netdna.bootstrapcdn.com/font-awesome/4.7.0/" "css/font-awesome.min.css"
    )
    app.connect("builder-inited", class_inherit_diagrams)
    app.connect("autodoc-process-docstring", process_docstring)
