# -*- coding: utf-8 -*-

import os
import re
import sys

confpath = os.path.dirname(__file__)
sys.path.append(confpath)
rootpath = os.path.realpath(os.path.join(confpath, "..", ".."))
sys.path.append(rootpath)

from docsutil import (
    insert_inheritance_diagram,
    insert_type_params,
    package_classes,
    run_conf_files,
)

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

# Some of the docstrings of wrapped jax functions include footnotes that
# are not referenced within the docstring. These are converted to
# citations (see process_docstring below) since, unlike footnotes, these
# can be unreferenced without a warning being raised by docutils, but
# sphinx itself raises a warning for unreferenced citations, which is
# suppressed here.
suppress_warnings = ["ref.citation"]


def class_inherit_diagrams(_):
    # Insert inheritance diagrams for classes that have base classes
    import scico

    custom_parts = {"scico.ray.tune.Tuner": 4}
    clslst = package_classes(scico)
    for cls in clslst:
        insert_inheritance_diagram(cls, parts=custom_parts)


def jax_type_params(_):
    # Make PEP 695 type parameters of jax classes included in the scico
    # namespace resolvable by typing.get_type_hints
    import jax

    insert_type_params(jax.custom_jvp, jax.custom_vjp)


def process_docstring(app, what, name, obj, options, lines):
    # Some jax docstrings (e.g. that of jax.numpy.promote_types, which is
    # included in the scico.numpy namespace) use ":type:" as if it were a
    # sphinx role, resulting in errors of the form
    #
    #   ERROR: Unknown interpreted text role "type".
    #
    # Since the intent is clearly a cross-reference to a type, these are
    # replaced with the ":class:" role.
    for n, line in enumerate(lines):
        if ":type:`" in line:
            lines[n] = line.replace(":type:`", ":class:`")

    # Manually numbered footnotes in docstrings are a problem when two
    # docstrings using the same footnote label (e.g. those of
    # jax.random.categorical and jax.random.orthogonal, both of which are
    # included in the scico.random namespace) appear on the same page,
    # resulting in warnings of the form
    #
    #   WARNING: Duplicate explicit target name: "1".
    #
    # Footnotes that are defined but never referenced (which is also the
    # case for some of the wrapped jax docstrings) are a problem too,
    # resulting in warnings of the form
    #
    #   WARNING: Footnote [1] is not referenced.
    #
    # Both are avoided by replacing manually numbered footnotes with
    # labels that are unique to the object being documented: referenced
    # footnotes are converted into auto-numbered footnotes (so that they
    # are still displayed as numbers), and unreferenced footnotes into
    # citations (which, unlike footnotes, may be unreferenced). Note that
    # some of the affected docstrings refer to their footnotes in plain
    # text (e.g. "See [1] for reference.") rather than via footnote
    # references; these plain text references are converted too.
    defpattern = r"^\s*\.\.\s+\[(\d+)\]\s"
    refpattern = r"(?<![\w\]\)])\[(\d+)\]_?"
    defined, referenced = set(), set()
    for line in lines:
        labels = re.findall(defpattern, line)
        if labels:
            defined |= set(labels)
        else:
            referenced |= set(re.findall(refpattern, line))
    if defined:
        prefix = name.rsplit(".", 1)[-1]
        # Auto-numbered footnote labels for referenced footnotes, citation
        # labels for unreferenced ones
        altlabels = {
            label: f"[#{prefix}-{label}]" if label in referenced else f"[{prefix}-{label}]"
            for label in defined
        }
        altlabel = lambda m: altlabels[m.group(1)]  # noqa: E731
        labelpattern = "|".join(sorted(defined))
        for n, line in enumerate(lines):
            if re.match(defpattern.replace(r"\d+", labelpattern), line):
                lines[n] = re.sub(r"\[(%s)\]" % labelpattern, altlabel, line, count=1)
            else:
                lines[n] = re.sub(
                    refpattern.replace(r"\d+", labelpattern), lambda m: altlabel(m) + "_", line
                )

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
    app.connect("builder-inited", jax_type_params)
    app.connect("autodoc-process-docstring", process_docstring)
