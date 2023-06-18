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


# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports
autodoc_mock_imports = ["astra", "svmbir", "ray"]


# See
#  https://stackoverflow.com/questions/2701998#62613202
#  https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
autosummary_generate = True


# See https://stackoverflow.com/questions/5599254
autoclass_content = "both"
