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
