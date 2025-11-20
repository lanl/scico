import os

on_rtd = os.environ.get("READTHEDOCS") == "True"


if on_rtd:
    print("Building on ReadTheDocs\n")
    print("  current working directory: {}".format(os.path.abspath(os.curdir)))
    print("  rootpath: %s" % rootpath)
    print("  confpath: %s" % confpath)

    html_static_path = []

    # See https://about.readthedocs.com/blog/2024/07/addons-by-default/#how-to-opt-in-to-addons-now
    html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True

    import matplotlib

    matplotlib.use("agg")

else:
    # Add any paths that contain custom static files (such as style sheets) here,
    # relative to this directory. They are copied after the builtin static files,
    # so a file named "default.css" will overwrite the builtin "default.css".
    html_static_path = ["_static"]
