import os

on_rtd = os.environ.get("READTHEDOCS") == "True"


if on_rtd:
    print("Building on ReadTheDocs\n")
    print("  current working directory: {}".format(os.path.abspath(os.curdir)))
    print("  rootpath: %s" % rootpath)
    print("  confpath: %s" % confpath)

    import matplotlib

    matplotlib.use("agg")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if on_rtd:
    html_static_path = []
else:
    html_static_path = ["_static"]
