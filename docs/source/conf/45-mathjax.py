import os

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
