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


# mathjax3_config must already be defined
latex_macros = []
for k, v in mathjax3_config["tex"]["macros"].items():
    if len(v) == 1:
        latex_macros.append(r"\newcommand{\%s}{%s}" % (k, v[0]))
    else:
        latex_macros.append(r"\newcommand{\%s}[1]{%s}" % (k, v[0]))

imgmath_latex_preamble = "\n".join(latex_macros)

latex_elements = {"preamble": "\n".join(latex_macros)}
