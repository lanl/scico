# Makefile for Sphinx documentation


# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS	?=
SPHINXBUILD	?= sphinx-build
SOURCEDIR	= source
BUILDDIR	= ../build/sphinx

.PHONY: help clean Makefile


# Put this first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -f $(SOURCEDIR)/_autosummary/*
	rm -f $(SOURCEDIR)/examples/*

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir -p source/examples; \
	for f in ../data/notebooks/*; do \
	  b=$$(basename $$f) ; \
	  if [ ! -f "source/examples/$$b" ]; then \
	      echo Creating soft link for notebook $$b ; \
	      ln -s -t source/examples "../../$$f" ; \
	  fi \
	done
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
