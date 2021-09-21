# Makefile for Jypyter notebooks
# To rebuild the notebooks without executing them, do
#    make [target] fast

SRCPTH = scripts
DSTPTH = notebooks
SRC = $(SRCPTH)/*.py
SRCBASE = $(shell echo $(SRC) | xargs -n 1 basename)
DSTBASE = $(SRCBASE:%.py=%.ipynb)
DST = $(shell echo $(DSTBASE) | xargs -n 1 printf "notebooks/%s ")

.PHONY: all fast
.SUFFIXES: .py .ipynb

all: $(DST)
fast: $(DST)

$(DSTPTH)/%.ipynb: $(SRCPTH)/%.py
ifneq (,$(findstring fast,$(MAKECMDGOALS)))  # source: https://www.gnu.org/software/make/manual/html_node/Testing-Flags.html and https://www.gnu.org/software/make/manual/html_node/Goals.html
	./pytojnb $< $@
else
	./pytojnb $< $@
	jupyter nbconvert --to=notebook --output $(shell basename $@) --execute $@
endif
