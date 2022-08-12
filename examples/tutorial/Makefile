OUT=../notebooks/tutorial

objects = $(OUT)/tutorial_CNN.ipynb  $(OUT)/tutorial_CNN_solution.ipynb \
          $(OUT)/tutorial_PnP.ipynb  $(OUT)/tutorial_PnP_solution.ipynb \
          $(OUT)/tutorial_cells.ipynb $(OUT)/tutorial_cells_solution.ipynb \
          $(OUT)/tutorial_funcs.py $(OUT)/tutorial_funcs_part2.py
all: $(objects)

$(OUT)/tutorial_%.ipynb: tutorial_%.py
	rm -rf temp
	mkdir temp
	# parse startq, starta, startq
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$1/gs' $< > temp/tutorial_$*_questions.py
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$2/gs' $< > temp/tutorial_$*_answers.py
	# turn .py files into .ipynb notebooks
	python -m py2jn temp/tutorial_$*_questions.py $(OUT)/tutorial_$*.ipynb
	python -m py2jn temp/tutorial_$*_answers.py   $(OUT)/tutorial_$*_solution.ipynb
	rm -rf temp

$(OUT)/%.py: %.py
	cp $< $@

clean:
	rm $(objects)
