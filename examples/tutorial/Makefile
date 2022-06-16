OUT=compiled_files

all: $(OUT)/tutorial_PnP.ipynb $(OUT)/tutorial_cells.ipynb $(OUT)/install.html

$(OUT)/tutorial_%.ipynb: tutorial_%.py
	rm -rf temp
	mkdir temp
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$1/gs' $< > temp/tutorial_$*_questions.py
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$2/gs' $< > temp/tutorial_$*_answers.py
	python -m py2jn temp/tutorial_$*_questions.py $@
	python -m py2jn temp/tutorial_$*_answers.py $(OUT)/tutorial_$*_solution.ipynb
	rm -rf temp

$(OUT)/install.html: install.md
	grip --export --title=" " install.md $(OUT)/install.html
