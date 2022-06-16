all: tutorial_cells.py
	rm -rf temp
	mkdir temp
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$1/gs' tutorial_cells.py > temp/tutorial_cells_student.py
	perl -0777 -pe 's/#\s*startq(.*?)#\s*starta(.*?)#\s*endqa/$$2/gs' tutorial_cells.py > temp/tutorial_cells_instructor.py
	python -m py2jn temp/tutorial_cells_instructor.py out.ipynb
