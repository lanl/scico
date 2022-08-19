paper.pdf: paper.md paper.bib
	docker run --rm \
	    --volume $(shell pwd)/:/data \
	    --user $(id -u):$(id -g) \
	    --env JOURNAL=joss \
	    openjournals/inara

clean:
	rm paper.pdf
