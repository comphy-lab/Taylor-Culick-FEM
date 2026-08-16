.PHONY: pdf clean

pdf: axi-validation.pdf

axi-validation.pdf: axi-validation.tex axi-validation.bib
	pdflatex -interaction=nonstopmode axi-validation
	bibtex axi-validation
	pdflatex -interaction=nonstopmode axi-validation
	pdflatex -interaction=nonstopmode axi-validation

clean:
	# axi-validation.bbl and .blg are tracked -- the .bbl is what lets the
	# note build without BibTeX -- so clean must not remove them.
	rm -f axi-validation.aux axi-validation.log axi-validation.out \
	      axi-validation.toc
