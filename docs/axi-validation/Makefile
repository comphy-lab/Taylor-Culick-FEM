.PHONY: pdf clean

pdf: axi-validation.pdf

axi-validation.pdf: axi-validation.tex axi-validation.bib
	pdflatex -interaction=nonstopmode axi-validation
	bibtex axi-validation
	pdflatex -interaction=nonstopmode axi-validation
	pdflatex -interaction=nonstopmode axi-validation

clean:
	rm -f axi-validation.aux axi-validation.bbl axi-validation.blg \
	      axi-validation.log axi-validation.out axi-validation.toc
