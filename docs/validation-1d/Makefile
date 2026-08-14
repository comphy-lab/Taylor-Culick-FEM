.PHONY: pdf clean

pdf: validation-1d.pdf

validation-1d.pdf: validation-1d.tex validation-1d.bib
	pdflatex -interaction=nonstopmode validation-1d
	bibtex validation-1d
	pdflatex -interaction=nonstopmode validation-1d
	pdflatex -interaction=nonstopmode validation-1d

clean:
	rm -f validation-1d.aux validation-1d.bbl validation-1d.blg \
	      validation-1d.log validation-1d.out validation-1d.toc
