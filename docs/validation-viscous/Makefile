.PHONY: pdf clean

pdf: validation-viscous.pdf

validation-viscous.pdf: validation-viscous.tex validation-viscous.bib
	pdflatex -interaction=nonstopmode validation-viscous
	bibtex validation-viscous
	pdflatex -interaction=nonstopmode validation-viscous
	pdflatex -interaction=nonstopmode validation-viscous

clean:
	rm -f validation-viscous.aux validation-viscous.bbl validation-viscous.blg \
	      validation-viscous.log validation-viscous.out validation-viscous.toc
