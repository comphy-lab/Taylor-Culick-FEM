.PHONY: pdf clean

pdf: validation-viscous.pdf

FIGURES := $(wildcard figures/*.pdf)

validation-viscous.pdf: validation-viscous.tex validation-viscous.bib $(FIGURES)
	pdflatex -interaction=nonstopmode validation-viscous
	bibtex validation-viscous
	pdflatex -interaction=nonstopmode validation-viscous
	pdflatex -interaction=nonstopmode validation-viscous

clean:
	# .bbl and .blg are tracked so the note builds without BibTeX.
	rm -f validation-viscous.aux validation-viscous.log validation-viscous.out \
	      validation-viscous.toc
