.PHONY: pdf clean

pdf: planar-campaign.pdf

FIGURES := $(wildcard figures/*.pdf figures/*.png)

planar-campaign.pdf: planar-campaign.tex planar-campaign.bib $(FIGURES)
	pdflatex -interaction=nonstopmode planar-campaign
	bibtex planar-campaign
	pdflatex -interaction=nonstopmode planar-campaign
	pdflatex -interaction=nonstopmode planar-campaign

clean:
	# .bbl and .blg are tracked so the note builds without BibTeX.
	rm -f planar-campaign.aux planar-campaign.log planar-campaign.out \
	      planar-campaign.toc
