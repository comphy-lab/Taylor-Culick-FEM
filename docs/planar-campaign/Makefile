.PHONY: pdf clean

pdf: planar-campaign.pdf

FIGURES := $(wildcard figures/*.pdf)

planar-campaign.pdf: planar-campaign.tex planar-campaign.bib $(FIGURES)
	pdflatex -interaction=nonstopmode planar-campaign
	bibtex planar-campaign
	pdflatex -interaction=nonstopmode planar-campaign
	pdflatex -interaction=nonstopmode planar-campaign

clean:
	rm -f planar-campaign.aux planar-campaign.bbl planar-campaign.blg \
	      planar-campaign.log planar-campaign.out planar-campaign.toc
