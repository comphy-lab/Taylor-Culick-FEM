$pdf_mode = 1;
# 0, not 2: the .bbl is tracked so the note builds without BibTeX, and
# $bibtex_use = 2 makes `latexmk -c` delete it unconditionally.
$bibtex_use = 0;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
