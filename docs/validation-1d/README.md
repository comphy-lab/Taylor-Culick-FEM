# Validation of the 1D viscous-sheet solver

A finite-element solver for the one-dimensional (long-wave) retraction of a
viscous liquid sheet, checked against an independent finite-difference
implementation and against the asymptotics of Ahsan, Brandão, Davidovitch and
Stone (arXiv:2507.04603).

| file | role |
|---|---|
| `validation-1d.tex` | source |
| `validation-1d.bib` | bibliography |
| `figures/` | figures included by the note |
| `.latexmkrc` | `latexmk` settings (PDF mode, BibTeX) |

Citations are numbered in order of first appearance (`unsrt`). Uncited entries
remain in the `.bib` and do not appear in the reference list.

## Build

From this directory:

```bash
make
```

or, if `latexmk` is installed,

```bash
latexmk -pdf validation-1d.tex
```

or the explicit cycle

```bash
pdflatex validation-1d
bibtex validation-1d
pdflatex validation-1d
pdflatex validation-1d
```

The output is `validation-1d.pdf`. `make clean` removes the auxiliary files
(`.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc`) and leaves the PDF.
