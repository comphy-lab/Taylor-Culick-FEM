# Validation of the viscous-limit (Trouton) planar sheet

A finite-element solver for the one-dimensional (long-wave) retraction of a
viscous liquid sheet, checked against an independent finite-difference
implementation and against the asymptotics of Ahsan, Brandão, Davidovitch and
Stone (arXiv:2507.04603). This is the viscous-limit reduced model, including
the solvers written here.

The two-dimensional inertial campaign is
[`../validation-inertial/`](../validation-inertial/). The axisymmetric
solver-construction note is [`../axi-validation/`](../axi-validation/). The
directory was previously `docs/validation-1d/`. The note is the same.

| file | role |
|---|---|
| `validation-viscous.tex` | source |
| `validation-viscous.bib` | bibliography |
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
latexmk -pdf validation-viscous.tex
```

or the explicit cycle

```bash
pdflatex validation-viscous
bibtex validation-viscous
pdflatex validation-viscous
pdflatex validation-viscous
```

The output is `validation-viscous.pdf`. `make clean` removes the auxiliary files
(`.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc`) and leaves the PDF.
