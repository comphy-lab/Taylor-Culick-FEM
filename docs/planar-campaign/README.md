# Newtonian Taylor–Culick retraction at arbitrary Ohnesorge number

A sharp-interface finite-element campaign measuring the retraction velocity of
a Newtonian liquid sheet across Ohnesorge number, and its approach to the
Taylor–Culick velocity. The physics, the conventions, the method and what the
Oh sweep does and does not establish are in `planar-campaign.tex`. This file
records what exists and how to build it.

The inertial planar FEM–Basilisk comparison (`Oh_SB < 0.1`) is
[`../validation-inertial/`](../validation-inertial/). The viscous Trouton/Ahsan
note is [`../validation-viscous/`](../validation-viscous/). The axisymmetric
solver-construction note is [`../axi-validation/`](../axi-validation/).

| file | role |
|---|---|
| `planar-campaign.tex` | source |
| `planar-campaign.bib` | bibliography |
| `figures/` | figures included by the note |
| `Makefile`, `.latexmkrc` | build |

Citations are numbered in order of first appearance (`unsrt`). Uncited entries
remain in the `.bib` and do not appear in the reference list.

## Build

From this directory:

```bash
make
```

or, if `latexmk` is installed,

```bash
latexmk -pdf planar-campaign.tex
```

The output is `planar-campaign.pdf`. `make clean` removes the auxiliary files
and leaves the PDF.

Figures are rebuilt from campaign output, when those volumes are mounted, from
the repository root:

```bash
uv run python postProcess/make_docs_figures.py --out docs/planar-campaign/figures
```

Simulation output does not live in this repository.
