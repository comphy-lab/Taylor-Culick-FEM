# Validation of the axisymmetric solvers

Verification and validation of four solvers for the opening of a circular hole
in a viscous sheet: two depth-averaged (Trouton/long-wave) formulations and two
direct free-surface Navier–Stokes formulations. The physics, the derivations,
the measured numbers and the numerical traps are in `axi-validation.tex`; this
file records what exists and how to run it.

| file | role |
|---|---|
| `axi-validation.tex` | source |
| `axi-validation.bib` | bibliography |
| `figures/` | figures included by the note, and the script that makes them |
| `Makefile`, `.latexmkrc` | build |

Citations are numbered in order of first appearance (`unsrt`). Uncited entries
remain in the `.bib` and do not appear in the reference list.

## The four solvers

All four live in `simulationCases/`.

| module | formulation |
|---|---|
| `trouton1d_axi.py` | depth-averaged, per-radian Lagrangian mass coordinate, radial metric `w = r²/2` carried as an unknown |
| `trouton1d_ale_axi.py` | depth-averaged, physical radius on a moving mesh |
| `axisymmetric_lab.py` | free-surface Navier–Stokes, laboratory frame, graded annulus |
| `axisymmetric_comoving.py` | free-surface Navier–Stokes, window of fixed length translating with the rim |

The design authority for all four — derivations, the reconciled scheme, the
internal-check list, the comparison protocol and the two source-level audits —
is `docs/axi-design.md`. The translating-window DNS is explained, with
schematics, in [`../axi-comoving-window.html`](../axi-comoving-window.html).

## Running the checks

Each suite runs the solver through the same driver the campaign uses, so it
exercises the production path rather than a bespoke harness. In every suite the
**planar limit is the gate**: it is the only check that compares against an
independently validated solver, and until it passes nothing else in the suite
is evidence of correctness.

```bash
# depth-averaged, mass coordinate: imports, metric, mass, planar-limit,
# tipthickness, exponential, dissipation
uv run python validation/trouton1d_axi_checks.py all
uv run python validation/trouton1d_axi_checks.py planar-limit

# depth-averaged, ALE: import, mass, planar-limit, tipthickness,
# affinity-roughness, exponential, cross-solver
uv run python validation/trouton1d_ale_axi_checks.py all
uv run python validation/trouton1d_ale_axi_checks.py planar-limit

# direct solver smoke ladder: S0 hoop-curvature discriminator, S1 FD Jacobian,
# S2 volume conservation, S4 physics probe
uv run python validation/axi_smoketests.py all
```

The smoke ladder deliberately omits S3, the planar reduction at `R0 = 100`: it
is a multi-hour comparison run, not a smoke test. It is reported in §5.1 of the
note instead.

Production runs go through the sweep and validation drivers:

```bash
uv run python validation/trouton1d_axi_sweep.py --oh ... --r0 ... --out ...
uv run python validation/trouton1d_ale_axi_sweep.py --oh ... --r0 ... --out ...
uv run python validation/run_validation_axi.py --oh ... --results ...
uv run python validation/run_validation_axi_comoving.py --oh ... --results ...
```

## Figures

`figures/make_axi_validation_figs.py` builds all three figures. It reads a small
cached copy of each plotted series from `figures/data/`, so the figures rebuild
from this repository alone:

```bash
python3 figures/make_axi_validation_figs.py
```

Simulation output does not live in this repository. To regenerate the cache
from a run tree, pass `--refresh` together with the source paths; the script's
docstring lists them.

## Build

From this directory:

```bash
make
```

or, if `latexmk` is installed,

```bash
latexmk -pdf axi-validation.tex
```

The output is `axi-validation.pdf`. `make clean` removes the auxiliary files
and leaves the PDF.

## Open slots

None in the note itself. The three measurements it was written around — the
fresh Newtonian axisymmetric rerun of the volume-of-fluid reference, the
comparison against digitised axisymmetric Savva & Bush curves, and the long
co-moving run — have all been made, so the `\todo{...}` markers and the macro
that rendered them in red are gone.

Two measurements the note names as wanted but does not contain:

- the **like-for-like volume-of-fluid rerun at `Oh_SB = 0.7071`**, which is what
  would settle whether the 1.6%/1.8% disagreement with the 2024 campaign belongs
  to that campaign or to a method (§5.2 says so in print);
- the **window-length ladder `L = 20/40/80`** for the co-moving solver, whose
  `L = 40` is currently asserted from planar experience rather than measured.

If either is added, reinstate a `\todo` macro rather than leaving a silent gap.
