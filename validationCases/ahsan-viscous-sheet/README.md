# Viscous sheet against Ahsan et al. (2025)

Code-to-code comparison, in the one-dimensional long-wave (Trouton) limit, of this
repository's finite-element solver with the finite-difference solvers released by
Ahsan, Brandão, Davidovitch & Stone, *Retraction of a viscous liquid sheet*,
arXiv:2507.04603 (2025), and with the asymptotic results of that paper.

| Code | Origin | Entry point |
|---|---|---|
| TC1D-FEM | this repository: pyoomph FEM in a Lagrangian mass coordinate | `src/taylor_culick_fem/trouton1d_pyoomph.py` |
| ABDS-CN | a numpy port of the authors' Crank–Nicolson MATLAB: `reduced-inf` (tip frame, infinite sheet) and `reduced-fin` (static frame, finite sheet) | `src/taylor_culick_fem/ahsan_thinfilm.py` |
| TC1D-MASS | a third, independently formulated solver of the semi-infinite problem | `src/taylor_culick_fem/tc1d_mass.py` |

| File | Role |
|---|---|
| `ahsan_campaign.py` | Runs one case of either code (`tcfem`, `abds`), the unit checks (`check-units`) and the figure-5 sweep (`fig5`). |
| `ahsan_campaign_jobs.py` | Prints the campaign's job list. |
| `ahsan_campaign_report.py` | Reduces a campaign run root into the comparison tables. |
| `ahsan_campaign_figs.py` | Draws the comparison figures from the reductions. |
| `ahsan_campaign_manifest.py` | Writes the campaign manifest describing a run root. |
| `um2_window.py` | Fits the late-time deficit coefficient over moving windows of a `u*(T)` record. |
| `digitize_ahsan_fig5.py` | Extraction record for `data/ahsan_fig5*.csv`; takes the paper with `--pdf` and checks it against the SHA-256 of the extraction copy. |
| `reference/` | Recorded `u*(T)` of the three codes on the semi-infinite problem. |

The paper scales time on the visco-inertial scale and thickness on the half-thickness;
`ahsan_thinfilm.py` documents the mapping to this repository's units, and
`ahsan_campaign.py check-units` verifies it.

```bash
uv run python validationCases/ahsan-viscous-sheet/ahsan_campaign.py check-units
uv run python validationCases/ahsan-viscous-sheet/ahsan_campaign.py tcfem --help
uv run python validationCases/ahsan-viscous-sheet/ahsan_campaign_report.py <run root>
```
