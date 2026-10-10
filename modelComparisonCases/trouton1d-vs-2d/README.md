# Two-dimensional sheet against the one-dimensional long-wave model

Comparison of the planar two-dimensional co-moving FEM with the one-dimensional
long-wave (Trouton) model at matched `Oh`. The long-wave model removes `Oh` by rescaling
time on the visco-inertial scale; that rescaling holds only while the thin-film
description does, so the comparison asks where in time and in `Oh` the two agree.

| File | Role |
|---|---|
| `compare_1d_2d.py` | Compares a one-dimensional sweep (`--oned`) with a two-dimensional sweep (`--twod`) case by case. |
| `oh_window.py` | Finds, for each `Oh`, the stretch of a two-dimensional history that follows the long-wave asymptotics. |

```bash
uv run python modelComparisonCases/trouton1d-vs-2d/compare_1d_2d.py --oned <1D run root> --twod <2D run root> --out <folder>
uv run python modelComparisonCases/trouton1d-vs-2d/oh_window.py --sweep <2D run root>
```
