# Planar sheet against Savva & Bush (2009)

Code-to-code comparison of the planar co-moving FEM with the numerical solutions of
Savva & Bush, *J. Fluid Mech.* **626**, 211 (2009), figures 5 and 6: the edge speed
`u*(t*)` against the viscous time `t* = t / tau_vis` at several `Oh`. Their curves come
from a lubrication (slender-sheet) model, so the comparison also tests where the
two-dimensional and the reduced descriptions part.

| File | Role |
|---|---|
| `run_validation.py` | Runs the planar co-moving solver over a list of `Oh` into `--results`. |
| `compare_savva_bush.py` | Compares runs with the digitised curves on the reference's own samples. |
| `plot_campaign.py` | Summary figure: window convergence, the comparison, and the approach to `V_TC`. |
| `digitize_savva_bush.py` | Extraction record: regenerates `data/` from the paper's figures. |
| `data/` | Digitised curves (headers carry provenance and uncertainty), extraction overlays, and the independent one-dimensional Trouton curves used in the comparison. |
| `literature/` | The source papers. |
| `launch-records/` | The launch scripts of the planar `Oh` sweep toward `V_TC`. |

The conventions matter: `Oh = mu / sqrt(2 h0 rho sigma)` carries a `sqrt(2)` the coded
viscosity does not, and `tau_vis = mu h0 / (2 sigma)` is the viscous clock, on which the
published curves collapse only for `Oh` of order one and above. The co-moving window
length must exceed both the steady influence length and the diffusive spread
`sqrt(4 nu t)`; `window_length` in `src/taylor_culick_fem/savva_bush.py` sizes it.

```bash
uv run python validationCases/savva-bush-planar/run_validation.py --results <run root> --oh 0.5 1 5
uv run python validationCases/savva-bush-planar/compare_savva_bush.py --results <run root>
uv run python validationCases/savva-bush-planar/plot_campaign.py --results <run root> --out <figure>
```
