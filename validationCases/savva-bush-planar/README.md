# Planar sheet against Savva & Bush (2009)

Code-to-code comparison of the planar co-moving FEM with the numerical solutions of
Savva & Bush, *J. Fluid Mech.* **626**, 211 (2009), figures 5 and 6: the edge speed
`u*(t*)` against the viscous time `t* = t / tau_vis` at several `Oh`. Their curves come
from a lubrication (slender-sheet) model, so the comparison also tests where the
two-dimensional and the reduced descriptions part.

| File | Role |
|---|---|
| `run_validation.py` | Runs one case (`physics.problem` = `planar_comoving`, `physics.Oh`) into a new folder; numerical options are recorded in its run manifest. |
| `cases/` | One case per `Oh` of the sweep. |
| `run_planar_sweep.sh` | The sweep: every case at the study's numerics, one new folder each under the given output root. |
| `compare_savva_bush.py` | Compares runs with the digitised curves on the reference's own samples; reads only runs whose manifest names a clean commit (`--allow-legacy` also reads runs that predate manifests). |
| `plot_campaign.py` | Summary figure: window convergence, the comparison, and the approach to `V_TC`. |
| `digitize_savva_bush.py` | Extraction record: regenerates `data/` from figures 5 and 6 of the paper, given with `--pdf` and checked against the SHA-256 of the extraction copy. |
| `data/` | Digitised curves (headers carry provenance and uncertainty), extraction overlays, and the independent one-dimensional Trouton curves used in the comparison. |
| `launch-records/` | The launch scripts of the planar `Oh` sweep toward `V_TC`. |

The conventions matter: `Oh = mu / sqrt(2 h0 rho sigma)` carries a `sqrt(2)` the coded
viscosity does not, and `tau_vis = mu h0 / (2 sigma)` is the viscous clock, on which the
published curves collapse only for `Oh` of order one and above. The co-moving window
length must exceed both the steady influence length and the diffusive spread
`sqrt(4 nu t)`; `window_length` in `src/taylor_culick_fem/savva_bush.py` sizes it.

```bash
validationCases/savva-bush-planar/run_planar_sweep.sh <output root>
uv run python validationCases/savva-bush-planar/run_validation.py \
    validationCases/savva-bush-planar/cases/planar-oh1.json --out <new folder> [--tstar-end 30 ...]
uv run python validationCases/savva-bush-planar/run_validation.py <case> --out <new folder> \
    --restart-from <run>/checkpoints/ckpt_NNNN.dump --tstar-end <later end>
uv run python validationCases/savva-bush-planar/compare_savva_bush.py --results <output root> --out <figure>
```

Each run writes `run-manifest.json` before any other output and refuses an output folder
that already holds anything; a continuation writes to its own new folder.
