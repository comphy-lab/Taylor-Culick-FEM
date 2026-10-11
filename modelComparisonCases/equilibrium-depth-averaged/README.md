# Arrested-hole equilibria against the static depth-averaged theory

With an infinite relaxation time an arrested hole is a static equilibrium of an
incompressible neo-Hookean sheet with surface tension. Two descriptions of it are
compared, both axisymmetric:

- the static limit of the depth-averaged theory, a one-dimensional radial problem for the
  column map (`lambda_r = dr/drho`, `lambda_theta = r/rho`, `lambda_z = 1/(lambda_r
  lambda_theta)`) in plane stress, with the bulk edge tension `-2 sigma` at a sharp edge
  and no resolved rim (`src/taylor_culick_fem/depth_averaged_equilibrium.py`);
- the steady axisymmetric equilibria of the full sheet, edge cap and free surface
  resolved, followed in `Ec` by
  [`run_elastocapillary_equilibrium.py`](../../simulationCases/run_elastocapillary_equilibrium.py).

The theory has no length scale besides the edge radius, so its hole radius scales
exactly with it. The comparison asks whether the ratio of the two hole radii tends to the
radius of the sharp-edged hole that holds the same material as the rounded cap,
`r0 + 1/2 - pi/8`, and at which distance from the edge the theory thickness equals the
resolved rim height.

| File | Role |
|---|---|
| `run_equilibrium_branches.sh` | Runs the two branches: the default case to `Ec = 0.41` and `cases/stiffened_cap.json`, whose edge-cap material is slightly stiffened so that the branch continues to `Ec = 0.39`. |
| `compare_equilibria.py` | Reads an equilibrium run's `branch.csv`; writes `comparison.csv` (theory radius, `D = ln(R_axi/R_theory)`, `d*` per point) and `fit.json` (`D(R) = D_inf - c R^-alpha`). |

```bash
modelComparisonCases/equilibrium-depth-averaged/run_equilibrium_branches.sh <output root>
uv run python modelComparisonCases/equilibrium-depth-averaged/compare_equilibria.py --branch <equilibrium run> --case <its case> --out <folder>
```

Below `Ec ~ 0.4` the whole reference edge cap is compressed into a thin nose at the tip of
the opened hole, and the incompressibility error there grows until Newton fails. The
stiffened cap (a Mooney--Rivlin term of relative weight `cap_beta = 1e-3` confined to the
cap material) delays this; where both branches exist, its effect on the hole radius and the
rim height is checked by comparing them.
