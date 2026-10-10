# One-dimensional Trouton solvers

Verification of the one-dimensional long-wave (Trouton) solvers in this repository
against each other and against their own convergence behaviour: the planar
mass-coordinate solver (`trouton1d_pyoomph.py`), its axisymmetric counterpart
(`trouton1d_axi.py`), the ALE formulations (`trouton1d_ale.py`, `trouton1d_ale_axi.py`)
and the independent finite-difference solver (`trouton1d.py`), all in
`src/taylor_culick_fem/`.

| File | Role |
|---|---|
| `trouton1d_sweep.py` | `u*(t)` at one `Oh`, with the domain and time step sized per case. |
| `trouton1d_convergence.py` | Mesh, time-step and tolerance convergence at one `Oh`. |
| `trouton1d_axi_sweep.py`, `trouton1d_axi_checks.py` | The axisymmetric solver and its exact internal checks (metric, mass identity, planar limit, tip closure, small-hole growth). |
| `trouton1d_ale_sweep.py`, `trouton1d_ale_axi_sweep.py` | The ALE formulations, in the same shape as the sweeps above. |
| `trouton1d_ale_axi_checks.py` | Internal checks of the axisymmetric ALE solver, including the cross-solver comparison. |
| `trouton1d_ale_report.py` | Reads a folder of ALE runs and prints the order and agreement tables. |

The time step is set by `--nsteps`, not by the error estimator, because on this problem
the estimator under-reports the discretisation error and never binds; each sweep
documents its sizing.

```bash
uv run python verificationCases/trouton1d-solvers/trouton1d_convergence.py --out <folder> --study all
uv run python verificationCases/trouton1d-solvers/trouton1d_axi_checks.py all
uv run python verificationCases/trouton1d-solvers/trouton1d_ale_report.py <run root>
```
