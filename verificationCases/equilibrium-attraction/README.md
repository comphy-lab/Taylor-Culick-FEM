# Attraction of the arrested-hole equilibria

The steady equilibria of
[`run_elastocapillary_equilibrium.py`](../../simulationCases/run_elastocapillary_equilibrium.py)
carry their own linear evidence: the leading eigenvalue of the overdamped energy flow at
every point of the branch, negative for a strict local minimum of the elastic plus surface
energy. With an infinite relaxation time that energy plus the kinetic energy decreases at
the rate of viscous dissipation, so a strict local minimum is asymptotically stable for
the full damped dynamics in the axisymmetric class.

This check tests the nonlinear basin directly: an equilibrium of one `Ec` relaxes under
the energy flow of another and must arrive at the steady solution of the second, from
either side, at the decay rate the leading eigenvalue predicts. Non-axisymmetric
perturbations are outside its scope.

| File | Role |
|---|---|
| `relax_to_equilibrium.py` | Relaxes the equilibrium at `--ec-start` under the flow at `--ec-target`; writes `run-manifest.json`, `relax.csv` and `summary.json`. |
| `relaxation.json` | Case supplying the reference geometry, a coarse mesh and the solver settings. |

```bash
uv run python verificationCases/equilibrium-attraction/relax_to_equilibrium.py verificationCases/equilibrium-attraction/relaxation.json --ec-start 0.8 --ec-target 0.7 --out <folder>
uv run python verificationCases/equilibrium-attraction/relax_to_equilibrium.py verificationCases/equilibrium-attraction/relaxation.json --ec-start 0.6 --ec-target 0.7 --out <folder>
```
