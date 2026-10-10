# AGENTS.md

Sharp-interface FEM study of Taylor–Culick sheet retraction in a tip co-moving
frame, built on pyoomph. Read `README.md` first for the physics and the case
list.

## Environment

`uv` owns the environment. Never `pip install` into `.venv` by hand, never
create a conda environment for this project, and never edit `.python-version`
or the `requires-python` bound without checking that the whole dependency set
still resolves and `python -m pyoomph check all` passes.

```bash
uv sync                                # build/refresh the environment
uv run python simulationCases/planar_comoving.py
```

pyoomph is a **source build** of an immutable CoMPhy release commit
(`comphy-2026.09.07.1`, `a186e07e`), declared through `[tool.uv.sources]`. Do
not follow `develop`; move the pin only deliberately, to another release tag.
It compiles vendored oomph-lib plus downloaded CLN/GiNaC, so a cold `uv sync`
is slow. `uv.lock` pins the exact pyoomph commit; commit the lockfile whenever
it changes so a result can be traced to a solver build.

## Physics invariants

Do not silently change any of these. They are what make the result comparable to
the Basilisk reference and to theory.

- Non-dimensionalisation is on the **full** sheet thickness `h0`, with
  `rho = sigma = h0 = 1`. The half-thickness is `0.5`. Getting this wrong moves
  the answer by exactly `sqrt(2)`.
- The target is `V_TC = sqrt(2 sigma / (rho h0)) = sqrt(2) = 1.41421...`.
- `Oh = mu / sqrt(2 h0 rho sigma)` is the project's ONLY Ohnesorge number
  (Savva & Bush convention). The coded viscosity `mu = sqrt(2) Oh` is never
  called `Oh`. Coded `mu = 0.05` is `Oh = 0.035`: a convenient default, NOT a
  case tuned to match anyone's published run. We do not claim to reproduce
  Sanjay et al.; at `Oh << 1` the behaviour is qualitatively insensitive to
  the precise `Oh`.
- Single-phase: there is no gas. Do not reintroduce a second fluid to "match"
  Basilisk; the whole point is the sharp free surface.

## pyoomph traps that apply to this repo

These are load-bearing. Violating them produces a plausible, wrong answer.

- **Never use `DirichletBC` with a value that depends on an unknown.** The
  co-moving frame velocity `U` is a global degree of freedom, so the far-field
  condition must use `EnforcedDirichlet` / `EnforcedBC`, which impose it through
  a Lagrange-multiplier field.
- **Point constraints on the symmetry axis vanish in axisymmetry.** The measure
  carries a factor `2*pi*r`, so any weak contribution evaluated at `r = 0`
  integrates to exactly zero. Pass `coordinate_system=cartesian` to
  `WeakContribution` / `EnforcedBC` when constraining the rim on the axis, or
  express the constraint as a surface integral instead. The integral form is
  better conditioned and is the pattern pyoomph's own examples validate.
- **Correct for frame acceleration.** When `U` is time-dependent, the co-moving
  frame is non-inertial. Feed `rho * partial_t(U)` into the momentum equation
  through the `bulkforce`/`gravity` argument of `NavierStokesEquations`, or the
  inertia is wrong during the transient.
- **A moving mesh needs a mesh-motion equation.** `NavierStokesFreeSurface`
  resolves `static_interface="auto"` to *moving* only when a
  `BaseMovingMeshEquations` is present on the bulk domain.
- `partial_t(..., ALE="auto")` is only ALE-corrected once mesh equations exist.

## Validation order

`simulationCases/planar_comoving.py` must pass before any axisymmetric result is
believed, because it is the only case with published data to check against:
Savva & Bush, JFM 626 (2009), figure 6. At `Oh = 0.5` this code gives
`U/V_TC = 0.9297` at `t* = 30` against their ~0.93. Treat that as the
regression target.

Neither geometry has a true steady state -- the rim collects mass and grows
without bound in both -- so a stationary solve diverges and the control only
*approaches* `sqrt(2)`. Do not read an intermediate plateau as a converged
value: the planar case sits near `0.83` from `t ~ 1` to `t ~ 3` and then climbs
away from it. Comparing two runs early in the transient proves nothing about
convergence, because every run agrees there regardless of mesh or domain.

## Repository conventions

- `simulationCases/` holds runnable problem definitions, one physical configuration each.
- `src/taylor_culick_fem/` holds shared equations, mesh templates and
  diagnostics. Anything used by two cases belongs here.
- `postProcess/` holds analysis and figure generation, never simulation state.
- Simulation output does not belong in this repository. Runs write to the
  project run root on the compute host; the tracker records custody.
