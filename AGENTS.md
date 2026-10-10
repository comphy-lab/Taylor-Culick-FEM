# AGENTS.md

Operating notes for this repository. Read `README.md` first for the problem and the
layout.

## Environment

`uv` owns the environment. Never `pip install` into an environment by hand, never
create a conda environment for this project, and never edit `.python-version` or the
`requires-python` bound without checking that the whole dependency set still resolves
and `python -m pyoomph check all` passes.

pyoomph is a **source build** of an immutable CoMPhy release commit
(`comphy-2026.09.07.1`, `a186e07e`), declared through `[tool.uv.sources]`. Do not follow
`develop`; move the pin only deliberately, to another release tag. It compiles vendored
oomph-lib plus downloaded CLN/GiNaC, so a cold `uv sync` is slow. `uv.lock` pins the
exact pyoomph commit; commit the lockfile whenever it changes so a result can be traced
to a solver build. Set `HWLOC_COMPONENTS=-gl` where hwloc could probe an unresponsive
graphics display while MPI starts; the probe then blocks indefinitely.

## Physics invariants

Do not silently change any of these. They are what make the result comparable to
other solvers and to theory.

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

## Scientific rules

- The planar sheet is checked first, because it is the configuration with published
  numerical solutions to compare against (`validationCases/savva-bush-planar/`); no
  axisymmetric result is believed before it.
- Neither geometry has a true steady state: the rim collects mass and grows, a stationary
  solve diverges and the control only *approaches* `V_TC`. Do not read an intermediate
  plateau as a converged value, and do not compare two runs early in the transient, where
  every run agrees regardless of mesh or domain.
- Report verification, convergence and comparison with published data as separate tests.
  An independent group's published numerical solution of the same equations is validation
  (`validationCases/`), stated as code-to-code agreement, never as agreement with
  experiment or physical reality. An independent discretisation of the same problem in
  this repository is verification (`verificationCases/`). A comparison with a different,
  reduced model belongs in `modelComparisonCases/`.

## Repository rules

- Layout: library code in `src/taylor_culick_fem/`; the entry point
  `simulationCases/run_axisymmetric_viscoelastic.py` and its default case in
  `simulationCases/`; each evidence folder owns its drivers, cases and reference data;
  tools acting on run outputs in `postProcess/`; tests in `testCases/`. No `docs/`, no
  second cases tree and no runner at the repository root. Library code imports no
  entry-point folder.
- An entry point is never named after the package: `simulationCases/taylor_culick_fem.py`
  would shadow `src/taylor_culick_fem/`, because a script's own folder is first on
  `sys.path`.
- `[tool.uv] package = false`, so `src/` is not installed. Runners, study scripts and
  `testCases/__init__.py` put it on `sys.path`; run the tests from the repository root
  with `python -m pytest testCases`.
- Every case is validated by `src/taylor_culick_fem/cases.py` before any output, and the
  validator names every problem. Keep `CASE_FIELDS_USED` and the case-schema table in
  `simulationCases/README.md` in step with the fields the runner reads.
- Every run writes `run-manifest.json` before any other output, binding the case hash, the
  component commit and the pyoomph commit, and never replaces it. Every run, fresh or
  restart, uses a new output folder; a continuation uses the same case with
  `--restart-from` and may change only the stop time and the membrane length (the
  elastic runner), or the stop time and the output count (the Newtonian study drivers,
  whose output count sets the step).
- Live metadata (`case.json`) is replaced atomically; histories are appended.
- Study run scripts (`run_*.sh`) carry the exact production options and take the output
  folder as their argument; `testCases/` parses each against its runner's options.
- Never commit run output, results, figures, run lists (`runs.toml`), `data-roots.toml`,
  run identities, archive paths, machine names, internal run numbers in code comments, or
  AI co-author trailers. Results and their discussion are not documented here; a
  published figure states only the solver commit and the pinned pyoomph version.
- Local environment and build directories (`.venv*`, `.pyoomph-*`) are ignored.
- `CLAUDE.md` is local and ignored; this file is the committed operating manual.
- pyoomph's coordinate-aware integrals already carry the axisymmetric `2*pi*r` measure;
  do not apply it twice.

## Documentation boundary

<!-- documentation-boundary-v1 -->

All content here, including this file, is public-candidate and uses calibrated research
prose. Provisional results, run records and debugging narrative do not belong in this
repository.
