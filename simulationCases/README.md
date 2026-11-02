# axisymmetric_viscoelastic

[`run_axisymmetric_viscoelastic.py`](run_axisymmetric_viscoelastic.py) opens one hole in
an elastic sheet for one case file and writes one new output folder:
`run-manifest.json` first (the case hash, the case fields used, the units, the command
line, the component commit and the installed pyoomph revision), then `progress.jsonl`,
the input-case snapshot `input-case.json`, the live metadata `case.json` (replaced
atomically), the solver output under `run_viscoelastic_axi/` and the restart states under
`checkpoints/`. [`axisymmetric_viscoelastic.json`](axisymmetric_viscoelastic.json) is the
default case.

```bash
python simulationCases/run_axisymmetric_viscoelastic.py simulationCases/axisymmetric_viscoelastic.json --out <new folder>
python simulationCases/run_axisymmetric_viscoelastic.py simulationCases/axisymmetric_viscoelastic.json --out <new folder> --check
python simulationCases/run_axisymmetric_viscoelastic.py <case> --out <new folder> \
    --restart-from <parent run>/checkpoints/ckpt_NNNNN.dump [--t-end T] [--far-field-extension E] \
    [--newton-tolerance-safety S]
```

The runner refuses an invalid case or option, and an output folder that already holds
anything, before it writes; `--check` stops there. `--restart-from` names a checkpoint
of an earlier run, and the runner pins the parent's checkpoint, input case, metadata and
histories by SHA-256 before continuing. A continuation may make a later stop time
(`--t-end`), a longer far-field membrane (`--far-field-extension`) and a looser Newton
acceptance (`--newton-tolerance-safety`), each recorded in the manifest's command line. Its case file is either the parent's (same case hash) or one that
differs from the parent's `input-case.json` only in the numerics below
(`cases.CONTINUATION_NUMERICS`). The manifest's `restart.case_change` then lists the
changed fields, and the driver records the new values from the restart row on in the
numerics schedule of `case.json`.

| Numerics a continuation may change | |
|---|---|
| `output.cadence`, `solver.max_step` | output and far-field coupling interval, and the step cap (at most the cadence) |
| `solver.interval_start`, `solver.initial_guess`, `solver.temporal_error_fields`, `solver.newton_floor_max_dt`, `tolerances.temporal_error` | step policy, Newton start and error control |
| `physics.far_field.extension` (longer only), `.interpolation`, `.history`, `.grading`, `.auto_extend` | far-edge treatment of a membrane far field |

`--newton-tolerance-safety S` (default 10, at least 1) accepts a Newton solve once the
residual's max-norm falls below `max(1e-8, S eps (X + L)^2 / dt_f)`. Here `X + L` is the
window's outer radius and `dt_f` the time step, capped at `solver.newton_floor_max_dt` when
the case sets it; `eps (X + L)^2 / dt_f` estimates the roundoff floor. On very fine tip
meshes the actual floor can exceed ten times that estimate, so Newton stagnates just above
the tolerance and rejects time steps that are already converged to roundoff. Raising `S`
moves the acceptance above that floor; it does not change the equations or the
discretisation. The run's `case.json` records the effective factor as
`newton_tolerance_safety`.
The exit status is 0 when the run ends at its declared stop with every check passing.

**Case file** (`pyoomph-case-v1`). Every field below is read by the runner and listed in
the manifest's `case_fields_used`; other fields are descriptive.

| Field | Meaning |
|---|---|
| `schema` | `pyoomph-case-v1` |
| `case_id` | identifier of the case: letters, digits, `.`, `_` or `-` |
| `geometry` | `{"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"}` |
| `nondimensionalization` | the length, velocity and time scales (full thickness `h0`, capillary scales) |
| `physics.Oh` | Ohnesorge number `mu / sqrt(2 h0 rho sigma)`; the coded viscosity is `sqrt(2) Oh` |
| `physics.elastic_modulus` | elastocapillary number `Ec = G h0 / sigma`; 0 requires `passive_conformation` |
| `physics.relaxation_time` | relaxation time in capillary units, or `"inf"` |
| `physics.stabilization` | conformation stabilisation, `"supg"` or `"none"` |
| `physics.supg_factor` | optional SUPG factor (default 1) |
| `physics.passive_conformation` | optional; `true` transports the conformation without its stress |
| `physics.sponge` | optional absorbing layer: `start`, `width`, `gamma_max`, `profile` (`"quintic"`) |
| `physics.far_field` | optional far edge: `{"mode": "rest"}` (default) or `{"mode": "membrane", "extension": E}` with optional `inner_fraction`, `node_spacing`, `interpolation` (`"hold"`, default, or `"linear"`: the edge data vary linearly over each coupling interval), `history` (`"overwrite"`, default, or `"keep"`: the far edge keeps its previous time level), `grading` (`{"near_length", "growth"}`: membrane nodes grow geometrically beyond `near_length`) and `auto_extend` (`{"tail_fraction", "factor"}`: the membrane lengthens itself before the pulse or the window reaches its outer `tail_fraction`) |
| `mesh.length`, `mesh.r0` | window length `L` and initial hole radius |
| `mesh.resolution`, `mesh.far_resolution` | target element sizes near the rim and in the far field |
| `output.cadence` | time between control samples |
| `output.field_stride` | optional; write one field every N samples (default 1) |
| `output.checkpoint_every` | write a restart state every N samples (a multiple of the stride; 0 disables) |
| `output.write_states` | must be `false`; the driver writes its own checkpoints |
| `solver.max_step`, `solver.max_newton` | largest time step and Newton iterations |
| `solver.interval_start` | optional first step of each output interval: `"legacy"` (default, 1% of the interval), `"max_step"` or `"carry"` (the step the error controller reached) |
| `solver.initial_guess` | optional Newton start of each step: `"previous"` (default) or `"extrapolate"` (linear extrapolation of the last two steps) |
| `solver.temporal_error_fields` | optional field weights that register the temporal error estimator, e.g. `{"X": 1, "U": 1}`; `X` and `U` need equal weights; without it every step runs at the cap |
| `solver.newton_floor_max_dt` | optional cap on the step used in the Newton round-off floor |
| `tolerances.temporal_error` | adaptive time-step error tolerance (used only with `solver.temporal_error_fields`) |
| `tolerances.max_volume_error`, `.max_tip_pin_residual`, `.min_radius_increase`, `.min_polymer_stress` | checks that fail a run |
| `tolerances.max_positive_sponge_power` | optional check on the absorbing layer |
| `arrest.*` | `opening_ustar`, `arrest_ustar`, `consecutive_samples`, `trend_tolerance`, `radius_span_tolerance`: the tip-arrest detector |
| `completion` | `{"kind": "time", "value": T}` with optional `allow_recoil`; `{"kind": "event", "value": "first_post_startup_turning_point", "horizon": T}` (stop at arrest); or `{"kind": "event", "value": "height_plateau", "horizon": T, ...}` with the plateau thresholds `hold`, `minimum_window`, `window_time_fraction`, `slope_window_fraction`, `instantaneous_slope_fraction`, `residual_rms_fraction`, `adjacent_mean_fraction`, `residual_envelope_growth_fraction` and optional `oscillation_period` |

# axisymmetric_elastocapillary_equilibrium

With an infinite relaxation time the elastic sheet is a Kelvin--Voigt neo-Hookean solid,
so an arrested hole is a static equilibrium between the elastic stress and surface
tension; it depends on `Ec` alone, not on `Oh` or inertia.
[`run_elastocapillary_equilibrium.py`](run_elastocapillary_equilibrium.py) follows these
equilibria by arclength continuation in `log Ec`, from large `Ec` towards small `Ec`,
for one case file into one new output folder.
[`elastocapillary_equilibrium.json`](elastocapillary_equilibrium.json) is the default
case.

```bash
python simulationCases/run_elastocapillary_equilibrium.py simulationCases/elastocapillary_equilibrium.json --out <new folder>
python simulationCases/run_elastocapillary_equilibrium.py <case> --out <new folder> \
    --restart-from <parent run>/segments/sNNN/state.npz [--ec-stop EC]
```

The problem is solved in displacement form on the reference body, the initial sheet of
the elastic runs (hole radius `r0` on the midplane, quarter-circle edge cap of radius
`h0/2`): quadratic displacement, discontinuous linear pressure, surface energy on the
deformed free surface, and the linear far field `u_r = A/r` as a Robin traction at the
far edge. The run proceeds in segments. Each segment designs a mesh in the deformed state
of the previous one, pulls every node back through that solution to the reference body
(`taylor_culick_fem.equilibrium_geometry`), solves at the stored `Ec` and continues until
the hole radius has grown by the case's factor. After every accepted point it computes
the leading eigenvalues of the overdamped energy flow; a negative leading eigenvalue
means a strict local minimum of the energy.

The output folder holds `run-manifest.json` first, then `progress.jsonl`, the live
metadata `case.json`, `branch.csv` (one row per accepted point: segment, `Ec`, hole
radius `R`, full rim height `H`, far-field amplitude `A`, leading eigenvalue,
incompressibility error near the edge, degrees of freedom, cap stiffening) and
`segments/sNNN/` (grids, final state, rows and deformed free surface of each segment).
A continuation uses the same case and may change only `--ec-stop`. The exit status is 0
when the branch passes `Ec_stop` and 1 when it stalls.

| Field | Meaning |
|---|---|
| `schema`, `case_id`, `nondimensionalization` | as above; lengths on `h0`, stresses on `sigma/h0` |
| `geometry` | `{"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "lagrangian"}`; the legacy frame spelling `"reference"` remains accepted |
| `physics.problem` | `"axisymmetric_elastocapillary_equilibrium"` |
| `physics.r0`, `physics.surface_tension` | reference hole radius and surface tension |
| `mesh.template` | optional `"rounded-sheet-blocks"`: the rounded-cap and sheet block geometry; the following controls set its resolution |
| `mesh.elements_per_quarter` | elements around each quarter of the edge cap (nine-node quadrilaterals) |
| `mesh.length` | radial extent of the reference sheet beyond the hole |
| `mesh.growth`, `mesh.remesh_growth` | radial growth ratio of the first mesh and of the pulled-back meshes |
| `mesh.max_spacing` | largest radial element size |
| `mesh.columns` | optional, `"vertical"` (default) or `"fan"`: sheet columns behind the edge cap are vertical, or fan out along the rim front so that their tops are spaced by arclength along it (needed once the cap is crushed into a small nose, below `Ec` about 0.4) |
| `mesh.shoulder_angle`, `mesh.blend` | optional, `"fan"` only: the rim top is the first surface point whose tangent is within `shoulder_angle` degrees (default 10) of horizontal; behind it the columns return to vertical over `blend` rim half-heights (default 4) |
| `continuation.Ec_start`, `.Ec_stop` | first `Ec` and the value the branch must pass |
| `continuation.initial_step`, `.max_step` | first arclength step in `log Ec` (negative: towards small `Ec`) and its bound |
| `continuation.segment_steps`, `.segment_log_radius_growth` | steps per segment and the growth of `ln R` that ends a segment |
| `continuation.max_stalled_segments` | consecutive segments without progress before the run stops |
| `solver.newton_tolerance`, `.max_newton`, `.eigenvalues` | Newton tolerance, iteration limit and the number of eigenvalues computed per point |
| `regularisation.cap_beta` | optional Mooney--Rivlin stiffening of the edge-cap material (default 0); a numerical regularisation for the most compressed cap |
| `regularisation.volume_penalty` | optional non-negative coefficient of the augmented energy `Ec (J-1)^2/2` (default 0); the mixed pressure constraint remains in place and the added stress vanishes at `J=1` |
