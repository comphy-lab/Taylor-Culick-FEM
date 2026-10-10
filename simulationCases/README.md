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
    --restart-from <parent run>/checkpoints/ckpt_NNNNN.dump [--t-end T] [--far-field-extension E]
```

The runner refuses an invalid case or option, and an output folder that already holds
anything, before it writes; `--check` stops there. A continuation uses the **same case
file**: `--restart-from` names a checkpoint of an earlier run of that case, whose
`run-manifest.json` must carry the same case hash, and the runner pins the parent's
checkpoint, input case, metadata and histories by SHA-256 before continuing. The only
changes a continuation may make are a later stop time (`--t-end`) and a longer far-field
membrane (`--far-field-extension`), and the manifest records both in its command line.
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
| `physics.far_field` | optional far edge: `{"mode": "rest"}` (default) or `{"mode": "membrane", "extension": E}` with optional `inner_fraction`, `node_spacing` |
| `mesh.length`, `mesh.r0` | window length `L` and initial hole radius |
| `mesh.resolution`, `mesh.far_resolution` | target element sizes near the rim and in the far field |
| `output.cadence` | time between control samples |
| `output.field_stride` | optional; write one field every N samples (default 1) |
| `output.checkpoint_every` | write a restart state every N samples (a multiple of the stride; 0 disables) |
| `output.write_states` | must be `false`; the driver writes its own checkpoints |
| `solver.max_step`, `solver.max_newton` | largest time step and Newton iterations |
| `tolerances.temporal_error` | adaptive time-step error tolerance |
| `tolerances.max_volume_error`, `.max_tip_pin_residual`, `.min_radius_increase`, `.min_polymer_stress` | checks that fail a run |
| `tolerances.max_positive_sponge_power` | optional check on the absorbing layer |
| `arrest.*` | `opening_ustar`, `arrest_ustar`, `consecutive_samples`, `trend_tolerance`, `radius_span_tolerance`: the tip-arrest detector |
| `completion` | `{"kind": "time", "value": T}` with optional `allow_recoil`; `{"kind": "event", "value": "first_post_startup_turning_point", "horizon": T}` (stop at arrest); or `{"kind": "event", "value": "height_plateau", "horizon": T, ...}` with the plateau thresholds `hold`, `minimum_window`, `window_time_fraction`, `slope_window_fraction`, `instantaneous_slope_fraction`, `residual_rms_fraction`, `adjacent_mean_fraction`, `residual_envelope_growth_fraction` and optional `oscillation_period` |
