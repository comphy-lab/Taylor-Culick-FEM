# Campaign scripts

The exact scripts used to run the `Oh` sweep on Stokes, kept here so a result
can be traced to the commands that produced it. They are **operational
records**, not a portable driver: cpusets, memory limits and case parameters
are those of the machine the campaign ran on. Run roots come from the
environment (`SWEEP_ROOT` for the sweep batches, `DT_STUDY_ROOT` for the
time-step study; the checkout defaults to this repository). Read them before reusing them.

Simulation data is not in this repository. These write to the project run root
on the compute host; custody is recorded in the project tracker.

| script | what it does |
|---|---|
| `launch_batch2.sh` | Restarts `Oh` 0.38372, 0.2377, 0.61945, 1.0 from their checkpoints. |
| `launch_batch3.sh` | Re-runs `Oh = 0.035` from `t = 0` (its field output was lost; see below). |
| `launch_batch4_dtstudy.sh` | The timestep study of `docs/planar-campaign/planar-campaign.tex` (`sec:timestep`). |

## Conventions these encode

**Batching is a memory constraint, not a preference.** Memory is dominated by
the PARDISO factorisation at roughly 10.5 kB/dof, and eight concurrent cases
reached 216 GiB resident on a 246 GiB host — enough to trigger a host-wide OOM
on 2026-08-08. Each script therefore refuses to start while another batch is
live, and each case runs in a systemd scope with its own `MemoryMax`.

**`--n-outputs` sets the timestep**, so it is a physics-relevant parameter and
not an output preference. See `docs/planar-campaign/planar-campaign.tex`
(`sec:method`, `--n-outputs`; `sec:timestep`). On a restart it
must be scaled down to preserve the original cadence; `run_validation.py` warns
and prints the matched count.

**`Oh = 0.035` lost its field output.** A restart run made before the
`--restart-from` defect was understood (fixed in `fc1ab41`) deleted that case's
`liquid/*.vtu` tree and its `_gmsh` remesh files. Its `control.txt` and
checkpoints survive, so the `u*(t)` series is intact and only the field output
is gone; `launch_batch3.sh` recomputes it from `t = 0`, which is the only route
back.
