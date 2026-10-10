# Launch records of the planar `Oh` sweep

The scripts that launched the planar sweep toward `V_TC`, kept so that a result can be
traced to the commands that produced it. They are records, not a portable driver:
the cpusets, memory limits and case parameters are those of the machine the sweep ran
on, and they call the driver interface of their time (`--oh`, `--results`); the current
driver takes one case file and `--out` (see the study README). Run roots come from the
environment (`SWEEP_ROOT` for the sweep batches, `DT_STUDY_ROOT` for the time-step
study); the checkout defaults to this repository.

| Script | What it does |
|---|---|
| `launch_batch2.sh` | Restarts `Oh` = 0.38372, 0.2377, 0.61945 and 1.0 from their checkpoints. |
| `launch_batch3.sh` | Runs `Oh` = 0.035 from `t = 0`. |
| `launch_batch4_dtstudy.sh` | The time-step study at `Oh` = 1. |

Two conventions are encoded here. Concurrent cases are limited by memory, which the
direct solver's factorisation dominates, so each script refuses to start while another
batch is live and each case runs with its own memory limit. `--n-outputs` sets the time
step as well as the output count, so it is a numerical parameter: on a restart it must be
scaled down to keep the original cadence, and `run_validation.py` warns and prints the
matched count.
