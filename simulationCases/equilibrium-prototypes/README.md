# Equilibrium prototypes: retained-grid, truncated-domain and window continuation

These scripts are the prototypes, as run, behind the static arrested-hole points that lie
beyond the continuation of the full-domain equilibrium runner
`simulationCases/run_elastocapillary_equilibrium.py`: the deepest full-domain points, which
were continued on a retained grid, and the truncated-domain and window points beyond them.
They are kept so that those points can be traced to the code that produced them. They are
**not** the supported solver interface: they have no case schema, no run manifest and no
tests, most read their options from environment variables, and each acceptance rule
reflects the stage of development at which it was used. New work should use the runner and the library in `src/`.

The only differences from the files as run are:

- the repository location (and so the library imported from `src/`) is derived from the
  script's own path;
- the case file and the retained seed segment are given through the environment
  variables `CASE` and `SEED` instead of fixed locations (the retained-grid tool already took
  them as command-line options);
- two option comments of the retained-grid tool no longer refer to its earlier version;
- the module docstrings describe the formulation instead of its development history;
- every file carries a `# ruff: noqa` header.

Numerics, defaults, tolerances and control flow are unchanged.

## What each script is

| Script | Role | Supports |
|---|---|---|
| `equilibrium_retained_checks-v6.py` | Full-domain continuation on a retained grid, after a restart gate | The deepest full-domain points of the static R_stop(Ec) and H_stop(Ec) curve |
| `proto_blob_h.py` | Truncated-domain continuation, membrane-map background | Truncated-domain points of the static R_stop(Ec) and H_stop(Ec) curve |
| `proto_blob_i.py` | Prototype H with the previous solution as the predictor background | Truncated-domain points; the start state of the window families |
| `proto_blob_l.py` | Analytic area-preserving map plus pinned nodal correction | Truncated-domain points at the deepest part of that branch |
| `extend_grid.py` | Extends the seed grid's sheet geometrically for the window runs | The reference grid of the window families |
| `march_plug2.py` | Marching-plug driver: runs a window segment script for one family of cut positions | Window H_stop(Ec) and R_stop(Ec) |
| `proto_window_q5a.py` ... `proto_window_q5l.py` | Plug-window segment scripts, eight successive acceptance versions | Window H_stop(Ec) and R_stop(Ec) |

The window versions share one formulation and differ only in how Newton iterates at the
round-off floor are accepted (see below). The letters follow the development sequence; versions that
produced no plotted point are not included, so the sequence has gaps.

## Formulations

**Retained-grid continuation (`equilibrium_retained_checks-v6.py`).** The full-domain
problem of the runner, continued from one of its saved segments without remeshing. The saved
total displacement is restored at identical nodes of the segment's own grid as a pinned
background, so no interpolation touches the crushed edge elements, and the unknown is the
increment on it. The restored state is re-solved and screened (sampled incompressibility
error, sign of the leading returned eigenvalue, residual), saved, rebuilt from the saved file
and re-solved; R, H and the incompressibility error must agree to 1e-8 across that restart.
The branch is then continued at fixed steps in log Ec, moving the increment into the
background before every step, with the same screens. The output folder must lie inside the
working directory, and `--cpus` must equal the CPU list the process is bound to.

**Truncated domain (H, I, L).** The domain is the retained grid of a full-domain equilibrium
segment, truncated at a reference cut `X_m` far behind the rim. Beyond the cut the sheet is
replaced by the depth-averaged membrane: the cut carries the membrane traction of the
universal decaying branch of the depth-averaged model, and the free surface continues
through it. The stress uses F, cof F and J directly, with the case's volume penalty; the cap
stiffening is scaled with the inverse square of the hole radius so that its relative strength
stays at its starting value. The prototypes differ in how the displacement is split into a
pinned background and an unknown, which controls the round-off floor as the hole radius grows:
H uses the membrane map of the universal branch as a nodal background; I resets that
background before every step to the previous solution moved by the membrane map; L uses the
analytic area-preserving map `sqrt(X^2 + K) - X`, with `K` a global parameter, plus a pinned
nodal correction, and a secant predictor in the radius and the nodal displacement. All three
accept a stalled Newton iterate when every reference band of `X` sits below its measured
round-off floor.

**Window (Q5a-Q5l, driven by `march_plug2.py`).** At large hole radius the material at the original
edge is crushed into the rim's nose, and a solver that meshes it fails there. The window
keeps only the sheet columns from an inner reference column `X_c` to the outer cut `X_m`. The
inner cut is a plug: its deformed shape is frozen at the start of a segment, it translates
radially by one global unknown, and that unknown balances the reactions of the kept material
against the hoop force of the removed material, `G I_c r` per radian with `I_c` the removed
region's integral of `1/X`, plus a nose constant calibrated so that the start state is an
exact equilibrium. The outer cut carries the same membrane traction as the truncated domain.
The frozen plug shape is the only approximation; its error falls with the scaled cut position
`chi = X_c / sqrt(R)`. The driver re-cuts the window at `X_c` close to `chi* sqrt(R)` each time the
radius has roughly doubled, so each family has a fixed `chi*`. The plotted window H is the
zero-cut extrapolation of the family results, computed outside these scripts.

**Acceptance versions.** The far film of a long window has a round-off floor that grows with
the hole radius, so absolute residual tolerances eventually fail although the state is
converged. Q5a applies the strict tolerance only near the rim. Q5d accepts a stall when every band is
within a factor of its measured floor and the state reproduces after a re-split and re-solve.
Q5f repeats that polish until two successive states agree. Q5g lets agreement decide
regardless of Newton's own verdict. Q5h makes the Newton tolerance adjustable. Q5i drops the residual gate at
the floor, so reproduction decides. Q5j re-splits inside Newton on a stall. Q5l measures
Newton progress on the near region only, with residual-neutral re-splits. Each script's
docstring states its rule and options in full.

## Inputs

The retained-grid tool takes command-line options: `--case` (the runner's case file),
`--grids` and `--state` (a saved segment of that run, or a state this tool saved), `--out`,
`--ndof` (the expected unknown count, checked), `--cpus`, and `--target-ec`. The other scripts
take:

- `CASE`: the case file of a full-domain run of `run_elastocapillary_equilibrium.py`; its
  material, surface tension, penalty and tolerance settings are reused.
- `SEED`: one retained segment folder of that run (`grids.npz` and `state.npz`). It is the
  reference grid and, unless `START` is set, the start state.
- `START` (optional): a saved point folder (`state.npz`, `grids.npz`) written by one of
  these scripts, to continue from it.
- For the window: `GRIDS`, a reference grid reaching far enough for the outer cut (made by
  `extend_grid.py` from the seed grid), and `INNER`, the even sheet-column index of the inner
  cut (set by the driver).

Every solver script writes `results.json` and one folder per target `Ec` with `state.npz` and
`grids.npz` into a new output folder. The driver writes `family.json` and one log per segment.

## How to run

From the repository root, with the pinned pyoomph environment, and with
`OMP_NUM_THREADS=1` and `HWLOC_COMPONENTS=-gl` set:

```bash
# full domain on a retained grid, from a saved segment down to a target Ec
taskset -c <cpus> python simulationCases/equilibrium-prototypes/equilibrium_retained_checks-v6.py \
    --case <case.json> --grids <segment>/grids.npz --state <segment>/state.npz \
    --out <new folder inside the working directory> --ndof <count> --cpus <cpus> --target-ec <Ec>

# truncated domain, from the seed segment to the listed targets
CASE=<case.json> SEED=<segment folder> \
    python simulationCases/equilibrium-prototypes/proto_blob_l.py <new folder> <X_m> <Ec> [<Ec> ...]

# long reference grid for the window
SEED=<segment folder> python simulationCases/equilibrium-prototypes/extend_grid.py <grids.npz>

# one window family: cut parameter chi*, start state, outer cut, targets
CASE=<case.json> SEED=<segment folder> GRIDS=<grids.npz> PY="$(command -v python)" \
QSCRIPT=simulationCases/equilibrium-prototypes/proto_window_q5l.py STEP=0.002 \
    python simulationCases/equilibrium-prototypes/march_plug2.py \
    <new family folder> <chi*> <start state folder> <X_m> <Ec> [<Ec> ...]
```

The window segment scripts take further options as environment variables (Newton tolerance,
stall cap, reproduction tolerances); the driver passes its environment to every segment.

The truncated-domain and window scripts ran against the library modules
`depth_averaged_equilibrium`, `elastocapillary_equilibrium`, `equilibrium_branch` and
`equilibrium_geometry` as of commit `730eec9`; those modules are unchanged in the commit that
adds this folder. The retained-grid tool ran against the library as of commit `df351e9`. Its
`inputs.json` still names that commit, because the recorded string was left as run. Since
then the library has gained an opt-in displacement background and more spectrum fields in
each recorded point. None of this changes the tool's equations or its R and H values.

## Known limitations

The scripts are kept exactly as run, so the following behaviours are documented rather than
fixed. A supported solver built from these formulations should remove them.

- **Nose calibration is not enforced.** When a window segment calibrates its nose constant
  from the start state and the calibration iterations do not reach their tolerance, the
  segment still continues from the best Newton solution. The start-radius mismatch is
  recorded in each segment's `results.json` as `start_dr_vs_start_state`; check it before
  using a segment's points.
- **The family driver always exits with status zero.** `march_plug2.py` stops at the first
  failed segment and writes the partial family summary, but it does not return a non-zero
  exit status, and it does not report targets that remain unfinished. Read the family
  summary to decide whether a family completed.
- **The grid extension assumes growth of at least one.** `extend_grid.py` does not validate
  its growth factor; a factor below one cannot reach the target length and the extension
  loop does not terminate.
