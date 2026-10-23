# Axisymmetric hole against Savva & Bush (2009)

Code-to-code comparison of the axisymmetric FEM with the hole-opening solutions of
Savva & Bush (2009), figures 10 (`Oh` = 10) and 16 (`Oh` = 1). Two formulations of the
same problem are provided, so they can also be compared with each other:

| File | Role |
|---|---|
| `run_validation_axi.py` | Laboratory frame: a graded annulus with an initial hole radius `--r0` and a fixed outer wall `--rout`, sized to contain the whole trajectory. |
| `run_validation_axi_comoving.py` | Co-moving window of length `--window-length` dragged by the rim; the stop time is unbounded by the domain. |
| `cases/` | One case per `Oh` for each formulation (`axi-lab-*`, `axi-comoving-*`). |
| `run_axi_lab_sweep.sh`, `run_axi_comoving_sweep.sh` | The two sweeps, one new folder per case under the given output root. |
| `data/` | Digitised curves with their extraction overlays. |

The two drivers take the same options apart from `--rout` and `--window-length`, so a
comparison is the same command line with one option changed. The co-moving driver is
also the production driver for the Newtonian axisymmetric runs at low `Oh`.

```bash
validationCases/savva-bush-axisymmetric/run_axi_comoving_sweep.sh <output root>
uv run python validationCases/savva-bush-axisymmetric/run_validation_axi.py \
    validationCases/savva-bush-axisymmetric/cases/axi-lab-oh1.json --out <new folder>
uv run python validationCases/savva-bush-axisymmetric/run_validation_axi_comoving.py \
    validationCases/savva-bush-axisymmetric/cases/axi-comoving-oh1.json --out <new folder>
```

Each driver runs one case into one new folder and writes `run-manifest.json` first; the
case carries `physics.problem` and `physics.Oh`, and every numerical parameter is an
option recorded in the manifest. A continuation (`--restart-from <state>`) writes to its
own new folder.
