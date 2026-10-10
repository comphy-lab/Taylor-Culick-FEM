# Axisymmetric hole against Savva & Bush (2009)

Code-to-code comparison of the axisymmetric FEM with the hole-opening solutions of
Savva & Bush (2009), figures 10 (`Oh` = 10) and 16 (`Oh` = 1). Two formulations of the
same problem are provided, so they can also be compared with each other:

| File | Role |
|---|---|
| `run_validation_axi.py` | Laboratory frame: a graded annulus with an initial hole radius `--r0` and a fixed outer wall `--rout`, sized to contain the whole trajectory. |
| `run_validation_axi_comoving.py` | Co-moving window of length `--window-length` dragged by the rim; the stop time is unbounded by the domain. |
| `data/` | Digitised curves with their extraction overlays. |

The two drivers take the same options apart from `--rout` and `--window-length`, so a
comparison is the same command line with one option changed. The co-moving driver is
also the production driver for the Newtonian axisymmetric runs at low `Oh`.

```bash
uv run python validationCases/savva-bush-axisymmetric/run_validation_axi.py --results <run root> --oh 1
uv run python validationCases/savva-bush-axisymmetric/run_validation_axi_comoving.py --results <run root> --oh 1
```
