#!/usr/bin/env bash
# The equilibrium branches of this study, one new folder each under the output root given
# as the first argument: the default case (Ec 10 -> 0.41), the stiffened-cap case
# (Ec 10 -> 0.39) and the stiffened-cap case with fanned sheet columns (Ec 10 -> 0.37).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: run_equilibrium_branches.sh <output root>}
export OMP_NUM_THREADS=1 HWLOC_COMPONENTS=-gl
uv run --project "$root" python "$root/simulationCases/run_elastocapillary_equilibrium.py" \
  "$root/simulationCases/elastocapillary_equilibrium.json" --out "$out/default"
uv run --project "$root" python "$root/simulationCases/run_elastocapillary_equilibrium.py" \
  "$here/cases/stiffened_cap.json" --out "$out/stiffened-cap"
uv run --project "$root" python "$root/simulationCases/run_elastocapillary_equilibrium.py" \
  "$here/cases/fan_columns.json" --out "$out/fan-columns"
