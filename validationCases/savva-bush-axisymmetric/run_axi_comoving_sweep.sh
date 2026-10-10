#!/usr/bin/env bash
# The axi-comoving sweep of this study: Oh = 0.5, 1, 5 and 10 to t* = 30, one case per new
# folder under the output root given as the first argument.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: run_axi_comoving_sweep.sh <output root>}
for c in axi-comoving-oh0p5 axi-comoving-oh1 axi-comoving-oh5 axi-comoving-oh10; do
  uv run --project "$root" python "$here/run_validation_axi_comoving.py" "$here/cases/$c.json" --out "$out/$c" \
    --resolution 0.06 --tstar-end 30 --n-outputs 400 --temporal-error 0.001 --r0 1 --window-length 40
done
