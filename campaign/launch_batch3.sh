#!/usr/bin/env bash
# Batch 3 of the u995 sweep: Oh = 0.035, re-run from t = 0.
#
# This is NOT a restart. That case's liquid/*.vtu field output and all 26 _gmsh
# REMESH files were destroyed on 2026-08-08 by a restart run made before the
# pyoomph runmode defect (fixed in fc1ab41) was understood. Its checkpoints
# cannot be loaded without those remesh files, so the only way to recover field
# output is to recompute the case.
#
# Runs alone: it is the longest case in the sweep (tstar_end = 243614 at L = 160,
# ~20.4 h to reach row 324 on the first attempt, and it must now reach 400), and
# it does not fit alongside batch 2 -- that batch already commits 140 GiB of the
# 170 GiB reserved, and this case was resident at 31.4 GiB when it was killed at
# row 324, so its peak on a full run is higher than the 30 GiB left over.
#
# The default runmode ("delete") is correct here and is deliberately not
# overridden: this is a fresh run, so clearing the stale output tree is wanted.
# pyoomph only empties SUBDIRECTORIES of run_validation, so control_part1.txt --
# a top-level file holding the original 324-row series -- survives, as does the
# sibling checkpoints_part1/. The re-run therefore produces a second, independent
# u* series in control.txt. Expect close agreement with control_part1.txt but not
# bit-identical values: PARDISO is threaded.
set -euo pipefail

B=${SWEEP_ROOT:?set SWEEP_ROOT to the u995 sweep run root}
C=${FEM_CHECKOUT:-$(cd "$(dirname "$0")/.." && pwd)}
D=$B/Oh_SB_0.035_u995

# Refuse to oversubscribe: no batch 1 or batch 2 case may still be running.
if systemctl --user list-units --all --no-legend 'etc-u995-*' | grep -q ' active '; then
  echo "another u995 case is still running; batch 3 runs alone" >&2
  systemctl --user list-units --all --no-legend 'etc-u995-*' >&2
  exit 1
fi

# The original series is the only surviving record of the destroyed run. Do not
# start a run that would overwrite control.txt unless it is already preserved.
if [[ ! -s "$D/run_validation/control_part1.txt" ]]; then
  echo "refusing to start: $D/run_validation/control_part1.txt is missing or empty" >&2
  echo "that file is the only copy of the original 324-row u* series" >&2
  exit 1
fi

systemd-run --user --unit=etc-u995-oh0.035-rerun \
  --description="elastic-taylor-culick u995 sweep Oh=0.035 full re-run from t=0" \
  --working-directory="$C" \
  --property=MemoryHigh=44G --property=MemoryMax=52G \
  --property=AllowedCPUs=8-31 \
  --setenv=OMP_NUM_THREADS=12 --setenv=HWLOC_COMPONENTS=-gl --collect \
  uv run python validation/run_validation.py \
    --oh 0.035 --tag u995 --results "$B" \
    --tstar-end 243614.0 --length 160.0 --resolution 0.06 --temporal-error 0.001 \
    --checkpoint-every 20

echo "batch 3 launched; watch with: journalctl --user -u etc-u995-oh0.035-rerun -f"
echo "expect >20 h; u* target is 0.995, the original reached 0.9891 at row 324"
