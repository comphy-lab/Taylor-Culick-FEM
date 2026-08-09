#!/usr/bin/env bash
# Batch 2 of the u995 sweep restart. Run this only once batch 1 has finished and
# released its memory -- all eight cases cannot fit in 246 GiB to completion.
#
# Batch 1 (Oh_SB 0.0565, 0.14724, 0.09121) reserves 170 GiB of MemoryMax between
# them. Check they are gone first:
#     systemctl --user list-units 'etc-u995-*'
#
# Every case restarts from its newest checkpoint. run_validation.py truncates
# control.txt on restart and renumbers checkpoints from ckpt_0020, so the
# pre-restart history is preserved per case as control_part1.txt and
# checkpoints_part1/. Concatenate them in post-processing.
set -euo pipefail

B=/mnt/stokesInt1/running-simulations/elastic-taylor-culick/u995-sweep
C=/home/vatsal/CMP-codes/Taylor-Culick-FEM

for u in 0.0565 0.14724 0.09121; do
  if systemctl --user is-active --quiet "etc-u995-oh${u}.service"; then
    echo "batch 1 case Oh_SB=${u} is still running; refusing to oversubscribe" >&2
    exit 1
  fi
done

launch() {
  local oh=$1 dir=$2 ts=$3 L=$4 ck=$5 hi=$6 mx=$7
  systemd-run --user --unit="etc-u995-oh${oh}" \
    --description="elastic-taylor-culick u995 sweep Oh_SB=${oh} restart" \
    --working-directory="$C" \
    --property=MemoryHigh="$hi" --property=MemoryMax="$mx" \
    --property=AllowedCPUs=8-31 \
    --setenv=OMP_NUM_THREADS=6 --setenv=HWLOC_COMPONENTS=-gl --collect \
    /home/vatsal/.local/bin/uv run python validation/run_validation.py \
      --oh "$oh" --tag u995 --results "$B" \
      --tstar-end "$ts" --length "$L" --resolution 0.06 --temporal-error 0.001 \
      --checkpoint-every 20 \
      --restart-from "$B/$dir/checkpoints/$ck"
}

# MemoryMax sums to 140 GiB, inside the 170 GiB reserved for this simulation.
# Values are the measured pre-crash resident sizes plus growth headroom.
launch 0.38372 Oh_SB_0.38372_u995 7556.0  250.0 ckpt_0320.dump 40G 46G
launch 0.2377  Oh_SB_0.2377_u995  14276.0 210.0 ckpt_0320.dump 34G 40G
launch 0.61945 Oh_SB_0.61945_u995 3918.0  290.0 ckpt_0320.dump 24G 28G
launch 1.0     Oh_SB_1_u995       2049.0  340.0 ckpt_0280.dump 22G 26G

echo "batch 2 launched; watch with: journalctl --user -u 'etc-u995-*' -f"
