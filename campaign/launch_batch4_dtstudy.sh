#!/usr/bin/env bash
# Batch 4: timestep-refinement study for the 2D co-moving FEM.
#
# WHY. The 1D Trouton solver was found on 2026-08-08 to be temporally
# unconverged in a way its own error control could not see: temporal_error from
# 1e-3 to 1e-6 gave bit-identical answers while the solver reported an estimated
# timestepping error of 1e-12, against a measured error of O(1e-3). dt therefore
# sat on maxstep every step, and halving maxstep moved u*(30) at a clean
# first-order rate out to a Richardson limit of 0.98073 -- about 3% above both
# the 2D FEM and the digitised Savva & Bush value it was said to agree with.
# See validation/trouton1d_convergence.py (commit e581054).
#
# The 2D solver shows the SAME signature in its logs -- "Estimated timestepping
# error is 1e-12" and "GLOBAL TEMPORAL ERROR 0" -- and it is on pyoomph's
# default BDF2, not the 1D case's forced BDF1, so this is not a BDF1 artefact.
# run_validation.py clamps maxstep to the current geometric output interval, so
# the 2D timestep is set by the OUTPUT SCHEDULE. If the same insensitivity holds
# here, every u* in the campaign is a function of how often we chose to write
# output, which is not a defensible basis for a published number.
#
# The magnitude in 2D is UNMEASURED. That is what this script settles.
#
# NOTE. --n-outputs did not exist until commit aae8255; the count was hardcoded
# at 400, so an earlier draft of this script varied only the output TAG and
# would have produced four identical runs. It sets dt, so it is the variable.
#
# METHOD. One Oh, short horizon, four output densities. Fixing --tstar-end well
# below the campaign target keeps each run cheap while still spanning the range
# where the schedule binds. Compare the final u* across nout: if it drifts at
# first order the way the 1D case did, the campaign needs a resolved dt before
# any u* is quoted.
#
# Oh = 1.0 is chosen deliberately: it is the case with published data to
# check against (Savva & Bush fig 6) and the stiffest viscous solve, so it is
# where a temporal problem shows up first.
set -euo pipefail

B=/mnt/stokesInt1/running-simulations/elastic-taylor-culick/u995-sweep
C=/home/vatsal/CMP-codes/Taylor-Culick-FEM
OUT=/mnt/stokesInt1/running-simulations/elastic-taylor-culick/dt-study-2d

if systemctl --user list-units --all --no-legend 'etc-u995-*' | grep -q ' active '; then
  echo "u995 cases are still running; this study runs after batch 3" >&2
  exit 1
fi

mkdir -p "$OUT"

# Each run is independent and cheap relative to a campaign case; run them in
# sequence so the comparison is not confounded by memory pressure.
for nout in 100 200 400 800; do
  echo "=== nout=${nout} ==="
  systemd-run --user --unit="etc-dt2d-n${nout}" --wait --collect \
    --description="elastic-taylor-culick 2D dt study, nout=${nout}" \
    --working-directory="$C" \
    --property=MemoryHigh=40G --property=MemoryMax=48G \
    --property=AllowedCPUs=8-31 \
    --setenv=OMP_NUM_THREADS=8 --setenv=HWLOC_COMPONENTS=-gl \
    /home/vatsal/.local/bin/uv run python validation/run_validation.py \
      --oh 1.0 --tag "dt${nout}" --results "$OUT" --n-outputs "${nout}" \
      --tstar-end 30.0 --length 40.0 --resolution 0.06 --temporal-error 0.001
done

echo
echo "final u* by output density (maxstep tracks the output interval):"
for nout in 100 200 400 800; do
  f="$OUT/Oh_SB_1_dt${nout}/run_validation/control.txt"
  [[ -s $f ]] && printf "  nout=%-5s u*(30) = %s\n" "$nout" "$(awk 'END{printf "%.10f", $4}' "$f")"
done
echo
echo "If these differ beyond ~1e-4, the campaign u* values are schedule-dependent"
echo "and a resolved dt is needed before any of them is published."
