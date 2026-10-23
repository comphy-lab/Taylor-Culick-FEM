"""Convergence study for the pyoomph 1D Trouton solver.

Exists because the original study was run from an ad-hoc driver that was never
committed, so its `u*(t* = 30) = 0.95180` could not be reproduced or re-checked.
Run this instead of hand-rolling another one.

What it establishes, at Oh = 1:

MESH is converged and the domain does not matter. u*(30) settles to ~1e-10 over
nelem 50 -> 400, and M = 120 and M = 240 agree to the same order. An earlier
report of "bit-identical" values across refinement is consistent with printing
at five decimals: the variation is real but lives in the 8th decimal and below.

TIME is NOT converged, and the tolerance does not control it. `temporal_error`
from 1e-3 to 1e-6 returns bit-identical answers while the solver reports
"Estimated timestepping error is 1e-12" and "GLOBAL TEMPORAL ERROR 0". The
measured discretisation error is O(1e-3), so the estimator under-reports by
about nine orders and never binds; dt is pinned to `maxstep` on every step.
Halving maxstep then changes u*(30) at a clean first-order rate (ratio 2.00,
which is BDF1 behaving as BDF1), and Richardson extrapolation to dt -> 0 gives

    u*(30) -> 0.98073

against 0.95026 from the 2D co-moving FEM and 0.9542 from the digitised
Savva & Bush curve. The previously reported three-way agreement to 0.16% / 0.25%
was therefore a coarse-timestep coincidence, not a converged result. Treat any
u* from this solver as a function of `maxstep` until that is resolved.

Print full precision. Do not reduce these to five decimals again.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from taylor_culick_fem.trouton1d_pyoomph import Trouton1DProblem  # noqa: E402

V_TC = math.sqrt(2.0)


def run_case(oh: float, mass: float, nelem: int, tstar_end: float,
             nsteps: int, temporal_error: float, outroot: Path) -> float:
    """One solve; returns u*(tstar_end) = u_tip / V_TC."""
    os.chdir(outroot)
    with Trouton1DProblem() as p:
        p.Oh = oh
        p.mass = mass
        p.nelem = nelem
        p.set_output_directory(
            str(outroot / f"oh{oh:g}_n{nelem}_M{mass:g}_s{nsteps}_e{temporal_error:g}"))
        p.initialise()
        t_end = tstar_end * p.tau_vis
        # startstep is deliberately tiny: the impulsively loaded tip has a
        # sqrt(t) start-up singularity and a large first step corrupts it.
        p.run(t_end, startstep=t_end * 1e-6, maxstep=t_end / nsteps,
              temporal_error=temporal_error)
        return p.tip_velocity() / V_TC


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oh", type=float, default=1.0)
    ap.add_argument("--tstar-end", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True,
                    help="output folder for the study")
    ap.add_argument("--study", choices=("mesh", "time", "tolerance", "all"), default="all")
    args = ap.parse_args()
    args.out = args.out.resolve()  # run_case changes directory
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Oh={args.oh}  t*_end={args.tstar_end}  V_TC={V_TC!r}", flush=True)

    if args.study in ("mesh", "all"):
        print("\n=== MESH (and domain length) ===", flush=True)
        for mass in (120.0, 240.0):
            print(f"  M = {mass}", flush=True)
            prev = None
            for nelem in (50, 100, 200, 400):
                u = run_case(args.oh, mass, nelem, args.tstar_end, 200, 1e-3, args.out)
                d = "" if prev is None else f"   diff={u - prev:+.3e}"
                print(f"    nelem={nelem:<5d} u*={u!r}{d}", flush=True)
                prev = u

    if args.study in ("tolerance", "all"):
        print("\n=== TOLERANCE (expect NO effect -- the estimator never binds) ===", flush=True)
        for terr in (1e-3, 1e-4, 1e-5, 1e-6):
            u = run_case(args.oh, 120.0, 200, args.tstar_end, 200, terr, args.out)
            print(f"    temporal_error={terr:<8g} u*={u!r}", flush=True)

    if args.study in ("time", "all"):
        print("\n=== TIMESTEP (first order; this is the one that matters) ===", flush=True)
        prev = prev_d = None
        for nsteps in (200, 400, 800, 1600, 3200, 6400):
            t0 = time.time()
            u = run_case(args.oh, 120.0, 200, args.tstar_end, nsteps, 1e-3, args.out)
            d = None if prev is None else u - prev
            r = "" if (d is None or prev_d is None) else f" ratio={prev_d / d:5.2f}"
            tail = "" if d is None else f"  diff={d:+.3e}"
            print(f"    maxstep=t_end/{nsteps:<6d} u*={u!r}{tail}{r}  [{time.time() - t0:.0f}s]",
                  flush=True)
            prev_d, prev = d, u
        if prev is not None and prev_d is not None:
            # First-order Richardson: u_exact ~ 2*u_fine - u_coarse.
            print(f"\n    Richardson (p=1) dt->0 limit: {prev + prev_d!r}", flush=True)


if __name__ == "__main__":
    main()
