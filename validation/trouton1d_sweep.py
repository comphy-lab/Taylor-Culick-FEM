#!/usr/bin/env python3
"""1D Trouton sheet: u*(t) at arbitrary Oh, sized and dt-controlled per case.

WHY THIS EXISTS. ``trouton1d_convergence.py`` answers "is the solver converged
at Oh_SB = 1, t* = 30" and returns a single scalar per run. It cannot produce
the quantity the campaign actually wants, which is the whole curve u*(t) at the
eight Oh_SB values the 2D co-moving FEM sweep covers, on a domain and timestep
sized for each one.

THE TIMESTEP IS THE WHOLE PROBLEM, AND IT IS NOT SELF-DIAGNOSING. On this
problem pyoomph's temporal error estimator reports ~1e-12 against a 1e-3
tolerance while the true discretisation error is O(1e-3) -- it under-reports by
about nine orders and therefore never binds. dt sits on ``maxstep`` on every
single step. The solver is additionally forced onto BDF1 (BDF2 fails at its
third timestep, see trouton1d_pyoomph.py), so the error is FIRST order in dt.
Consequence: any u* from this code is a function of maxstep until proven
otherwise, and a run that "looks converged" because the estimator says 1e-12 is
telling you nothing. That is how an earlier three-way agreement with Savva &
Bush to 0.16% turned out to be a coarse-timestep coincidence; Richardson
extrapolation to dt -> 0 moved u*(30) from 0.9764 to 0.98073.

So this driver holds maxstep FIXED for the whole run rather than tying it to
the output interval, which is what makes a clean two-level Richardson
extrapolation possible: halve nsteps, halve dt everywhere, and the first-order
error term halves too. Run each case at nsteps and 2*nsteps and combine with
``--richardson`` in postProcess, or read the pair directly.

Outputs are still GEOMETRIC in time, because u* rises through most of its range
in the first per cent of the run and uniform sampling renders the early curve
as straight segments. Output spacing and timestep are decoupled here; in
run_validation.py (2D) they are not, which is a separate trap documented there.

DOMAIN SIZING. The far end is a clamped undisturbed sheet (h=1, u=0), valid only
while the disturbance has not reached it. In the Lagrangian mass coordinate the
rim accumulates mass at roughly h_far * V_TC = V_TC per unit time, so the
disturbed region grows like V_TC * t. The default mass is 2.5x that, and
``far_field_drift`` is recorded every output so the assumption is checked rather
than assumed -- a disturbance arriving at a clamped inlet corrupts the answer
silently. Raising ``--mass`` costs elements: the graded mesh puts the tip cell
at M/N^2, so N is scaled to hold that at the value validated by the mesh study
(M=120, N=400 => 7.5e-4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulationCases"))
from trouton1d_pyoomph import Trouton1DProblem  # noqa: E402

V_TC = math.sqrt(2.0)

# Tip cell size in mass units validated by the mesh study at M=120, N=400,
# grade=2: 120 * (1/400)^2. Holding this fixed is what makes nelem(mass)
# defensible rather than arbitrary.
TIP_CELL = 120.0 / 400.0 ** 2


def auto_mass(t_end: float) -> float:
    return max(120.0, 2.5 * V_TC * t_end)


def auto_nelem(mass: float) -> int:
    return int(min(3000, max(400, round(math.sqrt(mass / TIP_CELL)))))


def run_case(oh_sb: float, tstar_end: float, nsteps: int, n_outputs: int,
             temporal_error: float, outroot: Path, mass: float | None,
             nelem: int | None, drift_tol: float,
             max_newton: int | None = None, gcn: bool = False) -> dict:
    # ONE CASE PER DIRECTORY. An earlier version put each case in a subdirectory
    # of a shared root; pyoomph did not honour that and wrote into the root
    # instead, so the metadata write then failed on a directory that did not
    # exist. The caller supplies a unique --out per case and this writes there.
    outdir = outroot
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    with Trouton1DProblem() as p:
        p.Oh_SB = oh_sb
        t_end = tstar_end * p.tau_vis
        p.mass = auto_mass(t_end) if mass is None else mass
        p.nelem = auto_nelem(p.mass) if nelem is None else nelem
        p.set_output_directory(str(outdir))
        p.initialise()
        # pyoomph dumps a full restart state on EVERY internal step. At the step
        # counts this driver runs (1e5-1e6) that is tens of GB of dumps for a
        # solve whose entire useful output is one text file. The 1D state is
        # cheap to recompute, so restartability is not worth that.
        p.write_states = False

        dt = t_end / nsteps
        meta = {
            "Oh_SB": oh_sb, "mu": p.mu, "tau_vis": p.tau_vis,
            "tstar_end": tstar_end, "t_end": t_end,
            "mass": p.mass, "nelem": p.nelem, "grade": p.grade,
            "nsteps": nsteps, "dt": dt, "n_outputs": n_outputs,
            "temporal_error": temporal_error,
            "with_capillarity": p.with_capillarity,
            "scheme": p.default_timestepping_scheme,
            "V_TC": V_TC,
        }
        print(f"[oh={oh_sb:g}] mu={p.mu:.6g} tau_vis={p.tau_vis:.6g} "
              f"t_end={t_end:.6g} M={p.mass:.6g} N={p.nelem} dt={dt:.6g}",
              flush=True)

        history = p.create_text_file_output(
            "control1d.txt",
            header=["t", "tstar", "u_tip", "ustar", "h_tip", "drift_h", "drift_u"])

        out_times = np.geomspace(t_end * 1e-4, t_end, n_outputs)
        prev = 0.0
        worst_drift = 0.0
        for t in out_times:
            # startstep tiny on the first leg only: the impulsively loaded tip
            # has a sqrt(t) start-up singularity and a large first step
            # corrupts it. maxstep is the SAME on every leg -- that is the
            # point of this driver.
            start = t_end * 1e-6 if prev == 0.0 else min(dt, t - prev)
            # outstep=False IS LOAD-BEARING. Problem.run documents outstep as
            # defaulting to True whenever temporal_error is passed -- and True
            # means write every Output object on EVERY INTERNAL TIMESTEP, not
            # once per run() call. At this driver's step counts that is one
            # sheet_*.txt per step: a measured 2976 files for 110 output rows,
            # extrapolating to ~550 GB per run and 6.6 TB across twelve
            # concurrent cases, against 382 GB free on reynolds. Suppress it and
            # call output() once per leg instead, giving exactly n_outputs
            # profiles at the times the series records.
            # LATE-TIME NEWTON FAILURE. Runs at Oh_SB = 0.035 and 0.09121 died
            # with the residual stalled near 1.4e-3 after the default 10 Newton
            # steps, at h_tip = 337 and 215 -- a rim two orders of magnitude
            # thicker than the sheet, which is far outside the long-wave model's
            # validity. Raising the iteration cap and using the globally
            # convergent line search buys range; it does not make the model
            # valid there, so a run that stops early is reporting a real limit
            # rather than only a solver setting.
            extra = {}
            if max_newton is not None:
                extra["max_newton_iterations"] = max_newton
            if gcn:
                extra["globally_convergent_newton"] = True
            p.run(t, startstep=start, maxstep=dt, temporal_error=temporal_error,
                  outstep=False, out_initially=False, **extra)
            p.output()
            now = float(p.get_current_time(as_float=True))
            u = p.tip_velocity()
            dh, du = p.far_field_drift()
            history.add_row(now, now / p.tau_vis, u, u / V_TC,
                            p.tip_thickness(), dh, du)
            worst_drift = max(worst_drift, abs(dh), abs(du))
            prev = t

        meta["ustar_final"] = p.tip_velocity() / V_TC
        meta["h_tip_final"] = p.tip_thickness()
        meta["worst_far_field_drift"] = worst_drift
        # Not fatal: the run is still the best available record and the drift is
        # in the output for anyone re-reading it. But an answer computed on a
        # domain the disturbance already reached is not a 1D Trouton answer, so
        # say so loudly rather than filing it silently alongside clean cases.
        meta["domain_ok"] = bool(worst_drift <= drift_tol)
        if not meta["domain_ok"]:
            print(f"[oh={oh_sb:g}] WARNING: far-field drift {worst_drift:.3e} "
                  f"exceeds {drift_tol:g}; the disturbance reached the clamped "
                  f"end and u* is contaminated. Re-run with a larger --mass.",
                  flush=True)
        (outdir / "case1d.json").write_text(json.dumps(meta, indent=2))
        print(f"[oh={oh_sb:g}] done: u*({tstar_end:g}) = {meta['ustar_final']:.10f} "
              f"drift={worst_drift:.2e} ok={meta['domain_ok']}", flush=True)
        return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oh", type=float, required=True, help="Oh_SB (Savva-Bush)")
    ap.add_argument("--tstar-end", type=float, required=True)
    ap.add_argument("--nsteps", type=int, required=True,
                    help="fixed dt = t_end/nsteps; run twice with 2x for Richardson")
    ap.add_argument("--n-outputs", type=int, default=300)
    ap.add_argument("--temporal-error", type=float, default=1e-3,
                    help="recorded for provenance; the estimator never binds")
    ap.add_argument("--mass", type=float, default=None, help="default 2.5*V_TC*t_end")
    ap.add_argument("--nelem", type=int, default=None, help="default holds tip cell fixed")
    ap.add_argument("--drift-tol", type=float, default=1e-4)
    ap.add_argument("--max-newton", type=int, default=None,
                    help="raise the Newton iteration cap (default 10) for the "
                         "late-time large-rim regime")
    ap.add_argument("--gcn", action="store_true",
                    help="globally convergent (line-search) Newton")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run_case(args.oh, args.tstar_end, args.nsteps, args.n_outputs,
             args.temporal_error, args.out.resolve(), args.mass, args.nelem,
             args.drift_tol, args.max_newton, args.gcn)


if __name__ == "__main__":
    main()
