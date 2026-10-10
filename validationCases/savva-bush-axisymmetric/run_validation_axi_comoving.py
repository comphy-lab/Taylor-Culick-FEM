"""Run the axisymmetric co-moving-window Newtonian hole-opening sweep.

Mirrors ``run_validation_axi.py`` (the lab-frame annulus driver) option for
option, so the two can be compared by running the same command line with one
word changed. The differences are exactly the differences between the two
formulations:

  * ``--window-length`` (L) replaces ``--rout``. The lab driver must SIZE its
    outer wall to contain the whole trajectory (``auto_rout`` = R0 + V_TC*t_end
    + disturbance footprint), so its cost grows with ``--tstar-end``. Here the
    window is dragged along by the rim, so L is a physical resolution choice
    (how much undisturbed film ahead of the rim is kept) and ``t_end`` is
    unbounded. L = 40 is the planar-validated window length scaled up from the
    planar campaign's 12; it is not auto-sized from ``t_end``, and must not be.
  * The control file carries the retraction speed twice, once as a solved
    global dof and once as the lab driver's finite difference (below).

CONTROL FILE LAYOUT (columns, in order):

    t            dimensional time
    tstar        t / tau_vis = 2*t/mu  (Savva-Bush's viscous clock)
    R            hole edge radius = the global dof X (equivalently, and to the
                 accuracy of the tip pin, min mesh_x on "interface")
    Rdot         dR/dt by BACKWARD FINITE DIFFERENCE between this and the
                 previous output row, identical in construction to the lab
                 driver's column of the same name, so the two files can be
                 compared column-for-column without worrying about estimator
                 differences
    ustar        Rdot / V_TC
    volume       IntegralObservables(volume=1) over the window
    volume_expected
                 the exact value that volume MUST take given how far the
                 window has swept: V(0) + (pi/2)*[(X+L)^2 - (X0+L)^2]. Volume
                 is NOT conserved in this formulation -- the "far" edge is an
                 inflow of undisturbed, at-rest film -- so this pair replaces
                 the lab case's conservation check and is just as sharp. See
                 ``AxisymmetricComovingProblem.expected_volume``.

``case.json`` additionally records ``U_final`` (the solved global dof dX/dt)
and the worst observed pin residual and volume error, since those are the
correctness statements a reader of the table needs and none of them belongs in
a column that is meant to line up with the lab driver's.

Conventions carried over unchanged from run_validation.py / savva_bush.py:

    Oh       = mu / sqrt(2 H rho gamma)         (the project's only Oh; the
                                                 coded mu omits the sqrt(2))
    u_c      = sqrt(2 gamma / (rho H))          (identical to our V_TC = sqrt(2))
    tau_vis  = mu H / (2 gamma)                 (the *viscous* clock)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from taylor_culick_fem.axisymmetric_comoving import AxisymmetricComovingProblem
from taylor_culick_fem.savva_bush import V_TC, tau_vis, viscosity

#: Same default list, and for the same reason, as run_validation_axi.py.
DEFAULT_OH_SB = [0.5, 1.0, 5.0, 10.0]

TSTAR_END = 30.0
X0_DEFAULT = 1.0
#: Fixed window length L, in h0. Scaled up from the planar campaign's validated
#: 12 to leave room for the axisymmetric rim's slower far-field decay; a
#: physical choice, deliberately NOT a function of t_end (that is the whole
#: point of a window).
WINDOW_LENGTH_DEFAULT = 40.0


def run_case(oh: float, outdir: Path, resolution: float, tstar_end: float,
             x0: float = X0_DEFAULT, window_length: float = WINDOW_LENGTH_DEFAULT,
             temporal_error: float = 1e-3, restart_from: Path | None = None,
             checkpoint_every: int = 0, n_outputs: int = 400,
             max_newton: int | None = None) -> dict:
    mu = viscosity(oh)
    tau = tau_vis(mu)
    t_end = tstar_end * tau

    meta = {
        # `Oh` is canonical (Savva & Bush convention). `Oh_savva_bush` and
        # `Oh_ours` are retained verbatim so archived post-processing and the
        # SHA-256-manifested run roots written before 2026-09-13 keep working.
        "Oh": oh,
        "Oh_savva_bush": oh,
        "Oh_ours": mu,
        "mu_coded": mu,
        "tau_vis": tau,
        "tstar_end": tstar_end,
        "t_end": t_end,
        "X0": x0,
        "window_length": window_length,
        "resolution": resolution,
        "temporal_error": temporal_error,
        "V_TC": V_TC,
        "max_newton_iterations": max_newton,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "case.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[Oh={oh}] mu={mu:.5f} tau_vis={tau:.5f} t_end={t_end:.3f} "
        f"X0={x0:.2f} L={window_length:.1f}",
        flush=True,
    )

    os.chdir(outdir)
    with AxisymmetricComovingProblem() as problem:
        problem.mu = mu
        problem.X0 = x0
        problem.window_length = window_length
        problem.resolution = resolution
        if max_newton is not None:
            problem.max_newton_iterations = max_newton
        if restart_from is not None:
            # Same trap as run_validation.py: pyoomph's runmode defaults to
            # "delete", which empties every subdirectory of the output
            # directory before initialise() runs, destroying _gmsh and liquid/.
            problem._runmode = "overwrite"
        problem.initialise()

        t_start = 0.0
        if restart_from is not None:
            problem.load_state(str(restart_from), ignore_outstep=True)
            t_start = float(problem.get_current_time(as_float=True))
            print(f"  restarted from {restart_from.name} at t = {t_start:.4f} "
                  f"(t* = {t_start / tau:.2f})", flush=True)
            if t_start >= t_end:
                raise SystemExit(f"restart state is already at t={t_start:.3f} >= "
                                 f"target t_end={t_end:.3f}; nothing to do")

        history = problem.create_text_file_output(
            "control.txt",
            header=["t", "tstar", "R", "Rdot", "ustar", "volume", "volume_expected"],
        )
        if t_start > 0.0:
            fresh_ratio = (t_end / (t_end * 1e-4)) ** (1.0 / (n_outputs - 1))
            this_ratio = (t_end / t_start) ** (1.0 / (n_outputs - 1))
            matched = math.ceil(1 + (n_outputs - 1) * math.log(t_end / t_start)
                                / math.log(1e4))
            if fresh_ratio - 1.0 > 2.0 * (this_ratio - 1.0):
                print(f"  WARNING: {n_outputs} outputs over the remaining span is "
                      f"{(fresh_ratio - 1.0) / (this_ratio - 1.0):.1f}x finer than the "
                      f"cold-start cadence, so dt is that much smaller and the run "
                      f"that much slower. --n-outputs {matched} would match it.",
                      flush=True)
            out_times = np.geomspace(t_start * 1.001, t_end, n_outputs)
        else:
            out_times = np.geomspace(t_end * 1e-4, t_end, n_outputs)
        prev = t_start
        if checkpoint_every:
            # Same storage trap as run_validation.py: pyoomph dumps a full
            # restart state at every internal step by default.
            problem.write_states = False
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)

        # Reference state for the modified volume check. On a restart, both the
        # volume and the radius are taken from the restarted state, so the
        # identity is anchored where the run actually begins.
        volume0 = problem.volume()
        x_ref = problem.window_position()

        R_prev = problem.window_position()
        t_prev_row = t_start
        Rdot = float("nan")
        worst_pin = 0.0
        worst_vol = 0.0
        for k, t in enumerate(out_times):
            # Same startstep/maxstep reasoning as run_validation_axi.py: the
            # planar driver's 1e-9 floor does not converge for this ALE
            # geometry's mesh-position dofs at the first (t_end*1e-4) output.
            run_kw = dict(
                startstep=max((t - prev) * 1e-2, 1e-6),
                maxstep=max(t - prev, 1e-9),
                temporal_error=temporal_error,
            )
            if max_newton is not None:
                run_kw["max_newton_iterations"] = max_newton
            problem.run(t, **run_kw)
            now = float(problem.get_current_time(as_float=True))
            R = problem.window_position()
            dt_row = now - t_prev_row
            Rdot = (R - R_prev) / dt_row if dt_row > 0 else float("nan")
            vol = problem.volume()
            vol_exp = problem.expected_volume(volume0, x_ref)
            history.add_row(now, now / tau, R, Rdot, Rdot / V_TC, vol, vol_exp)
            worst_pin = max(worst_pin, abs(R - problem.hole_radius()))
            worst_vol = max(worst_vol, abs(vol - vol_exp) / abs(vol_exp))
            prev = t
            R_prev = R
            t_prev_row = now
            if checkpoint_every and (k + 1) % checkpoint_every == 0:
                problem.save_state(str(ckpt_dir / f"ckpt_{k + 1:04d}.dump"), quiet=True)
        meta["R_final"] = R_prev
        meta["Rdot_final"] = Rdot
        meta["ustar_final"] = Rdot / V_TC
        # The solved rate dof, alongside the differenced one: their gap is a
        # free measure of how well the output cadence resolves dR/dt.
        meta["U_final"] = problem.retraction_speed()
        meta["ustar_final_from_dof"] = problem.retraction_speed() / V_TC
        meta["worst_pin_residual"] = worst_pin
        meta["worst_volume_rel_error"] = worst_vol

    (outdir / "case.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oh", type=float, nargs="*", default=None,
                    help="Ohnesorge numbers, Oh = mu/sqrt(2 h0 rho sigma). The coded viscosity is mu = sqrt(2) Oh.")
    # No committed default; see the note in run_validation_axi.py.
    ap.add_argument("--results", type=Path, required=True,
                    help="directory to write case output into, on the compute "
                         "host's run root (not inside this repository)")
    ap.add_argument("--resolution", type=float, default=0.06)
    ap.add_argument("--r0", type=float, default=X0_DEFAULT, dest="x0",
                    help="initial hole radius X0, in h0 (spelled --r0 to match "
                         "run_validation_axi.py)")
    ap.add_argument("--window-length", type=float, default=WINDOW_LENGTH_DEFAULT,
                    help="fixed radial extent L of the co-moving window, in h0. "
                         "Unlike the lab driver's --rout this is NOT sized from "
                         "t_end; the window travels instead of growing.")
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the case directory, for convergence studies")
    ap.add_argument("--tstar-end", type=float, default=TSTAR_END)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write a restart state every N outputs instead of every "
                         "internal step. 0 keeps pyoomph's default.")
    ap.add_argument("--n-outputs", type=int, default=400,
                    help="number of geometrically spaced output points. This is "
                         "NOT just an output setting: maxstep is clamped to the "
                         "output interval, so this value sets dt and therefore "
                         "both the cost and the temporal accuracy of the run.")
    ap.add_argument("--restart-from", type=Path, default=None,
                    help="pyoomph state dump to continue from")
    ap.add_argument("--max-newton", type=int, default=None,
                    help="Newton iteration cap (pyoomph default is 10). "
                         "40 matches the lab/1D drivers and avoids dt-halving "
                         "when the last residual is just above 1e-8.")
    args = ap.parse_args()
    # See run_validation_axi.py: the schedule divides by (n_outputs - 1).
    if args.n_outputs < 2:
        raise SystemExit(f"--n-outputs must be at least 2, got {args.n_outputs}")
    if args.restart_from is not None:
        # Resolve before run_case chdirs into the case directory; a relative
        # path would pass this check and then fail inside load_state().
        args.restart_from = args.restart_from.resolve()
        if not args.restart_from.exists():
            raise SystemExit(f"restart state not found: {args.restart_from}")
        if args.oh and len(args.oh) > 1:
            raise SystemExit("--restart-from takes a single --oh")

    oh_list = args.oh if args.oh else DEFAULT_OH_SB
    root = args.results.resolve()
    for oh in oh_list:
        name = f"Oh_SB_{oh:g}" + (f"_{args.tag}" if args.tag else "")
        run_case(oh, root / name, args.resolution, args.tstar_end,
                 args.x0, args.window_length, args.temporal_error,
                 args.restart_from, args.checkpoint_every, args.n_outputs,
                 args.max_newton)


if __name__ == "__main__":
    main()
