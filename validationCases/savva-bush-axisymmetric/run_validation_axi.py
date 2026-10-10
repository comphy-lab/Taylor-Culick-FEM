"""Run the axisymmetric lab-frame Newtonian hole-opening sweep.

Mirrors ``run_validation.py`` (the planar co-moving driver) as closely as the
geometry allows: same Savva-Bush Ohnesorge convention, same geometric output
schedule (which also sets ``maxstep``, so it controls both accuracy and cost),
same restart pattern. Two things are new because the geometry is new:

  * ``--r0`` / ``--rout``: the graded annulus needs an initial hole radius and
    a fixed outer (lab-frame) wall, neither of which the co-moving planar case
    has (its window is centred on a tip pinned at ``x = 0``).
  * The control file tracks ``R(t)`` (an observable, not a driven global dof)
    and its finite-difference derivative, rather than the co-moving frame
    velocity ``U`` a global dof carries directly.

CONTROL FILE LAYOUT (this is a NEW file; nothing else in the repository reads
it positionally, unlike the planar ``control.txt``, so it documents itself
here rather than assuming a shared convention):

    t        dimensional time
    tstar    t / tau_vis = 2*t/mu  (Savva-Bush's viscous clock)
    R        hole edge radius, problem.hole_radius() (min mesh_x on "interface")
    Rdot     dR/dt, backward finite difference between this and the previous
             output row (NOT a solved/smoothed quantity -- see the docstring
             of run_case() for why this, not a central difference, is used)
    ustar    Rdot / V_TC, the axisymmetric analogue of planar's U_over_VTC
    volume   the exact-conservation observable (IntegralObservables(volume=1)),
             a correctness diagnostic to carry alongside the physics, not a
             physical output in its own right

LARGE ``--r0`` (the planar-reduction cases, R0 ~ 100): this driver passes no
``newton_solver_tolerance``, and must not start doing so. The tolerance is
set problem-side, per timestep, in
``AxisymmetricTaylorCulickProblem.actions_before_newton_solve``, because the
residual this problem assembles carries ``2*pi*r`` from the measure and
``1/dt`` from the mass rows, so the floor a direct solve can reach grows with
both. Before that existed, ``--r0 100`` died inside the FIRST output interval
with Newton residuals oscillating at O(0.1) and the adaptive controller
halving ``dt`` to underflow -- which reads like a startstep or formulation
pathology and is neither: the first attempt converges quadratically to a
~3.5e-8 floor and is rejected only against an absolute 1e-8 bound. The full
evidence (an R0 = 1/10/30/100 scan on one translated mesh, the floor-vs-dt
law, and the FD-Jacobian, uniform-mesh, iteration-budget, line-search, BDF1
and startstep controls that rule out the alternatives) is in that module's
docstring.

One consequence is specific to the loop below: because each output row is its
own ``run()`` call, its last sub-step is whatever remains to land exactly on
``t``, and that leftover is occasionally a tiny sliver whose Newton floor is
correspondingly high. That is a property of this schedule, not of the
physics, and it is why the tolerance had to be made ``dt``-aware rather than
merely R-scaled: at coarse resolution, ``--r0 100`` cleared the first output
row under a fixed R-scaled tolerance and then died on a 3.4e-7 sliver in the
second.

Conventions carried over unchanged from run_validation.py / savva_bush.py:

    Oh       = mu / sqrt(2 H rho gamma)         (the project's only Oh; the
                                                 coded mu omits the sqrt(2))
    u_c      = sqrt(2 gamma / (rho H))          (identical to our V_TC = sqrt(2))
    tau_vis  = mu H / (2 gamma)                 (the *viscous* clock)

with ``H`` the FULL sheet thickness, so ``mu = sqrt(2) Oh`` and
``tau_vis = mu/2`` in our units (rho = gamma = H = 1).
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

from pyoomph.output.states import copy_state_file

from taylor_culick_fem.axisymmetric_lab import AxisymmetricTaylorCulickProblem
from taylor_culick_fem.run_contract import add_contract_options, run_case_runner, write_json_atomic
from taylor_culick_fem.savva_bush import V_TC, tau_vis, viscosity, window_length

# Ohnesorge numbers in the Savva-Bush convention. Same default list as
# run_validation.py, for the same reason (figures 5 and 6 coverage without the
# indistinguishable-in-the-source-figure large-Oh cases).
DEFAULT_OH_SB = [0.5, 1.0, 5.0, 10.0]

TSTAR_END = 30.0
R0_DEFAULT = 1.0


def auto_rout(r0: float, mu: float, t_end: float) -> float:
    """Outer (lab-frame) wall radius: full tip travel plus the planar footprint margin.

    The lab-frame far boundary is FIXED, unlike the planar co-moving window
    (which only ever needs to span the disturbed region around a tip held at
    x=0). So R_out here must cover two additive things:

      1. The full distance the tip can travel: at most V_TC * t_end (the tip
         never moves faster than the Taylor-Culick speed it is approaching),
         starting from R0.
      2. Beyond the FARTHEST point the tip ever reaches, the same disturbance
         footprint the planar window needs -- run_validation.window_length(),
         which is the larger of the steady Stokes influence length (~Oh) and
         the transient diffusive spread sqrt(4*mu*t_end) -- so the far
         DirichletBC still sees only undisturbed film, never the retracting
         rim's near field.

    This is deliberately generous (the tip's average speed over the run is
    less than V_TC, since it starts from rest), trading a somewhat larger
    graded-far-film region for never having to re-derive this per case.
    """
    return r0 + V_TC * t_end + window_length(mu, t_end)


def run_case(oh: float, outdir: Path, resolution: float, tstar_end: float,
             r0: float = R0_DEFAULT, rout: float | None = None,
             temporal_error: float = 1e-3, restart_from: Path | None = None,
             checkpoint_every: int = 0, n_outputs: int = 400) -> dict:
    mu = viscosity(oh)
    tau = tau_vis(mu)
    t_end = tstar_end * tau
    R_out = rout if rout is not None else auto_rout(r0, mu, t_end)

    meta = {
        # `Oh` is canonical (Savva & Bush convention). `Oh_savva_bush` and
        # `Oh_ours` are retained verbatim so archived post-processing and the
        # SHA-256-manifested run roots written before 2026-09-13 keep working.
        "Oh": oh,
        "Oh_savva_bush": oh,
        "Oh_ours": mu,  # mu/sqrt(rho*gamma*h0) with rho=gamma=h0=1
        "mu_coded": mu,
        "tau_vis": tau,
        "tstar_end": tstar_end,
        "t_end": t_end,
        "R0": r0,
        "R_out": R_out,
        "resolution": resolution,
        "temporal_error": temporal_error,
        "V_TC": V_TC,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(outdir / "case.json", meta, sort_keys=False, allow_nan=True)

    print(
        f"[Oh={oh}] mu={mu:.5f} tau_vis={tau:.5f} t_end={t_end:.3f} "
        f"R0={r0:.2f} R_out={R_out:.1f}",
        flush=True,
    )

    os.chdir(outdir)
    with AxisymmetricTaylorCulickProblem() as problem:
        # The runner's options are not pyoomph's: --out would abbreviate its --outdir.
        problem.ignore_command_line = True
        problem.mu = mu
        problem.R0 = r0
        problem.R_out = R_out
        problem.resolution = resolution
        if restart_from is not None:
            # Same trap as run_validation.py: pyoomph's runmode defaults to
            # "delete", which empties every subdirectory of the output
            # directory before initialise() does anything else -- destroying
            # _gmsh (whose REMESH_*.msh the state dump reloads by name) and
            # liquid/ (the field output) before the load_state below runs.
            # "overwrite" writes into the existing tree without deleting it.
            problem._runmode = "overwrite"
        problem.initialise()

        t_start = 0.0
        if restart_from is not None:
            # The state stays in its parent run; a copy with its mesh files is loaded here.
            relocated = outdir / "restart-input" / restart_from.name
            relocated.parent.mkdir(exist_ok=True)
            copy_state_file(str(restart_from), str(relocated))
            problem.load_state(str(relocated), ignore_outstep=True)
            t_start = float(problem.get_current_time(as_float=True))
            print(f"  restarted from {restart_from.name} at t = {t_start:.4f} "
                  f"(t* = {t_start / tau:.2f})", flush=True)
            if t_start >= t_end:
                raise SystemExit(f"restart state is already at t={t_start:.3f} >= "
                                 f"target t_end={t_end:.3f}; nothing to do")

        history = problem.create_text_file_output(
            "control.txt", header=["t", "tstar", "R", "Rdot", "ustar", "volume"]
        )
        # Same geometric output schedule as run_validation.py, for the same
        # reason: the interesting structure (here, the near-critical-radius
        # transient near t=0 -- see verificationCases/axisymmetric-construction/axi_smoketests.py's s4lite
        # docstring for what that looks like at R0=1) is at early time, and
        # maxstep is clamped to the output interval (the temporal-error
        # controller does not bind in practice), so this schedule sets dt.
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
            # restart state at every internal step by default. Write states
            # on a schedule instead.
            problem.write_states = False
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)

        R_prev = problem.hole_radius()
        t_prev_row = t_start
        for k, t in enumerate(out_times):
            # Deviation from run_validation.py's startstep=max((t-prev)*1e-3, 1e-9):
            # that formula gives a starting step near 1e-9 for this geometry's
            # first (t_end*1e-4-scale) output, and BDF2's Newton iteration does
            # not converge there -- it oscillates between two residual values
            # for all 10 allowed iterations, the timestep controller keeps
            # halving in response, and the run aborts once dt underflows below
            # 1e-12 ("Tried to reduce dt to ... which is less than the minimum
            # dt"). Confirmed empirically (not a one-off): reproduces on a
            # bare problem.run() call at the same t0, resolution and R_out.
            # A ten-times-less-aggressive starting step (1e-2 of the interval,
            # floored at 1e-6 rather than 1e-9) converges immediately at the
            # same first output time -- the planar driver's floor was simply
            # too aggressive for this ALE geometry's extra mesh-position dofs.
            # That is an R0 = 1 statement. The superficially identical abort at
            # large --r0 is NOT a startstep problem (a larger startstep only
            # buys a factor 1.7 of margin there); it is the absolute-Newton-
            # tolerance scaling handled in axisymmetric_lab.py -- see this
            # module's docstring and that module's TRAP section.
            problem.run(t, startstep=max((t - prev) * 1e-2, 1e-6),
                        maxstep=max(t - prev, 1e-9), temporal_error=temporal_error)
            now = float(problem.get_current_time(as_float=True))
            R = problem.hole_radius()
            dt_row = now - t_prev_row
            # Backward finite difference between consecutive OUTPUT rows, not
            # a solver-internal quantity: R(t) is a pure geometric observable
            # (the axisymmetric design note (axi-design.md)), with no ODE carrying its rate directly
            # (contrast planar's U, a genuine global dof), so this is the
            # only Rdot available without differentiating the mesh state.
            Rdot = (R - R_prev) / dt_row if dt_row > 0 else float("nan")
            vol = problem.volume()
            history.add_row(now, now / tau, R, Rdot, Rdot / V_TC, vol)
            prev = t
            R_prev = R
            t_prev_row = now
            if checkpoint_every and (k + 1) % checkpoint_every == 0:
                problem.save_state(str(ckpt_dir / f"ckpt_{k + 1:04d}.dump"), quiet=True)
        meta["R_final"] = R_prev
        meta["Rdot_final"] = Rdot
        meta["ustar_final"] = Rdot / V_TC

    write_json_atomic(outdir / "case.json", meta, sort_keys=False, allow_nan=True)
    return meta


PROBLEM = "axisymmetric_lab"


def build_parser() -> argparse.ArgumentParser:
    """The case, the run-contract options and this driver's numerical options."""
    ap = add_contract_options(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--resolution", type=float, default=0.06)
    ap.add_argument("--r0", type=float, default=R0_DEFAULT,
                    help="initial hole radius, in h0")
    ap.add_argument("--rout", type=float, default=None,
                    help="override the auto-sized outer (lab-frame) wall radius")
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--tstar-end", type=float, default=TSTAR_END)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write a restart state every N outputs instead of every "
                         "internal step. 0 keeps pyoomph's default.")
    ap.add_argument("--n-outputs", type=int, default=400,
                    help="number of geometrically spaced output points. This is "
                         "NOT just an output setting: maxstep is clamped to the "
                         "output interval, so this value sets dt and therefore "
                         "both the cost and the temporal accuracy of the run.")
    return ap


def option_problems(args: argparse.Namespace) -> list[str]:
    """Every invalid numerical option; empty when all are valid."""
    checks = [
        (args.resolution <= 0, "--resolution must be positive"),
        (args.tstar_end <= 0, "--tstar-end must be positive"),
        (args.temporal_error <= 0, "--temporal-error must be positive"),
        (args.checkpoint_every < 0, "--checkpoint-every must be non-negative"),
        (args.n_outputs < 2, "--n-outputs must be at least 2"),
        (args.r0 <= 0, "--r0 must be positive"),
        (args.rout is not None and args.rout <= args.r0, "--rout must exceed --r0"),
    ]
    return [message for failed, message in checks if failed]


def solve(args: argparse.Namespace, case: dict, state: Path | None) -> dict:
    """Run the case's ``Oh`` into ``--out``; ``state`` continues an earlier run of the same case.

    The automatic outer wall grows with the stop time, so a continuation reuses the radius
    its parent resolved: the checkpoint's mesh has that wall.
    """
    rout = args.rout
    if state is not None and rout is None:
        rout = json.loads((args.parent_run / "case.json").read_text())["R_out"]
    return run_case(case["physics"]["Oh"], args.out.resolve(), args.resolution, args.tstar_end,
                    args.r0, rout, args.temporal_error, state, args.checkpoint_every,
                    args.n_outputs)


def main(argv: list[str] | None = None) -> int:
    return run_case_runner(argv, runner=Path(__file__), root=_ROOT, parser=build_parser(),
                           problem=PROBLEM, option_problems=option_problems, solve=solve)


if __name__ == "__main__":
    raise SystemExit(main())
