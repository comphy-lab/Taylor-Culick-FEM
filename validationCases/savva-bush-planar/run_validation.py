"""Run the planar Newtonian retraction sweep that Savva & Bush figures 5 and 6 test.

One case per Ohnesorge number, each marched to ``t* = t/tau_vis = 30`` so it
spans both published figures (figure 5 covers ``t* <= 1``, figure 6 ``t* <= 30``).
The recovered control velocity ``U(t)`` is written as ``control.txt`` per case
and compared against the digitised curves by ``compare_savva_bush.py``.

Conventions, which differ from ours and are the usual source of error:

    Oh       = mu / sqrt(2 H rho gamma)         (the project's only Oh; the
                                                 coded mu omits the sqrt(2))
    u_c      = sqrt(2 gamma / (rho H))         (identical to our V_TC = sqrt(2))
    tau_vis  = mu H / (2 gamma)                (the *viscous* clock)

with ``H`` the FULL sheet thickness. In our units (rho = gamma = H = 1) this
gives ``mu = sqrt(2) Oh`` and ``tau_vis = mu/2``.

DOMAIN LENGTH. The co-moving window must contain the region the tip disturbs.
Deka & Pierson (PRF 5, 093603, 2020) state it directly -- "the region of
influence of the tip motion is proportional to Oh" -- and their Eq. (6) carries
the inertial term with prefactor ``(L0/h0)^2 / Oh^2``, so the sheet behaves as
pseudo-infinite (and reaches the Taylor-Culick speed) when

    L / h0  >>  Oh .

The dependence is LINEAR in Oh, not quadratic: the square in that prefactor is
the square of the ratio, not a different scaling. ``window_length`` below
applies a safety factor to that requirement and is reported per case so the
assumption is visible rather than implicit.
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

from taylor_culick_fem.planar_comoving import PlanarTaylorCulickProblem
from taylor_culick_fem.run_contract import add_contract_options, run_case_runner, write_json_atomic
from taylor_culick_fem.savva_bush import V_TC, tau_vis, viscosity, window_length

# Ohnesorge numbers in the SAVVA & BUSH convention, matching their figures.
# Figure 5 shows Oh = 0.5, 1, 5, >100; figure 6 shows Oh = 0.5, 1, 10, 100/1000.
# The very large values are omitted: they are indistinguishable from each other
# in the source figures and cost the most to run.
DEFAULT_OH_SB = [0.5, 1.0, 5.0, 10.0]

TSTAR_END = 30.0  # spans both figures
def run_case(oh: float, outdir: Path, resolution: float, tstar_end: float,
             length: float | None = None, temporal_error: float = 1e-3,
             restart_from: Path | None = None,
             checkpoint_every: int = 0, n_outputs: int = 400) -> dict:
    mu = viscosity(oh)
    tau = tau_vis(mu)
    t_end = tstar_end * tau
    L = length if length is not None else window_length(mu, t_end)

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
        "domain_length": L,
        "L_over_Oh": L / mu,
        "L_over_diffusive": L / math.sqrt(4.0 * mu * t_end),
        "resolution": resolution,
        "temporal_error": temporal_error,
        "V_TC": V_TC,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(outdir / "case.json", meta, sort_keys=False, allow_nan=True)

    print(
        f"[Oh={oh}] mu={mu:.5f} tau_vis={tau:.5f} t_end={t_end:.3f} "
        f"L={L:.1f} (L/Oh={L / mu:.1f})",
        flush=True,
    )

    os.chdir(outdir)
    with PlanarTaylorCulickProblem() as problem:
        # The runner's options are not pyoomph's: --out would abbreviate its --outdir.
        problem.ignore_command_line = True
        problem.mu = mu
        problem.domain_length = L
        problem.resolution = resolution
        if restart_from is not None:
            # pyoomph's runmode defaults to "delete", and in that mode
            # initialise() empties EVERY subdirectory of the output directory
            # before anything else runs. That destroys _gmsh, whose
            # REMESH_*.msh the state dump reloads by name, and liquid/, which
            # holds the entire field output -- and it happens before the
            # load_state below, so the default turns a restart into silent data
            # loss followed by a missing-file crash. "overwrite" writes into the
            # existing tree without deleting it.
            #
            # There is no public setter. parse_cmd_line(), which initialise()
            # calls, only assigns _runmode when a --runmode flag is present, so
            # a value set here survives.
            problem._runmode = "overwrite"
        problem.initialise()

        t_start = 0.0
        if restart_from is not None:
            # Continue an existing run rather than repeating the first t_start
            # units. The geometric output schedule spends most of its steps
            # early, so extending a finished run from its last state costs a
            # fraction of redoing it, and the state dump carries the mesh and
            # the rim it grew -- neither of which the initial condition knows.
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
            "control.txt", header=["t", "tstar", "U", "U_over_VTC"]
        )
        # GEOMETRIC output spacing, not uniform. The interesting structure is at
        # early time -- u* rises through most of its range in the first few per
        # cent of the run -- and in the inertial regime tau_vis is so small that
        # uniform sampling in t gives a step of several units in t*, which
        # renders the early curve as straight segments. Geometric spacing puts
        # the samples where the curve actually bends.
        # On a restart the interesting structure is no longer at t = 0, so the
        # schedule spans the REMAINING interval. Anchoring it at t_end * 1e-4
        # would put most of the samples behind the restart point and then jump
        # the solver forward in one enormous step.
        if t_start > 0.0:
            # The output cadence IS the timestep: maxstep below is set to the
            # output interval and the temporal-error controller never binds (it
            # reports ~1e-12 against a 1e-3 tolerance), so dt sits on that cap
            # every step. Keeping n_outputs at its cold-start value here would
            # pack the same number of samples into the shorter remaining span
            # and silently do 5-10x more solver work than the original schedule
            # -- which is exactly what happened on the 2026-08-08 restart.
            # Warn loudly rather than silently rescaling: finer is slower AND
            # more accurate, so it is the operator's call, not a default.
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
            # pyoomph dumps a full restart state at EVERY internal step by
            # default. At 300k dof that is ~15 MiB a step and it dominated the
            # campaign's storage. Restartability is worth keeping, so states are
            # written on a schedule instead of being switched off entirely.
            problem.write_states = False
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)

        for k, t in enumerate(out_times):
            # This driver owns the output schedule.  Suppress pyoomph's
            # temporal-error default, which otherwise fires every Output object
            # on every internal timestep, then emit exactly one field state for
            # this scheduled target.
            problem.run(
                t,
                startstep=max((t - prev) * 1e-3, 1e-9),
                maxstep=max(t - prev, 1e-9),
                temporal_error=temporal_error,
                outstep=False,
                out_initially=False,
            )
            problem.output()
            u = problem.control_velocity()
            now = float(problem.get_current_time(as_float=True))
            history.add_row(now, now / tau, u, u / V_TC)
            prev = t
            if checkpoint_every and (k + 1) % checkpoint_every == 0:
                problem.save_state(str(ckpt_dir / f"ckpt_{k + 1:04d}.dump"), quiet=True)
        meta["U_final"] = problem.control_velocity()
        meta["ustar_final"] = meta["U_final"] / V_TC

    write_json_atomic(outdir / "case.json", meta, sort_keys=False, allow_nan=True)
    return meta


PROBLEM = "planar_comoving"


def build_parser() -> argparse.ArgumentParser:
    """The case, the run-contract options and this driver's numerical options."""
    ap = add_contract_options(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--resolution", type=float, default=0.06)
    ap.add_argument("--length", type=float, default=None,
                    help="override the Oh-derived co-moving window length")
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--tstar-end", type=float, default=TSTAR_END)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write a restart state every N outputs instead of every "
                         "internal step. 0 keeps pyoomph's default. The default "
                         "wrote ~2400 dumps per case and 86%% of a 507 GiB tree; "
                         "10 checkpoints give the same restart capability.")
    ap.add_argument("--n-outputs", type=int, default=400,
                    help="number of geometrically spaced output points. This is "
                         "NOT just an output setting: maxstep is clamped to the "
                         "output interval and the temporal-error controller never "
                         "binds, so this value sets dt and therefore both the cost "
                         "and the temporal accuracy of the run.")
    return ap


def option_problems(args: argparse.Namespace) -> list[str]:
    """Every invalid numerical option; empty when all are valid."""
    checks = [
        (args.resolution <= 0, "--resolution must be positive"),
        (args.tstar_end <= 0, "--tstar-end must be positive"),
        (args.temporal_error <= 0, "--temporal-error must be positive"),
        (args.checkpoint_every < 0, "--checkpoint-every must be non-negative"),
        (args.n_outputs < 2, "--n-outputs must be at least 2"),
        (args.length is not None and args.length <= 0, "--length must be positive"),
    ]
    return [message for failed, message in checks if failed]


def solve(args: argparse.Namespace, case: dict, state: Path | None) -> dict:
    """Run the case's ``Oh`` into ``--out``; ``state`` continues an earlier run of the same case.

    The automatic window length grows with the stop time, so a continuation reuses the
    length its parent resolved: the checkpoint's mesh has that length.
    """
    length = args.length
    if state is not None and length is None:
        length = json.loads((args.parent_run / "case.json").read_text())["domain_length"]
    return run_case(case["physics"]["Oh"], args.out.resolve(), args.resolution, args.tstar_end,
                    length, args.temporal_error, state, args.checkpoint_every, args.n_outputs)


def main(argv: list[str] | None = None) -> int:
    return run_case_runner(argv, runner=Path(__file__), root=_ROOT, parser=build_parser(),
                           problem=PROBLEM, option_problems=option_problems, solve=solve)


if __name__ == "__main__":
    raise SystemExit(main())
