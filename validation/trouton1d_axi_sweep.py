#!/usr/bin/env python3
"""Axisymmetric 1D hole opening: u*(t) and R(t) at arbitrary Oh, per case.

The axisymmetric counterpart of ``trouton1d_sweep.py``, driving
``simulationCases/trouton1d_axi.py``. Everything about the timestep and output
discipline is inherited deliberately; what changes is the domain sizing, which
does NOT transfer from planar, and the columns, which gain the rim radius and
the tip-thickness identity.

OUTPUT TABLE ``control1d_axi.txt``, tab separated, one row per output time:

    col 1  t          simulation time in code units (rho = sigma = h0 = 1)
    col 2  tstar      t / tau_vis, the viscous clock, tau_vis = mu/2
    col 3  u_tip      tip velocity u(m = 0)
    col 4  ustar      u_tip / V_TC, V_TC = sqrt(2)
    col 5  R          rim radius, the global unknown, R(0) = r0
    col 6  h_tipode   tip thickness from the closure identity (see below)
    col 7  drift_u    |u| at the far end m = M; the domain-adequacy check

``h_tipode`` is NOT the solver's tip thickness. It is the independent
integration of the identity the tip closure implies exactly,

    dh_tip/dt = 1/(2 mu) - h_tip u_tip / (2 R),

driven by the solver's own ``u_tip`` and ``R``. The planar limit ``R ->
infinity`` recovers ``h_tip = 1 + t*/4`` for every Oh, which is the planar
solver's ``h_tip_exact`` column; the late-time axisymmetric limit is
``h_tip -> t*/6``, azimuthal stretching thinning the rim to exactly two thirds
of the planar rate. Comparing this column against the solver's own first-cell
thickness is a closure check, done by ``trouton1d_axi_checks.py tipthickness``.
The integration is an implicit trapezoid, sub-stepped inside each output
interval because the output cadence is geometric and its late intervals are
long; the scheme is unconditionally stable in the linear-in-h identity.

THE TIMESTEP IS FIXED, AND THAT IS LOAD-BEARING. pyoomph's temporal error
estimator under-reports on this family of problems by about nine orders, so it
never binds and dt sits on ``maxstep`` on every step. Holding ``maxstep``
fixed for the whole run is what makes a convergence claim checkable at all:
halve ``--nsteps``, halve dt everywhere, and compare. A run that "looks
converged" because the estimator reports 1e-12 is telling you nothing.

``outstep=False`` and ``out_initially=False`` are equally load-bearing.
``Problem.run`` documents ``outstep`` as defaulting to True whenever
``temporal_error`` is passed, and True writes every Output object on every
INTERNAL timestep, not once per ``run()`` call -- at this driver's step counts
that is one profile file per step, projecting to hundreds of GB for a solve
whose useful output is one text file.

DOMAIN SIZING IS NOT THE PLANAR RULE. The planar driver's
``max(120, 2.5 V_TC t_end)`` is a LENGTH, and in a planar sheet of unit
thickness mass and length coincide. Here the domain is the per-radian mass
``M = int_R^{r_max} r h dr = (r_max^2 - R0^2)/2``, quadratic in the radial
extent, so the planar expression under-sizes by a factor of order ``r_max``.
The radial margin itself is sized as planar and then squared into a mass; see
``trouton1d_axi.auto_mass``. ``drift_u`` checks the result every output.

The element count then holds the tip cell fixed in RADIUS rather than in mass,
because that is what resolves the diffusive start-up layer: with the grading
``m = M xi^grade`` the tip cell is ``M/N^grade`` in mass, hence
``M/(N^grade R0)`` in radius near the tip, and the planar mesh study validated
a radial tip cell of ``120/400^2 = 7.5e-4``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulationCases"))
from trouton1d_axi import Trouton1DAxiProblem, auto_mass  # noqa: E402

V_TC = math.sqrt(2.0)

#: Radial size of the tip cell validated by the planar mesh study at
#: M = 120, N = 400, grade = 2.
TIP_CELL_R = 120.0 / 400.0 ** 2

COLUMNS = ["t", "tstar", "u_tip", "ustar", "R", "h_tipode", "drift_u"]


def auto_nelem(mass: float, r0: float, grade: float = 2.0) -> int:
    """Element count holding the tip cell at ``TIP_CELL_R`` in radius."""
    n = (mass / (r0 * TIP_CELL_R)) ** (1.0 / grade)
    return int(min(3000, max(400, round(n))))


def _advance_h_tip(h: float, t0: float, t1: float, mu: float,
                   u0: float, r0: float, u1: float, r1: float,
                   nsub: int = 32) -> float:
    """Integrate ``dh/dt = 1/(2 mu) - h u/(2 R)`` from t0 to t1.

    The identity is linear in h with a positive damping coefficient
    ``b = u/(2R)``, so the implicit trapezoid below is unconditionally stable
    and second order. ``u`` and ``R`` are known only at the endpoints, so they
    are interpolated linearly across the sub-steps; that interpolation, not the
    ODE integrator, is the accuracy limit of this column.
    """
    a = 1.0 / (2.0 * mu)
    dt = (t1 - t0) / nsub
    if dt <= 0.0:
        return h
    for k in range(nsub):
        s0, s1 = k / nsub, (k + 1) / nsub
        b0 = (u0 + (u1 - u0) * s0) / (2.0 * (r0 + (r1 - r0) * s0))
        b1 = (u0 + (u1 - u0) * s1) / (2.0 * (r0 + (r1 - r0) * s1))
        h = (h + 0.5 * dt * (2.0 * a - b0 * h)) / (1.0 + 0.5 * dt * b1)
    return h


def run_case(oh_sb: float, tstar_end: float, nsteps: int, n_outputs: int,
             outroot: Path, r0: float = 5.0, mass: float | None = None,
             nelem: int | None = None, grade: float = 2.0,
             temporal_error: float = 1e-3, drift_tol: float = 1e-4,
             max_newton: int | None = None, gcn: bool = False,
             tag: str | None = None,
             tip_azimuthal_correction: bool = False,
             probe: Callable[[Trouton1DAxiProblem, dict], None] | None = None
             ) -> dict:
    """Run one axisymmetric case and return its metadata plus the series.

    ``probe(problem, row)`` is called after every output row with the live
    problem, which is how the check suite reads per-cell quantities that no
    column carries. The returned dict holds ``"series"``, a dict of numpy
    arrays keyed by :data:`COLUMNS`; it is stripped before the metadata is
    written to ``case1d_axi.json``.
    """
    # ONE CASE PER DIRECTORY. pyoomph writes relative to the process working
    # directory in places, so the caller supplies a unique --out per case.
    outdir = outroot
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    series: dict[str, list[float]] = {c: [] for c in COLUMNS}

    with Trouton1DAxiProblem() as p:
        # PYOOMPH READS sys.argv, AND ARGPARSE ABBREVIATES. Problem.initialise
        # calls parse_known_args() on the real command line, which carries
        # --outdir; argparse's prefix matching means a caller's --out (or
        # --outd, --outdi) is accepted as that flag and OVERWRITES the output
        # directory set below, for every problem built in the process. A single
        # -case script never notices, because the two paths coincide; a driver
        # that runs several cases in one process silently writes them all into
        # one directory, and the second case's metadata write then lands beside
        # the first case's tables. Refuse the command line here: everything
        # this driver needs is set programmatically.
        p.ignore_command_line = True
        p.Oh_SB = oh_sb
        p.r0 = r0
        p.grade = grade
        p.tip_azimuthal_correction = tip_azimuthal_correction
        t_end = tstar_end * p.tau_vis
        p.mass = auto_mass(r0, t_end) if mass is None else mass
        p.nelem = auto_nelem(p.mass, r0, grade) if nelem is None else nelem
        p.set_output_directory(str(outdir))
        p.initialise()
        # pyoomph dumps a full restart state on EVERY internal step; at these
        # step counts that is tens of GB for a 1D solve that is cheap to redo.
        p.write_states = False

        dt = t_end / nsteps
        r_max = math.sqrt(r0 * r0 + 2.0 * p.mass)
        meta = {
            "tag": tag,
            "Oh_SB": oh_sb, "mu": p.mu, "tau_vis": p.tau_vis,
            "r0": r0, "r0_star": p.r0_star, "r_max": r_max,
            "tstar_end": tstar_end, "t_end": t_end,
            "mass": p.mass, "nelem": p.nelem, "grade": p.grade,
            "tip_cell_mass": p.mass / p.nelem ** p.grade,
            "nsteps": nsteps, "dt": dt, "n_outputs": n_outputs,
            "temporal_error": temporal_error,
            "with_capillarity": p.with_capillarity,
            "tip_azimuthal_correction": p.tip_azimuthal_correction,
            "scheme": p.default_timestepping_scheme,
            "V_TC": V_TC,
        }
        label = f"oh={oh_sb:g},r0={r0:g}" + (f",{tag}" if tag else "")
        print(f"[{label}] mu={p.mu:.6g} tau_vis={p.tau_vis:.6g} "
              f"t_end={t_end:.6g} M={p.mass:.6g} r_max={r_max:.6g} "
              f"N={p.nelem} dt={dt:.6g} R0*={p.r0_star:.4g}", flush=True)

        history = p.create_text_file_output("control1d_axi.txt",
                                            header=COLUMNS)

        out_times = np.geomspace(t_end * 1e-4, t_end, n_outputs)
        prev = 0.0
        # The identity is integrated from the initial state h_tip = 1, u = 0.
        h_ode, u_prev, r_prev = 1.0, 0.0, r0
        worst_drift = 0.0
        for t in out_times:
            # startstep tiny on the first leg only: the impulsively loaded tip
            # has a sqrt(t) start-up singularity and a large first step
            # corrupts it. maxstep is the SAME on every leg.
            start = t_end * 1e-6 if prev == 0.0 else min(dt, t - prev)
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
            rad = p.rim_radius()
            _, du = p.far_field_drift()
            tstar = now / p.tau_vis
            h_ode = _advance_h_tip(h_ode, prev, now, p.mu,
                                   u_prev, r_prev, u, rad)
            row = dict(zip(COLUMNS,
                           (now, tstar, u, u / V_TC, rad, h_ode, du)))
            history.add_row(*[row[c] for c in COLUMNS])
            for c in COLUMNS:
                series[c].append(row[c])
            if probe is not None:
                probe(p, row)
            worst_drift = max(worst_drift, abs(du))
            prev, u_prev, r_prev = now, u, rad

        meta["ustar_final"] = p.tip_velocity() / V_TC
        meta["R_final"] = p.rim_radius()
        meta["h_tip_cell_final"] = p.tip_thickness()
        meta["h_tipode_final"] = h_ode
        meta["metric_defect_final"] = p.metric_defect()
        meta["metric_closure_final"] = p.metric_closure()
        meta["tstar_reached"] = float(
            p.get_current_time(as_float=True)) / p.tau_vis
        meta["worst_far_field_drift"] = worst_drift
        # Not fatal: the run is still the best available record and the drift
        # is in the table. But an answer computed on a domain the disturbance
        # already reached is not a 1D answer, so say so loudly.
        meta["domain_ok"] = bool(worst_drift <= drift_tol)
        if not meta["domain_ok"]:
            print(f"[{label}] WARNING: far-field drift {worst_drift:.3e} "
                  f"exceeds {drift_tol:g}; the disturbance reached the clamped "
                  f"end and the result is contaminated. Raise --mass.",
                  flush=True)
        (outdir / "case1d_axi.json").write_text(json.dumps(meta, indent=2))
        print(f"[{label}] done: u*({tstar_end:g}) = {meta['ustar_final']:.10f} "
              f"R = {meta['R_final']:.6g} drift = {worst_drift:.2e} "
              f"ok = {meta['domain_ok']}", flush=True)

    meta["series"] = {c: np.asarray(v) for c, v in series.items()}
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oh", type=float, required=True, help="Oh_SB (Savva-Bush)")
    ap.add_argument("--r0", type=float, default=5.0,
                    help="initial hole radius in h0; R0 = 0 is excluded and "
                         "R0 >= 5 is the production floor (default 5)")
    ap.add_argument("--tstar-end", type=float, required=True)
    ap.add_argument("--nsteps", type=int, required=True,
                    help="fixed dt = t_end/nsteps; re-run with 2x to check it")
    ap.add_argument("--n-outputs", type=int, default=300)
    ap.add_argument("--temporal-error", type=float, default=1e-3,
                    help="recorded for provenance; the estimator never binds")
    ap.add_argument("--mass", type=float, default=None,
                    help="per-radian domain mass; default (r_max^2-R0^2)/2")
    ap.add_argument("--nelem", type=int, default=None,
                    help="default holds the tip cell fixed in radius")
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--drift-tol", type=float, default=1e-4)
    ap.add_argument("--max-newton", type=int, default=None,
                    help="raise the Newton iteration cap (default 10)")
    ap.add_argument("--gcn", action="store_true",
                    help="globally convergent (line-search) Newton")
    ap.add_argument("--tip-azimuthal-correction", action="store_true",
                    help="N_rr(R) = -2 + pi h_tip/(2R) instead of -2")
    ap.add_argument("--tag", type=str, default=None,
                    help="free-form label recorded in case1d_axi.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run_case(args.oh, args.tstar_end, args.nsteps, args.n_outputs,
             args.out.resolve(), r0=args.r0, mass=args.mass, nelem=args.nelem,
             grade=args.grade, temporal_error=args.temporal_error,
             drift_tol=args.drift_tol, max_newton=args.max_newton,
             gcn=args.gcn, tag=args.tag,
             tip_azimuthal_correction=args.tip_azimuthal_correction)


if __name__ == "__main__":
    main()
