#!/usr/bin/env python3
"""Driver for TC1D-ALE: one case per directory, fixed timestep, geometric output.

Deliberately the same shape as ``validation/trouton1d_sweep.py`` so that a
TC1D-FEM and a TC1D-ALE run of the same case differ only in the solver, and the
two output tables can be compared row for row. In particular:

  * ``maxstep`` is held FIXED for the whole run rather than tied to the output
    interval. pyoomph's temporal error estimator under-reports by about nine
    orders of magnitude on this problem and never binds, so any claim of
    temporal convergence has to come from running the same case at ``nsteps``
    and ``2 nsteps``. That is only meaningful if dt is the same everywhere.

    THAT ``temporal_error`` IS INERT IS A MEASUREMENT, NOT AN ASSUMPTION, since
    an adaptive controller quietly rejecting steps would invalidate every
    convergence order below. Counted over the reference run (Oh_SB = 1,
    t* = 30, nsteps = 6400): 6354 of 6424 accepted steps were taken at exactly
    ``maxstep``, and each of the remaining 70 is a distinct value appearing once
    -- the clip onto an output time at the end of each of the 40 legs, plus the
    deliberately tiny start-up leg. There is no repeated reduced step, i.e. no
    rejected step anywhere in the run. The solver reports "GLOBAL TEMPORAL
    ERROR 0" and asks to increase dt by a factor of 1000 on every step, which
    ``maxstep`` refuses.
  * outputs are GEOMETRIC in time, because u* rises through most of its range in
    the first per cent of the run.
  * ``outstep=False`` is load-bearing: ``Problem.run`` documents ``outstep`` as
    defaulting to True whenever ``temporal_error`` is passed, and True means one
    profile file per INTERNAL timestep.

WHAT THIS DRIVER MEASURES THAT THE MASS-COORDINATE ONE CANNOT. The edge position
x_c is a degree of freedom here, so the displacement law

    X_c = T - 1 + (3/2)/(1+T) + O(T^-2)

is checked against a directly computed X_c rather than against a time integral
of the tip velocity. Both are recorded, and their difference is reported as
``xc_integral_mismatch``.

READ THAT DIAGNOSTIC WITH CARE. It is NOT a measurement of how well the
kinematic condition is enforced. The integral is a trapezoidal sum over the
OUTPUT legs, which are geometrically spaced and few, so its own quadrature error
dominates: at t* = 30 it is 2.10e-2 with 40 outputs and 2.09e-4 with 400, a
factor of 100 for a factor of 10 in spacing, which is the trapezoidal rule
behaving as the trapezoidal rule. It is independent of the solver timestep
(identical to three digits across a 32x range of dt). What it does establish is
that the edge degree of freedom and the tip velocity are the same physical
quantity to the accuracy the sampling can see; the kinematic condition itself is
imposed exactly by the Lagrange multiplier, to Newton tolerance.

UNITS. Ahsan et al.'s variables, in which the late-time results are stated, are
X = x/l_I with l_I = 4 Oh_SB, T = t*/4, V = u*. See
``validation/ahsan_thinfilm.py`` for the derivation of the mapping.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulationCases"))
from trouton1d_ale import Trouton1DALEProblem  # noqa: E402

V_TC = math.sqrt(2.0)


def run_case(oh_sb: float, tstar_end: float, nsteps: int, n_outputs: int,
             outroot: Path, l0: float | None, nelem: int, grade: float,
             temporal_error: float = 1e-3, drift_tol: float = 1e-4,
             max_newton: int | None = None, gcn: bool = False,
             ale: bool = True,
             profiles: bool = False,
             output_spacing: str = "geom") -> dict:
    # Reject an unknown schedule rather than falling through to the
    # geometric default: tc_dtc finite-differences the record with a
    # single dT, so a typo here would hand it a geometric grid and it
    # would return a plausible, wrong derivative without complaint.
    if output_spacing not in ("geom", "uniform"):
        raise ValueError(f"output_spacing must be 'geom' or 'uniform', "
                         f"got {output_spacing!r}")
    outdir = outroot
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    with Trouton1DALEProblem() as p:
        p.Oh_SB = oh_sb
        p.nelem = nelem
        p.grade = grade
        p.ale = ale
        t_end = tstar_end * p.tau_vis
        # DOMAIN SIZING. The edge travels X_c ~ T, i.e. x_c ~ l_I t*/4, and the
        # disturbance beyond it spans ~ln(1+T) + 5 in X. Default leaves the
        # larger of 20% headroom and 30 l_I of undisturbed sheet at the end; the
        # assumption is checked every output by far_field_drift rather than
        # assumed.
        if l0 is None:
            xc_end = p.l_I * (tstar_end / 4.0)
            l0 = xc_end + max(0.2 * xc_end, 30.0 * p.l_I)
        p.l0 = l0
        p.set_output_directory(str(outdir))
        p.initialise()
        p.write_states = False

        dt = t_end / nsteps
        meta = {
            "code": "TC1D-ALE", "Oh_SB": oh_sb, "mu": p.mu,
            "tau_vis": p.tau_vis, "l_I": p.l_I,
            "tstar_end": tstar_end, "T_end": tstar_end / 4.0, "t_end": t_end,
            "l0": p.l0, "nelem": p.nelem, "grade": p.grade,
            "ale": p.ale,
            "nsteps": nsteps, "dt": dt, "n_outputs": n_outputs,
            "scheme": p.default_timestepping_scheme,
            "tip_cell_initial": p.tip_cell_size(),
            "mass_initial": p.total_mass(),
            "V_TC": V_TC,
        }
        print(f"[ale oh={oh_sb:g}] mu={p.mu:.6g} l0={p.l0:.6g} N={p.nelem} "
              f"grade={p.grade:g} t_end={t_end:.6g} dt={dt:.6g} "
              f"tip_cell={meta['tip_cell_initial']:.3e}", flush=True)

        history = p.create_text_file_output(
            "control1d_ale.txt",
            header=["t", "tstar", "T", "xc", "Xc", "u_tip", "ustar", "h_tip",
                    "h_tip_exact", "R", "Xc_pred", "Xc_int", "mass",
                    "affinity", "roughness", "drift_h", "drift_u",
                    "tip_cell"])

        # Geometric by default, because the early-time laws this solver is
        # checked against are power laws in T and a uniform grid wastes almost
        # all of its samples on the last decade. A uniform grid is available
        # because tc_dtc (ahsan_campaign.py) takes a single-dT finite
        # difference of the record and therefore REQUIRES one; resampling a
        # geometric record onto a uniform grid and then differencing it would
        # corrupt the derivative, so the schedule has to change at source.
        if output_spacing == "uniform":
            out_times = np.linspace(t_end / n_outputs, t_end, n_outputs)
        elif n_outputs == 1:
            # geomspace(a, b, 1) returns [a], so a single geometric output
            # would land at t_end*1e-4 -- the START of the range -- and the run
            # would record its first instant instead of its last.
            out_times = np.array([t_end])
        else:
            out_times = np.geomspace(t_end * 1e-4, t_end, n_outputs)
        prev = 0.0
        u_prev = 0.0
        xc_int = 0.0                      # trapezoidal integral of u dt
        worst_drift = 0.0
        worst_affinity = 0.0
        worst_rough = 0.0
        rows = []
        t0 = time.time()
        for t in out_times:
            # Tiny first step only: the impulsively loaded edge has a sqrt(t)
            # start-up singularity. maxstep is identical on every leg.
            start = t_end * 1e-6 if prev == 0.0 else min(dt, t - prev)
            extra = {}
            if max_newton is not None:
                extra["max_newton_iterations"] = max_newton
            if gcn:
                extra["globally_convergent_newton"] = True
            p.run(t, startstep=start, maxstep=dt, temporal_error=temporal_error,
                  outstep=False, out_initially=False, **extra)
            # Profiles are OFF by default. None of the verification targets need
            # them and they are the bulk of the disk cost: one file per output
            # leg per run, at N nodes each.
            if profiles:
                p.output()
            now = float(p.get_current_time(as_float=True))
            tstar = now / p.tau_vis
            T = tstar / 4.0
            u = p.tip_velocity()
            ustar = u / V_TC
            xc = p.edge_position()
            Xc = xc / p.l_I
            # dXc/dT = u* exactly (l_I/t_I = V_TC), so R needs no derivative.
            R = (1.0 + T) ** 2 * (1.0 - ustar)
            Xc_pred = T - 1.0 + 1.5 / (1.0 + T)
            xc_int += 0.5 * (u + u_prev) * (now - prev)
            dh, du = p.far_field_drift()
            aff = p.mesh_affinity_residual()
            rough = p.thickness_roughness()
            history.add_row(now, tstar, T, xc, Xc, u, ustar, p.tip_thickness(),
                            1.0 + tstar / 4.0, R, Xc_pred, xc_int / p.l_I,
                            p.total_mass(), aff, rough, dh, du,
                            p.tip_cell_size())
            rows.append([T, Xc, ustar, R])
            worst_drift = max(worst_drift, abs(dh), abs(du))
            worst_affinity = max(worst_affinity, aff)
            worst_rough = max(worst_rough, rough)
            prev, u_prev = now, u

        tab = np.array(rows)
        tstar_reached = float(p.get_current_time(as_float=True)) / p.tau_vis
        T_final = tstar_reached / 4.0
        meta.update({
            "wall_s": time.time() - t0,
            "tstar_reached": tstar_reached,
            "T_reached": T_final,
            "ustar_final": p.tip_velocity() / V_TC,
            "h_tip_final": p.tip_thickness(),
            "h_tip_exact_final": 1.0 + tstar_reached / 4.0,
            "h_tip_rel_err": abs(p.tip_thickness() / (1.0 + tstar_reached / 4.0) - 1.0),
            "Xc_final": p.edge_position() / p.l_I,
            "Xc_pred_final": T_final - 1.0 + 1.5 / (1.0 + T_final),
            "R_final": float(tab[-1, 3]),
            "xc_integral_mismatch": abs(p.edge_position() - xc_int),
            "mass_final": p.total_mass(),
            "mass_rel_drift": abs(p.total_mass() / meta["mass_initial"] - 1.0),
            "worst_far_field_drift": worst_drift,
            "worst_mesh_affinity": worst_affinity,
            "worst_thickness_roughness": worst_rough,
            "tip_cell_final": p.tip_cell_size(),
        })
        meta["domain_ok"] = bool(worst_drift <= drift_tol)
        if not meta["domain_ok"]:
            print(f"[ale oh={oh_sb:g}] WARNING: far-field drift "
                  f"{worst_drift:.3e} exceeds {drift_tol:g}; the disturbance "
                  f"reached the truncation. Re-run with a larger --l0.",
                  flush=True)
        (outdir / "case1d_ale.json").write_text(json.dumps(meta, indent=2))
        for Tq in (1, 5, 7.5, 10, 20, 50, 100, 200):
            if Tq <= T_final + 1e-9:
                i = int(np.argmin(np.abs(tab[:, 0] - Tq)))
                print(f"  T={tab[i,0]:8.4f} t*={4*tab[i,0]:9.4f} "
                      f"u*={tab[i,2]:.7f} R={tab[i,3]:.6f} "
                      f"Xc={tab[i,1]:.6f} Xc-pred={tab[i,1]-(tab[i,0]-1+1.5/(1+tab[i,0])):+.3e}",
                      flush=True)
        print(f"[ale oh={oh_sb:g}] done: u*({tstar_end:g}) = "
              f"{meta['ustar_final']:.10f} R={meta['R_final']:.6f} "
              f"affinity={worst_affinity:.2e} rough={worst_rough:.2e} "
              f"mass_drift={meta['mass_rel_drift']:.2e} "
              f"drift={worst_drift:.2e} wall={meta['wall_s']:.0f}s", flush=True)
        return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oh", type=float, default=1.0, help="Oh_SB (Savva-Bush)")
    ap.add_argument("--tstar-end", type=float, required=True)
    ap.add_argument("--nsteps", type=int, required=True,
                    help="fixed dt = t_end/nsteps; run twice with 2x for the order")
    ap.add_argument("--nelem", type=int, default=800)
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--l0", type=float, default=None)
    ap.add_argument("--n-outputs", type=int, default=300)
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--drift-tol", type=float, default=1e-4)
    ap.add_argument("--max-newton", type=int, default=None)
    ap.add_argument("--gcn", action="store_true")
    ap.add_argument("--output-spacing", choices=("geom", "uniform"),
                    default="geom",
                    help="output-time schedule; uniform is required by "
                         "tc_dtc, which finite-differences the record")
    ap.add_argument("--profiles", action="store_true",
                    help="also write an h/u profile per output leg")
    ap.add_argument("--no-ale", dest="ale", action="store_false",
                    help="drop the ALE correction: WRONG on a moving mesh, "
                         "kept to measure how large the correction is")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # A negative or zero value here does not fail loudly inside pyoomph -- it
    # produces a non-physical Oh, a zero-length run or a mesh with no interior,
    # any of which yields a plausible-looking output file. Stop before setup.
    for name, value in (("--oh", a.oh), ("--tstar-end", a.tstar_end),
                        ("--nsteps", a.nsteps), ("--n-outputs", a.n_outputs),
                        ("--grade", a.grade), ("--drift-tol", a.drift_tol)):
        if value <= 0:
            ap.error(f"{name} must be positive, got {value!r}")
    if a.nelem < 2:
        ap.error(f"--nelem must be at least 2, got {a.nelem!r}")
    if a.l0 is not None and a.l0 <= 0:
        ap.error(f"--l0 must be positive, got {a.l0!r}")

    run_case(a.oh, a.tstar_end, a.nsteps, a.n_outputs, a.out.resolve(),
             a.l0, a.nelem, a.grade, a.temporal_error, a.drift_tol,
             a.max_newton, a.gcn, a.ale, a.profiles,
             output_spacing=a.output_spacing)


if __name__ == "__main__":
    main()
