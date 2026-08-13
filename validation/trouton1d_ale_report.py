#!/usr/bin/env python3
"""Read a directory of TC1D-ALE runs and print the verification tables.

Kept separate from the driver so that the numbers below are computed from what
was actually written to disk, not from live solver state: an accuracy claim that
can only be reproduced by re-running the solver is not a checkable one.

Three tables:

  ORDER   successive differences and their ratios along a ladder in which ONE
          parameter is halved (or doubled) and everything else is held fixed.
          The observed order is log2 of the ratio of successive differences, and
          the Richardson limit is the finest value plus the last difference
          divided by (ratio - 1).

  LATE    the quantity under test, R = (1+T)^2 (1 - dXc/dT), sampled from the
          series, together with the two displacement checks. dXc/dT is u* here,
          exactly, because l_I/t_I = V_TC.

  HEALTH  the diagnostics that would betray a broken moving mesh: the departure
          of the node positions from the affine stretch, the checkerboard
          indicator on h, mass conservation, the tip-thickness law h = 1 + t*/4
          which the solver is never told, and the mismatch between the edge
          position as a degree of freedom and as the time integral of the tip
          velocity.

  CROSS    optional whole-curve comparison against a TC1D-FEM run of the same
           case (``--fem <dir>``), which is the only one of the four solvers
           sharing a framework with this one. Agreement there separates "the
           ALE machinery works" from "the two share a bug in pyoomph"; it can
           establish the first and by construction cannot establish the second.

Usage: uv run python validation/trouton1d_ale_report.py <runroot> [--fem DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

#: The published constants this solver is being held to.
USTAR_30_REFERENCE = 0.98073          # Richardson limit, docs/README.md
R_REFERENCE = {50: 1.498623, 100: 1.499660, 200: 1.499919, 300: 1.499993}


def load(rundir: Path):
    meta = json.loads((rundir / "case1d_ale.json").read_text())
    series = np.loadtxt(rundir / "control1d_ale.txt", skiprows=1)
    with (rundir / "control1d_ale.txt").open() as fh:
        cols = fh.readline().lstrip("#").split()
    return meta, series, {c: i for i, c in enumerate(cols)}


def at_T(series, col, Tq: float) -> float:
    """Value of a column at exactly T = Tq, by linear interpolation.

    Not the nearest row. The outputs are geometrically spaced, so near T = 100
    the nearest row can be a per cent away, and R = (1+T)^2 (1 - u*) is steep
    enough there that the difference is comparable to the 5e-4 tolerance the
    coefficient is being quoted to. Comparing a value at T = 99.0 against a
    reference at T = 100 would manufacture a discrepancy.
    """
    T = series[:, 2]
    return float(np.interp(Tq, T, series[:, col]))


def order_table(title: str, knob: str, runs: list[tuple[float, float]]) -> None:
    """runs: (knob value, measured quantity), coarsest first."""
    print(f"\n{title}")
    print(f"  {knob:>10s}  {'u*(t*=30)':>16s}  {'difference':>13s}  "
          f"{'ratio':>7s}  {'order':>6s}")
    prev_d = None
    for j, (k, v) in enumerate(runs):
        if j == 0:
            print(f"  {k:10g}  {v:16.10f}")
            continue
        d = v - runs[j - 1][1]
        if prev_d is None or d == 0.0:
            print(f"  {k:10g}  {v:16.10f}  {d:13.3e}")
        else:
            r = prev_d / d
            print(f"  {k:10g}  {v:16.10f}  {d:13.3e}  {r:7.3f}  "
                  f"{math.log2(abs(r)):6.3f}")
        prev_d = d
    if len(runs) >= 3:
        d1 = runs[-2][1] - runs[-3][1]
        d2 = runs[-1][1] - runs[-2][1]
        if d2 != 0.0:
            r = d1 / d2
            lim = runs[-1][1] + d2 / (r - 1.0) if abs(r - 1.0) > 1e-12 else float("nan")
            print(f"  Richardson limit: {lim:.10f}   "
                  f"(reference {USTAR_30_REFERENCE:.5f}, "
                  f"relative {abs(lim - USTAR_30_REFERENCE) / USTAR_30_REFERENCE:.2e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runroot", type=Path)
    ap.add_argument("--fem", type=Path, default=None,
                    help="a TC1D-FEM run directory (control1d.txt) to compare "
                         "the whole u*(t*) curve against")
    ap.add_argument("--fem-against", default=None,
                    help="name of the TC1D-ALE run to compare with --fem")
    a = ap.parse_args()
    root = a.runroot

    cases = {}
    for d in sorted(root.iterdir()):
        if (d / "case1d_ale.json").exists():
            cases[d.name] = load(d)

    dt_runs = sorted(((m["nsteps"], m["ustar_final"])
                      for n, (m, _, _) in cases.items() if n.startswith("dt_n")))
    if dt_runs:
        order_table("TEMPORAL CONVERGENCE (Oh_SB = 1, t* = 30, N = 800, BDF2)",
                    "nsteps", dt_runs)

    dx_runs = sorted(((m["nelem"], m["ustar_final"])
                      for n, (m, _, _) in cases.items() if n.startswith("dx_N")))
    if dx_runs:
        order_table("SPATIAL CONVERGENCE (Oh_SB = 1, t* = 30, nsteps = 12800)",
                    "nelem", dx_runs)

    l0_runs = sorted((m["l0"], m["ustar_final"])
                     for n, (m, _, _) in cases.items() if n.startswith("l0_"))
    if l0_runs:
        print("\nDOMAIN TRUNCATION (t* = 30, N = 800, nsteps = 6400)")
        for L, v in l0_runs:
            print(f"  l0 = {L:8g}   u*(30) = {v:.10f}")

    late = {n: c for n, c in cases.items() if n.startswith("T")}
    if late:
        print("\nLATE-TIME COEFFICIENT  R = (1+T)^2 (1 - dXc/dT)")
        print(f"  {'run':>18s} {'T':>7s} {'R':>10s} {'R_ref':>10s} "
              f"{'Xc-Xc_pred':>12s} {'T-Xc':>10s} {'(T-Xc)_pred':>12s}")
        for n, (m, s, c) in sorted(late.items()):
            for Tq in (50, 100, 200, 300):
                if Tq > m["T_reached"] + 1e-9:
                    continue
                T = float(Tq)
                R = at_T(s, c["R"], Tq)
                Xc = at_T(s, c["Xc"], Tq)
                pred = T - 1.0 + 1.5 / (1.0 + T)
                ref = R_REFERENCE.get(Tq, float("nan"))
                print(f"  {n:>18s} {T:7.2f} {R:10.6f} {ref:10.6f} "
                      f"{Xc - pred:12.3e} {T - Xc:10.6f} "
                      f"{1.0 - 1.5 / (1.0 + T):12.6f}")

    if late:
        # R is (1+T)^2 times a small difference, so at T = 100 an error of 1e-7
        # in u* is 1e-3 in R -- twice the tolerance the coefficient is quoted
        # to. A single run cannot settle it; the ladders below can, because both
        # discretisations are second order (measured above) and the same
        # Richardson step therefore applies.
        print("\nR EXTRAPOLATED, holding one discretisation fixed "
              "(second order assumed, as measured)")
        groups: dict[tuple, list] = {}
        for n, (m, s, c) in late.items():
            groups.setdefault(("nelem", m["T_end"], m["nsteps"]), []).append(
                (m["nelem"], s, c))
            groups.setdefault(("nsteps", m["T_end"], m["nelem"]), []).append(
                (m["nsteps"], s, c))
        for (knob, T_end, fixed), members in sorted(groups.items()):
            if len(members) < 2:
                continue
            # Sort on the knob alone. A bare sort() falls through to comparing
            # the series arrays whenever two runs share a knob value, and numpy
            # raises on the ambiguous truth value rather than ordering them.
            members.sort(key=lambda item: item[0])
            for Tq in (50, 100, 200, 300):
                if Tq > T_end + 1e-9:
                    continue
                vals = [(k, at_T(s, c["R"], Tq)) for k, s, c in members]
                coarse, fine = vals[-2][1], vals[-1][1]
                # Second-order Richardson: the familiar /3 assumes a factor-two
                # refinement between the last two members. A ladder built from,
                # say, 400 and 1600 would extrapolate silently wrong, so take the
                # ratio from the data and refuse a ladder that is not usable.
                r = float(vals[-1][0]) / float(vals[-2][0])
                if not 1.05 < r < 100.0:
                    raise ValueError(
                        f"refinement ratio {r:.3g} between the last two members "
                        f"is not a usable Richardson ladder")
                lim = fine + (fine - coarse) / (r ** 2 - 1.0)
                ref = R_REFERENCE.get(Tq, float("nan"))
                ladder = " ".join(f"{k:g}:{v:.6f}" for k, v in vals)
                print(f"  T={Tq:5g}  vary {knob:6s} (other fixed at {fixed:g})  "
                      f"{ladder}  ->  {lim:.6f}   ref {ref:.6f}   "
                      f"diff {lim - ref:+.2e}")

    if a.fem is not None and a.fem_against is not None:
        fem = np.loadtxt(a.fem / "control1d.txt", skiprows=1)  # t tstar u_tip u* ...
        _, s, c = cases[a.fem_against]
        # Compare on the OVERLAP only, and on the FEM sample times, so neither
        # curve is extrapolated. Both drivers use the same geometric output
        # schedule, so the overlap is essentially the whole run.
        lo = max(fem[0, 1], s[0, c["tstar"]])
        hi = min(fem[-1, 1], s[-1, c["tstar"]])
        sel = (fem[:, 1] >= lo) & (fem[:, 1] <= hi)
        ts, ufem = fem[sel, 1], fem[sel, 3]
        uale = np.interp(ts, s[:, c["tstar"]], s[:, c["ustar"]])
        d = uale - ufem
        print(f"\nTC1D-ALE vs TC1D-FEM, u*(t*) over {lo:.3g} <= t* <= {hi:.3g} "
              f"({sel.sum()} samples)")
        print(f"  max |du*| = {np.max(np.abs(d)):.3e}   "
              f"RMS = {float(np.sqrt(np.mean(d ** 2))):.3e}   "
              f"max relative = {np.max(np.abs(d) / np.maximum(np.abs(ufem), 1e-12)):.3e}")

    print("\nHEALTH OF THE MOVING MESH")
    print(f"  {'run':>18s} {'affine':>10s} {'rough':>10s} {'mass drift':>11s} "
          f"{'h_tip err':>10s} {'xc mismatch':>12s} {'far drift':>10s} {'wall/s':>8s}")
    for n, (m, _, _) in sorted(cases.items()):
        print(f"  {n:>18s} {m['worst_mesh_affinity']:10.2e} "
              f"{m['worst_thickness_roughness']:10.2e} "
              f"{m['mass_rel_drift']:11.2e} {m['h_tip_rel_err']:10.2e} "
              f"{m['xc_integral_mismatch']:12.2e} "
              f"{m['worst_far_field_drift']:10.2e} {m['wall_s']:8.0f}")


if __name__ == "__main__":
    main()
