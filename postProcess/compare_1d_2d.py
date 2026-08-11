#!/usr/bin/env python3
"""Compare the 1D long-wave model against the 2D co-moving FEM, at matched Oh.

WHAT THIS ANSWERS. The 2D sweep established that the velocity deficit decays as
``1 - u* = C sqrt(Oh_SB / t*)`` with ``C -> 1/(2 sqrt(pi))``. The obvious next
question is whether the reduced-order long-wave model reproduces that. It turns
out it cannot be asked directly, because the 1D solver does not survive to the
asymptotic regime -- see the breakdown discussion below -- so this script does
two separate comparisons instead: an EARLY-TIME one over the window where both
solvers have data, and an ASYMPTOTIC one where the 1D side is a power-law
extrapolation rather than a measurement, and is labelled as such.

THE TIMESTEP IS NOT OPTIONAL HERE -- FOR THE DATA THIS SCRIPT READS. Those runs
predate the 2026-08-11 switch to BDF2, so they are first order and need the
Richardson treatment below. A re-run on the current solver would not.
The 1D solver was forced onto BDF1 and its
temporal error estimator under-reports by about nine orders, so dt sits on
maxstep every step and every u* is a function of it. Each Oh was therefore run
at two or three fixed dt. This script interpolates each dt level onto a common
t* grid and Richardson-extrapolates to dt -> 0 assuming the first-order rate
that the Oh_SB = 1 ladder measured directly (successive ratios 1.97/1.99/1.99).
Raw single-dt 1D numbers are never compared against 2D here; only extrapolated
ones, with the extrapolation size reported so the reader can judge it.

THE 1D BREAKDOWN. Every 1D run terminated with Newton stalled at its iteration
cap, at a tip thickness of 62 (Oh_SB = 1) to 337 (Oh_SB = 0.035) times the
initial sheet thickness. The termination time is essentially independent of dt
-- at Oh_SB = 0.38372, nsteps 169242/338485/676971 stopped at t* = 431/405/405
-- and a retry with a 40-iteration cap and line-search Newton stopped in the
same place. That is a property of the model and its formulation, not of the
timestep or the nonlinear solver, so the 1D curves simply end where they end.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import warnings

import numpy as np

V_TC = math.sqrt(2.0)
C_STAR = 1.0 / (2.0 * math.sqrt(math.pi))

# The full-range 1D runs, per Oh_SB, coarsest last. Named explicitly because the
# sweep directory also holds superseded shorter runs (nsteps 360000/180000/
# 353553) from an earlier t*-capped sizing, and one of those falls BETWEEN two
# valid levels at Oh_SB = 0.09121, so "take the largest few" silently picks
# wrong. rsync had no --delete, which is why they are still on disk.
RUNS: dict[float, list[int]] = {
    0.035:   [1019351, 509675],
    0.0565:  [1040258, 520129],
    0.09121: [1165686, 582843, 291421],
    0.14724: [1007411, 503705],
    0.2377:  [868970, 434485],
    0.38372: [676971, 338485, 169242],
    0.61945: [1716152, 858076],
    1.0:     [1448862, 724431],
}
CASE2D = {oh: f"Oh_SB_{oh:g}_u995" for oh in RUNS}


def load_1d(root: Path, oh: float, nsteps: int) -> np.ndarray | None:
    p = root / f"oh{oh:g}_s{nsteps}" / "control1d.txt"
    if not p.exists():
        return None
    a = np.loadtxt(p)
    return a if a.ndim == 2 and len(a) > 5 else None


def load_2d(root: Path, case: str) -> np.ndarray:
    """Concatenate every preserved segment. Each restart truncates control.txt,
    so the history lives across control_part*.txt and must be reassembled."""
    d = root / case / "run_validation"
    rows = []
    for f in sorted(d.glob("control*.txt")):
        # A live case's control.txt can be header-only between outputs, and a
        # just-restarted one is empty; both are normal, not errors.
        if f.stat().st_size == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = np.loadtxt(f)
        except Exception:
            continue
        if a.size == 0:
            continue
        rows.append(a[None, :] if a.ndim == 1 else a)
    a = np.vstack(rows)
    a = a[np.argsort(a[:, 1])]
    _, i = np.unique(a[:, 1], return_index=True)
    return a[i]


def richardson(root: Path, oh: float, ngrid: int = 160):
    """u*(t*) extrapolated to dt -> 0, on a common log-spaced grid.

    Returns (t*, u*_extrapolated, u*_finest, correction) or None. The correction
    is |u_extrap - u_finest|: if it is not small compared with 1 - u*, the
    extrapolation is doing more work than the data supports and the caller
    should say so rather than quoting the number.
    """
    levels = []
    for ns in RUNS[oh]:
        a = load_1d(root, oh, ns)
        if a is not None:
            levels.append((ns, a))
    if len(levels) < 2:
        return None
    # Common range: every level must cover it. Levels stop at slightly different
    # t* because the breakdown time drifts a little with dt.
    lo = max(l[1][0, 1] for l in levels)
    hi = min(l[1][-1, 1] for l in levels)
    if not (hi > lo * 1.5):
        return None
    ts = np.geomspace(lo, hi, ngrid)
    interp = [np.interp(ts, a[:, 1], a[:, 3]) for _, a in levels]
    fine, coarse = interp[0], interp[1]          # levels[0] is finest (dt halved)
    ext = 2.0 * fine - coarse                    # first-order Richardson
    return ts, ext, fine, np.abs(ext - fine)


def fit_law(ts, u, frac=0.5):
    """Free-exponent fit 1 - u* = C t*^a. Returns (a, C)."""
    m = (ts > frac * ts.max()) & (u < 1.0)
    if m.sum() < 6:
        return float("nan"), float("nan")
    a, b = np.polyfit(np.log(ts[m]), np.log(1.0 - u[m]), 1)
    return a, math.exp(b)


def collapse_const(ts, u, oh, frac=0.5):
    """C*sqrt(Oh) with the exponent CONSTRAINED to -1/2.

    This is the collapse constant the 2D sweep is testing against
    1/(2 sqrt(pi)), and it is a different number from the free-fit prefactor
    above whenever the measured exponent is not -1/2. Reporting one as the other
    is how the 2D constants first came out wrong here: at Oh_SB = 0.035 the free
    fit gives a = -0.43 and a prefactor of 0.194, against 0.407 for the
    constrained form. Quote the constrained one only alongside the measured
    exponent, so a case that is not yet on the -1/2 law cannot masquerade as one
    that is.
    """
    m = (ts > frac * ts.max()) & (u < 1.0)
    if m.sum() < 6:
        return float("nan")
    return math.exp(np.mean(np.log(1.0 - u[m]) + 0.5 * np.log(ts[m]))) * math.sqrt(oh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oned", type=Path, default=Path(
        "/mnt/stokesInt1/running-simulations/elastic-taylor-culick/trouton1d-oh-sweep"))
    ap.add_argument("--twod", type=Path, default=Path(
        "/mnt/stokesInt1/running-simulations/elastic-taylor-culick/u995-sweep"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    print("Asymptotic behaviour.  2D C*sqrt(Oh) has the exponent constrained to")
    print(f"-1/2 and is meaningful only where 2D a is close to it; target {C_STAR:.5f}.\n")
    print(f"{'Oh_SB':>8} {'t*end_1D':>9} {'h_tip':>6} | {'1D a':>7} | "
          f"{'2D a':>7} {'2D C√Oh':>8} | {'Rich.corr':>9} {'1-u* @end':>10}")
    for oh in sorted(RUNS):
        r = richardson(args.oned, oh)
        d2 = load_2d(args.twod, CASE2D[oh])
        a2, _ = fit_law(d2[:, 1], d2[:, 3])
        C2 = collapse_const(d2[:, 1], d2[:, 3], oh)
        if r is None:
            print(f"{oh:8g} {'--':>9} {'--':>6} | {'--':>7} | "
                  f"{a2:7.4f} {C2:8.4f} | {'--':>9} {'--':>10}")
            continue
        ts, ext, fine, corr = r
        a1, _ = fit_law(ts, ext)
        raw = load_1d(args.oned, oh, RUNS[oh][0])
        htip = raw[-1, 4]
        defic = 1.0 - ext[-1]
        print(f"{oh:8g} {ts[-1]:9.1f} {htip:6.0f} | {a1:7.4f} | "
              f"{a2:7.4f} {C2:8.4f} | {corr[-1]:9.2e} {defic:10.2e}")
        summary[f"{oh:g}"] = dict(
            tstar_end_1d=float(ts[-1]), h_tip_end=float(htip),
            a_1d=float(a1),
            a_2d=float(a2), C_sqrtOh_2d=float(C2),
            richardson_correction=float(corr[-1]),
            deficit_1d_end=float(defic),
            # Honest flag: the extrapolation is only meaningful while the
            # correction is small against the quantity being extrapolated.
            extrapolation_trustworthy=bool(corr[-1] < 0.25 * defic),
        )

    # ---- early-time comparison, where BOTH solvers have real data ----
    print("\nEarly time, u* at matched t* (1D Richardson vs 2D):")
    probes = [1.0, 3.0, 10.0, 30.0, 100.0]
    hdr = "  ".join(f"t*={p:g}".rjust(9) for p in probes)
    print(f"{'Oh_SB':>8}  {'':4}{hdr}")
    for oh in sorted(RUNS):
        r = richardson(args.oned, oh)
        if r is None:
            continue
        ts, ext, *_ = r
        d2 = load_2d(args.twod, CASE2D[oh])
        row1, row2 = [], []
        for p in probes:
            v1 = np.interp(p, ts, ext, left=np.nan, right=np.nan) if ts[0] <= p <= ts[-1] else np.nan
            v2 = np.interp(p, d2[:, 1], d2[:, 3], left=np.nan, right=np.nan) if d2[0, 1] <= p <= d2[-1, 1] else np.nan
            row1.append(v1); row2.append(v2)
        f = lambda v: ("  n/a  " if not np.isfinite(v) else f"{v:.5f}").rjust(9)
        print(f"{oh:8g}  1D  " + "  ".join(f(v) for v in row1))
        print(f"{'':8}  2D  " + "  ".join(f(v) for v in row2))
        summary.setdefault(f"{oh:g}", {})["early"] = {
            f"t{p:g}": dict(oned=None if not np.isfinite(a) else float(a),
                            twod=None if not np.isfinite(b) else float(b))
            for p, a, b in zip(probes, row1, row2)}

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json_out}")

    make_figure(args, summary)


def make_figure(args, summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 1.0,
                         "figure.dpi": 200, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    cmap = plt.get_cmap("viridis")
    ohs = sorted(RUNS)
    for k, oh in enumerate(ohs):
        col = cmap(k / max(1, len(ohs) - 1))
        d2 = load_2d(args.twod, CASE2D[oh])
        ax[0].semilogx(d2[:, 1], d2[:, 3], "-", color=col, lw=1.5,
                       label=f"{oh:g}")
        ax[1].loglog(d2[:, 1], 1 - d2[:, 3], "-", color=col, lw=1.5)
        r = richardson(args.oned, oh)
        if r is None:
            continue
        ts, ext, fine, corr = r
        # Draw the extrapolation only where it means something. Once the
        # Richardson correction exceeds the deficit it is correcting, the
        # extrapolated curve is dominated by the correction and bends upward --
        # visible at Oh_SB = 0.035, where it would read as a spurious rise in
        # 1 - u* at late time. Plot the finest measured level there instead.
        ok = corr < (1.0 - ext)
        if ok.sum() > 5:
            ax[0].semilogx(ts[ok], ext[ok], "--", color=col, lw=1.1)
            ax[1].loglog(ts[ok], 1 - ext[ok], "--", color=col, lw=1.1)
            ax[1].plot(ts[ok][-1], 1 - ext[ok][-1], "x", color=col, ms=5, mew=1.4)
        else:
            ax[0].semilogx(ts, fine, ":", color=col, lw=1.1)
            ax[1].loglog(ts, 1 - fine, ":", color=col, lw=1.1)
            ax[1].plot(ts[-1], 1 - fine[-1], "x", color=col, ms=5, mew=1.4)

    tt = np.geomspace(1, 1e5, 50)
    for oh in (0.035, 1.0):
        ax[1].loglog(tt, C_STAR * np.sqrt(oh / tt), ":", color="0.35", lw=1.0)
    ax[0].set_xlabel(r"$t^*$"); ax[0].set_ylabel(r"$u^* = U/V_{TC}$")
    ax[0].set_ylim(0.4, 1.02); ax[0].axhline(1.0, color="0.8", lw=0.8, zorder=0)
    ax[0].set_title("(a)  approach to $V_{TC}$", loc="left", fontsize=9)
    ax[1].set_xlabel(r"$t^*$"); ax[1].set_ylabel(r"$1-u^*$")
    ax[1].set_title(r"(b)  deficit;  dotted $=\frac{1}{2\sqrt{\pi}}\sqrt{Oh/t^*}$",
                    loc="left", fontsize=9)
    leg = ax[0].legend(title=r"$Oh_{SB}$", fontsize=6.5, title_fontsize=7,
                       ncol=2, loc="lower right", frameon=False)
    ax[0].text(0.03, 0.93, "solid: 2D FEM\ndashed: 1D long-wave ($dt\\to0$)\n"
               r"$\times$: 1D breakdown", transform=ax[0].transAxes,
               fontsize=6.5, va="top")
    for a in ax:
        a.grid(True, which="both", alpha=0.15, lw=0.5)
    out = args.out / "fig3_1d_vs_2d.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
