"""How large must ``Oh`` be before a 2D sheet shows the long-wave asymptotics?

The long-wave (1D) theory of Ahsan et al. removes ``Oh`` from the problem by
rescaling time on the visco-inertial scale.  That rescaling is only legitimate
while the thin-film description holds, which bounds the accessible times from
above by ``T = O(Oh_p)``.  A 2D direct simulation makes no such assumption, so
it can be asked the question the reduced model cannot answer itself: at what
``Oh`` does a real sheet actually spend a resolvable stretch of its history in
the long-wave regime?

The diagnostic used here is the local logarithmic slope

    n(T) = d ln(1 - u*) / d ln T ,

which is ``-2`` inside the long-wave window and ``-1/2`` once the flow has
crossed over to the two-dimensional behaviour.  Reading a slope is far more
robust than fitting a prefactor: it is invariant under rescaling of ``1 - u*``
and so does not depend on the constant that the reduced theory leaves
undetermined.

Usage::

    uv run python postProcess/oh_window.py --sweep <run-root>/u995-sweep
    uv run python postProcess/oh_window.py --sweep ... --reference /tmp/abds_inf.txt
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# t* = t / tau_vis and T = t / t_I are both viscous times; see docs/planar-campaign/README.md
# section 2.  The reduced-model literature works in T, this code in t*.
TSTAR_PER_T = 4.0


def load_control(case_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every ``control*.txt`` fragment of a case into one record.

    Restarted runs leave the pre-restart history in ``control_partN.txt``, so a
    case that was extended must be reassembled before its late-time slope means
    anything.  Columns are ``t, tstar, U, U/V_TC``.
    """
    frags = sorted(glob.glob(os.path.join(case_dir, "**", "control*.txt"), recursive=True))
    tstar, ustar = [], []
    for path in frags:
        try:
            block = np.loadtxt(path)
        except (ValueError, StopIteration):
            continue
        if block.size == 0:
            continue
        if block.ndim == 1:
            block = block[None, :]
        if block.shape[1] < 4:
            continue
        tstar.append(block[:, 1])
        ustar.append(block[:, 3])
    if not tstar:
        raise FileNotFoundError(f"no usable control record under {case_dir}")
    return dedupe_monotone(np.concatenate(tstar), np.concatenate(ustar))


def dedupe_monotone(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by ``x`` and drop repeated abscissae, keeping the first of each.

    Needed because the frame times written by a fixed-cadence solver can repeat
    once the timestep falls below the output interval — ``abds_cn_ustar.txt``
    carries 178 such repeats. ``np.gradient`` divides by the spacing, so a
    duplicate turns the slope into ``inf`` and quietly poisons everything
    downstream rather than raising.
    """
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.concatenate([[True], np.diff(x) > 0])
    return x[keep], y[keep]


def local_slope(T: np.ndarray, deficit: np.ndarray, window: int = 5) -> np.ndarray:
    """Smoothed ``d ln(deficit) / d ln T``.

    The smoothing window is deliberately short.  The output cadence is
    geometric, so a handful of points already spans a useful fraction of a
    decade, and a longer window would flatten the very feature being measured.
    """
    slope = np.gradient(np.log(deficit), np.log(T))
    if window > 1 and len(slope) >= window:
        slope = np.convolve(slope, np.ones(window) / window, mode="same")
    return slope


def profile(t_star: np.ndarray, u_star: np.ndarray, t_min: float = 0.5) -> dict:
    """Reduce one velocity record to its slope history and extremal slope."""
    T = t_star / TSTAR_PER_T
    deficit = 1.0 - u_star
    good = (T > t_min) & (deficit > 0)
    T, deficit = T[good], deficit[good]
    slope = local_slope(T, deficit)
    # Ignore the ends, where the one-sided gradient and the smoothing window are
    # both unreliable, when locating the steepest point.
    interior = slice(3, -3)
    idx = np.argmin(slope[interior]) + 3 if len(slope) > 8 else int(np.argmin(slope))
    return {
        "T": T,
        "deficit": deficit,
        "slope": slope,
        "T_steepest": float(T[idx]),
        "slope_steepest": float(slope[idx]),
        "T_max": float(T[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, help="root holding the Oh_SB_* case directories")
    ap.add_argument("--reference", help="two-column T, u* record from the L=inf reduced solver")
    ap.add_argument("--t-min", type=float, default=0.5, help="ignore T below this")
    ap.add_argument("--gap", type=float, default=0.10,
                    help="slope difference from the reference that counts as departure")
    args = ap.parse_args()

    cases = sorted(glob.glob(os.path.join(args.sweep, "Oh_SB_*")))
    rows = []
    for case in cases:
        if not os.path.isdir(case):
            continue
        tag = os.path.basename(case).replace("Oh_SB_", "")
        oh = float(tag.split("_")[0])
        try:
            t, u = load_control(case)
        except FileNotFoundError:
            print(f"  skipping {tag}: no control record")
            continue
        p = profile(t, u, args.t_min)
        rows.append((oh, p))

    print(f"{'Oh_SB':>9} {'Oh_p':>7} {'T_max':>9} {'steepest n':>11} {'at T':>8} "
          f"{'window T<<Oh_p':>15}")
    for oh, p in sorted(rows):
        print(f"{oh:9.5f} {2 * oh:7.3f} {p['T_max']:9.2f} {p['slope_steepest']:11.3f} "
              f"{p['T_steepest']:8.2f} {2 * oh:15.2f}")

    if args.reference:
        ref = np.loadtxt(args.reference)
        ref_T, ref_u = dedupe_monotone(ref[:, 0], ref[:, 1])
        rp = profile(ref_T * TSTAR_PER_T, ref_u, t_min=0.03)
        print(f"\nreduced model, L = inf: record to T = {rp['T_max']:.1f}, "
              f"n = {rp['slope_steepest']:.3f} at its right edge")
        print("Note n -> -2 only as T -> inf; for a deficit A/(1+T)^2 the local")
        print("slope is -2T/(1+T) exactly, so a 2D case must be compared with")
        print("this curve at matched T rather than against -2.\n")

        print(f"{'Oh_SB':>9} {'Oh_p':>7} {'T_start':>8} {'T_dep':>7} {'T_dep/Oh_p':>11}")
        fit_x, fit_y = [], []
        for oh, p in sorted(rows):
            # A case whose record begins after the departure cannot measure it;
            # reporting the first sample instead would fabricate a boundary.
            if p["T"][0] > 0.25:
                print(f"{oh:9.5f} {2 * oh:7.3f} {p['T'][0]:8.3f}  record starts too late")
                continue
            grid = np.geomspace(0.25, min(p["T"][-1], rp["T"][-1]), 3000)
            gap = np.abs(np.interp(grid, p["T"], p["slope"])
                         - np.interp(grid, rp["T"], rp["slope"]))
            past = np.where(gap > args.gap)[0]
            if not len(past):
                print(f"{oh:9.5f} {2 * oh:7.3f} {p['T'][0]:8.3f}  never departs")
                continue
            t_dep = grid[past[0]]
            print(f"{oh:9.5f} {2 * oh:7.3f} {p['T'][0]:8.3f} {t_dep:7.2f} "
                  f"{t_dep / (2 * oh):11.3f}")
            fit_x.append(2 * oh)
            fit_y.append(t_dep)

        if len(fit_x) >= 2:
            slope, intercept = np.polyfit(np.log(fit_x), np.log(fit_y), 1)
            print(f"\n  T_dep = {np.exp(intercept):.3f} * Oh_p^{slope:.2f} "
                  f"over {len(fit_x)} cases")
            print(f"  a decade of long-wave validity (T_dep = 10) needs "
                  f"Oh_p = {np.exp((np.log(10) - intercept) / slope):.0f}")


if __name__ == "__main__":
    main()
