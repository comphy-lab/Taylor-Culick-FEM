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

# t* = t / tau_vis and T = t / t_I are both viscous times; see docs/README.md
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
    t = np.concatenate(tstar)
    u = np.concatenate(ustar)
    order = np.argsort(t)
    t, u = t[order], u[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], u[keep]


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
        p = profile(ref[:, 0] * TSTAR_PER_T, ref[:, 1], args.t_min)
        print(f"\nreduced model, L = inf: steepest n = {p['slope_steepest']:.3f} "
              f"at T = {p['T_steepest']:.2f}, record to T = {p['T_max']:.1f}")
        print("This is the value a 2D case must approach for the long-wave")
        print("description to be doing any work at that Oh.")


if __name__ == "__main__":
    main()
