"""Compare the simulated edge velocity against the digitised Savva & Bush curves.

Produces a two-panel figure matching the source figures (early transient and
long time) and a quantitative error table. The comparison is over the whole
`u*(t*)` curve, not a single point: a single agreeing point is weak evidence,
whereas a curve across Ohnesorge number can actually falsify the formulation.

Errors are reported against the digitised reference on the reference's own
`t*` samples, restricted to the overlap of the two time ranges. The digitisation
uncertainty (recorded in the CSV headers) is a floor on any agreement claim and
is printed alongside, so a 1% "discrepancy" is not read as physics when the
reference itself is only good to 1%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from savva_bush import load_reference  # noqa: E402

COLORS = {0.5: "#0072B2", 1.0: "#D55E00", 5.0: "#009E73", 10.0: "#CC79A7",
          100.0: "#333333"}


def load_runs(results: Path) -> dict[float, tuple[np.ndarray, dict]]:
    """Our control histories, keyed by Savva-Bush Ohnesorge number."""
    out = {}
    for case in sorted(results.glob("Oh_SB_*")):
        meta_f, ctrl = case / "case.json", case / "planar_comoving" / "control.txt"
        if not ctrl.exists():
            ctrl = case / "control.txt"
        if not (meta_f.exists() and ctrl.exists()):
            continue
        meta = json.loads(meta_f.read_text())
        d = np.loadtxt(ctrl, comments="#")
        if d.ndim == 1 or d.shape[1] < 4:
            continue
        out[float(meta["Oh_savva_bush"])] = (d, meta)
    return out


def compare(ref: np.ndarray, sim: np.ndarray) -> tuple[float, float, int]:
    """RMS and max |Delta u*| on the reference samples inside the overlap."""
    t_lo = max(ref[:, 0].min(), sim[:, 1].min())
    t_hi = min(ref[:, 0].max(), sim[:, 1].max())
    m = (ref[:, 0] >= t_lo) & (ref[:, 0] <= t_hi)
    if m.sum() < 3:
        return float("nan"), float("nan"), 0
    interp = np.interp(ref[m, 0], sim[:, 1], sim[:, 3])
    d = interp - ref[m, 1]
    return float(np.sqrt((d**2).mean())), float(np.abs(d).max()), int(m.sum())


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--results", type=Path,
                    default=Path("/home/vatsal/running-simulations/elastic-taylor-culick/"
                                 "newtonian-comoving-pyoomph/validation"))
    ap.add_argument("--out", type=Path, default=here / "data" / "validation_savva_bush.png")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no completed runs under {args.results}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    rows = []
    for ax, (figname, xlim, ylim, title) in zip(
        axes,
        [("fig5", (0, 1.0), (0, 0.7), "early transient (their figure 5)"),
         ("fig6", (0, 30.0), (0, 1.05), "long time (their figure 6)")],
    ):
        ref = load_reference(figname)
        for oh, r in sorted(ref.items()):
            ax.plot(r[:, 0], r[:, 1], "--", lw=2.5, color=COLORS.get(oh, "0.4"),
                    label=rf"S\&B $Oh={oh:g}$" if figname == "fig6" else None)
        for oh, (d, _meta) in sorted(runs.items()):
            ax.plot(d[:, 1], d[:, 3], "-", lw=2.5, color=COLORS.get(oh, "0.4"),
                    label=rf"this code $Oh={oh:g}$" if figname == "fig6" else None)
            if oh in ref:
                rms, mx, n = compare(ref[oh], d)
                rows.append((figname, oh, rms, mx, n))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$t^* = t/\tau_{\mathrm{vis}}$", fontsize=26, labelpad=8)
        ax.set_ylabel(r"$u^* = u/u_c$", fontsize=26, labelpad=8)
        ax.set_title(title, fontsize=22, pad=10)
        ax.tick_params(which="both", direction="out", width=2, labelsize=19)
        for sp in ax.spines.values():
            sp.set_linewidth(2)
        ax.minorticks_on()
    axes[1].axhline(1.0, color="k", lw=1.5, ls=":")
    axes[1].legend(fontsize=15, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    print(f"\n{'figure':>8} {'Oh_SB':>8} {'RMS du*':>10} {'max du*':>10} {'pts':>5}")
    for figname, oh, rms, mx, n in rows:
        print(f"{figname:>8} {oh:>8g} {rms:>10.4f} {mx:>10.4f} {n:>5d}")
    print("\nDigitisation uncertainty is a floor on these numbers; see the CSV headers.")


if __name__ == "__main__":
    main()
