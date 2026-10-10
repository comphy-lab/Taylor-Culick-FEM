"""Summary figure for the completed planar validation cases.

Three panels, each answering one question the campaign was set up to ask:

  (a) DOMAIN CONVERGENCE -- is the answer numerically converged?
      u*(t*=30) against the co-moving window length L at Oh = 1, with the
      digitised Savva & Bush value marked. This is the panel that shows the
      window, not the mesh, was the controlling parameter.

  (b) AGAINST THE LITERATURE -- our converged FEM and our independent 1D
      Trouton solve, both against the digitised Savva & Bush curves.

  (c) APPROACH TO V_TC -- 1 - u* on log-log, for the viscous and inertial
      cases, with fitted power laws. This is the panel that shows the speed is
      NOT steady at the times reached.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = Path(".")  # set from --results in main()


def load(tag: str):
    for sub in ("run_validation", "planar_comoving"):
        f = RUNS / tag / sub / "control.txt"
        if f.exists():
            a = np.loadtxt(f, comments="#")
            if a.ndim == 2 and len(a) > 5:
                return a
    return None


def main() -> None:
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="run root holding the run_validation.py case folders")
    ap.add_argument("--out", type=Path, default=Path("campaign_summary.png"),
                    help="output figure path")
    args = ap.parse_args()
    RUNS = args.results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 8))

    # ---- (a) domain convergence ------------------------------------------
    series = [(12, "Oh_SB_1_tol1e4"), (16, "Oh_SB_1_L16"), (25, "Oh_SB_1_L25"),
              (40, "Oh_SB_1_L40"), (60, "Oh_SB_1_L60")]
    Ls, us = [], []
    for L, tag in series:
        a = load(tag)
        if a is None:
            continue
        Ls.append(L)
        us.append(float(np.interp(30.0, a[:, 1], a[:, 3])))
    ax1.plot(Ls, us, "o-", ms=14, lw=3, color="#0072B2", label="this code (FEM)")
    ref = np.loadtxt(DATA / "savva_bush_2009_fig6_Oh1.csv", delimiter=",", comments="#")
    sb = float(np.interp(30.0, ref[:, 0], ref[:, 1]))
    ax1.axhline(sb, color="k", ls="--", lw=2.5, label=rf"Savva \& Bush ({sb:.4f})")
    ax1.axhline(1.0, color="0.5", ls=":", lw=2, label=r"$V_{\mathrm{TC}}$")
    ax1.set_xlabel(r"co-moving window length $L/h_0$", fontsize=24, labelpad=8)
    ax1.set_ylabel(r"$u^*(t^*=30)$", fontsize=24, labelpad=8)
    ax1.set_title(r"(a) domain convergence, $Oh=1$", fontsize=22, pad=10)
    ax1.set_ylim(0.78, 1.02)
    ax1.legend(fontsize=15, loc="lower right")
    if len(us) >= 2:
        ax1.annotate(f"converged\n{us[-1]:.5f}", xy=(Ls[-1], us[-1]),
                     xytext=(Ls[-1] - 22, us[-1] - 0.075), fontsize=16,
                     arrowprops=dict(arrowstyle="->", lw=2))

    # ---- (b) against the literature --------------------------------------
    colors = {0.5: "#0072B2", 1.0: "#D55E00", 5.0: "#009E73", 10.0: "#CC79A7",
              100.0: "#444444"}
    for oh, tag in [(1.0, "Oh_SB_1_L60")]:
        a = load(tag)
        if a is not None:
            ax2.plot(a[:, 1], a[:, 3], "-", lw=3.5, color=colors[oh],
                     label=rf"FEM (2D NS) $Oh={oh:g}$")
    for f in sorted(glob.glob(str(DATA / "trouton1d_Oh*.csv"))):
        oh = float(os.path.basename(f).split("Oh")[1].replace(".csv", ""))
        d = np.loadtxt(f, delimiter=",", comments="#")
        ax2.plot(d[:, 0], d[:, 1], "-.", lw=2.5, color=colors.get(oh, "0.4"),
                 label=rf"1D Trouton $Oh={oh:g}$")
    for oh in (0.5, 1.0, 10.0, 100.0):
        p = DATA / f"savva_bush_2009_fig6_Oh{oh:g}.csv"
        if p.exists():
            r = np.loadtxt(p, delimiter=",", comments="#")
            ax2.plot(r[:, 0], r[:, 1], "--", lw=2, color=colors.get(oh, "0.4"),
                     alpha=0.8, label=rf"S\&B $Oh={oh:g}$")
    ax2.axhline(1.0, color="k", ls=":", lw=2)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$t^*=t/\tau_{\mathrm{vis}}$", fontsize=24, labelpad=8)
    ax2.set_ylabel(r"$u^*=u/u_c$", fontsize=24, labelpad=8)
    ax2.set_title("(b) DNS vs reduced-order vs literature", fontsize=22, pad=10)
    ax2.legend(fontsize=12, loc="lower right", ncol=2, framealpha=0.95)

    # ---- (c) approach to V_TC --------------------------------------------
    for tag, lab, c, clock in [
        ("Oh_SB_1_t100_L60", r"$Oh=1$ (viscous clock $t^*$)", "#D55E00", 1),
        ("Oh_SB_0.035_inertial_log", r"$Oh=0.035$ (capillary clock $t$)",
         "#009E73", 0),
    ]:
        a = load(tag)
        if a is None:
            continue
        x = a[:, 1] if clock else a[:, 0]
        y = 1.0 - a[:, 3]
        m = (x > (3 if clock == 0 else 5)) & (y > 0)
        ax3.loglog(x[m], y[m], "-", lw=3, color=c, label=lab)
        p = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
        ax3.loglog(x[m], np.exp(p[1]) * x[m] ** p[0], ":", lw=2.5, color=c,
                   label=rf"  fit $\propto t^{{{p[0]:.2f}}}$")
    ax3.set_xlabel(r"time on its own clock", fontsize=24, labelpad=8)
    ax3.set_ylabel(r"$1-u^*$", fontsize=24, labelpad=8)
    ax3.set_title(r"(c) approach to $V_{\mathrm{TC}}$ -- not steady", fontsize=22, pad=10)
    ax3.legend(fontsize=14, loc="lower left")

    for ax in (ax1, ax2, ax3):
        ax.tick_params(which="both", direction="out", width=2, labelsize=19)
        for sp in ax.spines.values():
            sp.set_linewidth(2)
        ax.minorticks_on()

    fig.tight_layout()
    out = args.out
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  converged u*(t*=30) = {us[-1]:.5f} at L={Ls[-1]}  vs S&B {sb:.4f} "
          f"({abs(us[-1]-sb)/sb*100:.2f}%)")


if __name__ == "__main__":
    main()
