"""Generate the figures for ``docs/`` from campaign output.

Two figures:

``fig1_schematic``
    The planar co-moving problem: geometry, the global unknown ``U`` and the
    boundary conditions. Drawn, not rendered from data.

``fig2_phenomenology``
    Free-surface shapes at a common inertio-capillary time across the Ohnesorge
    range, showing the rim-plus-capillary-wave structure at low ``Oh_SB``
    giving way to a smooth, spread-out thickening as ``Oh_SB`` approaches one.

Run from the repository root with the campaign volumes mounted::

    uv run python postProcess/make_docs_figures.py --out docs/figures

Simulation output is NOT in this repository. ``--sources`` points at the run
roots; the defaults are the Stokes paths used for the campaign, and any case
whose data is missing is skipped with a warning rather than failing the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_planar import free_surface, load_frame, read_pvd  # noqa: E402

V_TC = np.sqrt(2.0)

# Ohnesorge numbers are quoted in the Savva & Bush / JFM convention throughout
# the docs. See docs/README.md; the coded value is a factor sqrt(2) larger.
PANELS = [
    ("0.035", "Oh_SB_0.035_inertial_log", "archive"),
    ("0.2377", "Oh_SB_0.2377_u995", "sweep"),
    ("0.61945", "Oh_SB_0.61945_u995", "sweep"),
    ("1.0", "Oh_SB_1_u995", "sweep"),
]

STYLE = {
    "font.size": 9,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "dejavusans",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}


def _sheet_profile(L: float, half: float, rim: float, waves: bool):
    """Upper half of the sheet: rim cap, optional capillary waves, flat film.

    Proportions follow the computed shapes in figure 2, where the rim stands
    roughly five half-thicknesses proud of the film.
    """
    # Nose: quarter ellipse from the pinned tip (0,0) to the apex (rim, rim),
    # which arrives with zero slope.
    phi = np.linspace(0.0, np.pi / 2, 200)
    xn = rim * (1.0 - np.cos(phi))
    yn = rim * np.sin(phi)
    # Tail: a Gaussian relaxation from the apex down to the film. It also has
    # zero slope at the apex, so nose and tail meet without a kink.
    xt = np.linspace(rim, L, 500)
    s = xt - rim
    yt = half + (rim - half) * np.exp(-(s / (1.35 * rim)) ** 2)
    if waves:
        # Decaying ripple ahead of the neck. Small enough that its non-zero
        # slope at the join is invisible at figure scale.
        yt = yt + 0.16 * np.exp(-s / 2.6) * np.sin(1.9 * s)
    return np.concatenate([xn, xt]), np.concatenate([yn, yt])


def fig1_schematic(out: Path) -> None:
    """Geometry, co-moving frame and boundary conditions."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    L = 13.0          # drawn window length, deliberately not to scale
    half = 0.42       # half-thickness; the sheet is h0 thick in full
    rim = 1.9         # drawn rim half-height

    xs, ys = _sheet_profile(L, half, rim, waves=True)
    ax.fill_between(xs, -ys, ys, color="#cfe0f5", zorder=1)
    ax.plot(xs, ys, color="#1b3b6f", lw=2.0, zorder=3)
    ax.plot(xs, -ys, color="#1b3b6f", lw=2.0, zorder=3)

    # Midplane symmetry line, labelled with a leader so nothing sits on the sheet.
    ax.plot([0, L], [0, 0], color="0.4", lw=1.0, ls=(0, (7, 5)), zorder=5)
    ax.annotate("midplane symmetry", xy=(L * 0.62, 0.0),
                xytext=(L * 0.76, -rim - 1.45), fontsize=8, color="0.4", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))

    # Co-moving window.
    ax.add_patch(Rectangle((0, -rim - 0.5), L, 2 * (rim + 0.5), fill=False,
                           ec="0.6", lw=1.0, ls=(0, (2, 2)), zorder=4))
    ax.annotate("", xy=(0, rim + 1.30), xytext=(L, rim + 1.30),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(L / 2, rim + 1.45, r"co-moving window $L$", ha="center",
            fontsize=8.5, color="0.35")

    # Far-field inflow at -U, imposed through a Lagrange multiplier.
    for yy in np.linspace(-rim + 0.25, rim - 0.25, 6):
        ax.add_patch(FancyArrowPatch((L + 1.5, yy), (L + 0.15, yy),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color="#8a1c1c", lw=1.2))
    ax.text(L + 1.7, 0.0, r"$u_x=-U$" "\n" "EnforcedDirichlet", fontsize=8,
            color="#8a1c1c", va="center")

    # Pinned tip: the constraint that defines U.
    ax.plot([0], [0], marker="o", ms=6.0, color="#8a1c1c", zorder=6)
    ax.annotate(r"tip pinned $\Rightarrow$ global unknown $U(t)$",
                xy=(0.0, -0.12), xytext=(0.1, -rim - 1.45), fontsize=8.5, ha="left",
                color="#8a1c1c",
                arrowprops=dict(arrowstyle="->", color="#8a1c1c", lw=1.0))

    # Thickness annotation, placed on the flat film clear of other labels.
    xa = L * 0.80
    ax.annotate("", xy=(xa, -half), xytext=(xa, half),
                arrowprops=dict(arrowstyle="<->", color="0.15", lw=1.1))
    ax.text(xa + 0.28, 0.0, r"$h_0$", fontsize=9, va="center",
            bbox=dict(fc="white", ec="none", pad=0.8))

    ax.text(0.0, rim + 0.62, r"free surface: $\sigma$, no gas phase",
            fontsize=8.5, color="#1b3b6f")
    ax.annotate("capillary waves (low $Oh$)", xy=(2.7 * rim, half + 0.30),
                xytext=(3.1 * rim, rim * 0.72), fontsize=8, color="#1b3b6f",
                arrowprops=dict(arrowstyle="->", color="#1b3b6f", lw=0.9))

    ax.set_xlim(-1.0, L + 4.6)
    ax.set_ylim(-rim - 2.0, rim + 1.9)
    ax.axis("off")
    fig.savefig(out / "fig1_schematic.png")
    fig.savefig(out / "fig1_schematic.pdf")
    plt.close(fig)
    print(f"  wrote {out/'fig1_schematic.png'}")


def _interface_at(case_dir: Path, t_target: float):
    """Free-surface polyline at the frame nearest ``t_target``."""
    pvd = case_dir / "run_validation" / "liquid.pvd"
    if not pvd.exists():
        return None
    frames = read_pvd(pvd)
    if not frames:
        return None
    t, path = min(frames, key=lambda p: abs(p[0] - t_target))
    if not path.exists():
        return None
    meta_path = case_dir / "case.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pts, sub, _, _ = load_frame(path)
    segs = free_surface(pts, sub, meta.get("domain_length", pts[:, 0].max()))
    return t, segs


def fig2_phenomenology(out: Path, sources: dict[str, Path], t_target: float) -> None:
    """Interface shape across Oh at a common inertio-capillary time."""
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.4, 5.4), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.08, 0.82, len(PANELS)))

    any_data = False
    for ax, (label, case, src), col in zip(axes, PANELS, colours):
        root = sources[src]
        got = _interface_at(root / case, t_target)
        if got is None:
            ax.text(0.5, 0.5, f"data unavailable: {case}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.5")
            ax.set_ylabel(rf"$Oh={label}$", fontsize=8.5)
            continue
        any_data = True
        t, segs = got
        # Mirror about the midplane so the sheet reads as a sheet.
        lc_up = LineCollection(segs, colors=[col], linewidths=1.5)
        lc_dn = LineCollection([s * np.array([1, -1]) for s in segs],
                               colors=[col], linewidths=1.5)
        ax.add_collection(lc_up)
        ax.add_collection(lc_dn)
        xs = np.concatenate([s[:, 0] for s in segs])
        ys = np.concatenate([s[:, 1] for s in segs])
        ax.set_xlim(0, min(xs.max(), 26))
        ax.set_ylim(-1.35 * ys.max(), 1.35 * ys.max())
        ax.set_ylabel(rf"$Oh={label}$", fontsize=8.5)
        ax.text(0.985, 0.86, rf"$t={t:.1f}$", transform=ax.transAxes,
                ha="right", fontsize=7.5, color="0.4")
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)

    if not any_data:
        print("  WARNING: no case data found; figure 2 is a placeholder", file=sys.stderr)
    axes[-1].set_xlabel(r"distance from the pinned tip, $x/h_0$")
    fig.align_ylabels(axes)
    fig.suptitle(
        r"Free surface at a common time; $Oh$ in the Savva \& Bush convention"
        if matplotlib.rcParams["text.usetex"] else
        "Free surface at a common time; Oh in the Savva & Bush convention",
        fontsize=9, y=0.995)
    fig.savefig(out / "fig2_phenomenology.png")
    fig.savefig(out / "fig2_phenomenology.pdf")
    plt.close(fig)
    print(f"  wrote {out/'fig2_phenomenology.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--sweep", type=Path,
                    default=Path("/mnt/stokesInt1/running-simulations/"
                                 "elastic-taylor-culick/u995-sweep"))
    ap.add_argument("--archive", type=Path,
                    default=Path("/run/media/vatsal/stokesExt2/Projects/"
                                 "elastic-taylor-culick/newtonian-comoving-pyoomph/"
                                 "validation"))
    ap.add_argument("--time", type=float, default=25.0,
                    help="inertio-capillary time at which to compare interfaces")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        fig1_schematic(args.out)
        fig2_phenomenology(args.out, {"sweep": args.sweep, "archive": args.archive},
                           args.time)


if __name__ == "__main__":
    main()
