"""Generate the figures for ``docs/`` from campaign output.

Four figures:

``fig1_schematic``
    The planar co-moving problem: geometry, the global unknown ``U`` and the
    boundary conditions. The sheet OUTLINE is a real computed interface --
    ``Oh_SB = 0.035`` at ``t ~ 15``, which carries a genuine rim and a genuine
    capillary-wave train -- with the annotation drawn over it. An earlier
    version hand-rolled the profile from a quarter ellipse and a Gaussian, and
    it looked like what it was. A drawn fallback remains for the case where no
    run is mounted, but it is the fallback, not the intent.

``fig2_phenomenology``
    Velocity magnitude and viscous dissipation rate at a common
    inertio-capillary time across the Ohnesorge range, mirrored about the
    midplane exactly as in the animations, with the free surface over both.
    Shows the rim-plus-capillary-wave structure at low ``Oh_SB`` giving way to
    a smooth, spread-out thickening as ``Oh_SB`` approaches one -- and, in the
    lower halves, where the energy actually goes.

``fig3_velocity``
    The retraction velocity itself: ``u*(t) = U/V_TC`` for the eight-point
    ``Oh_SB`` sweep against the inertio-capillary clock, which is the primary
    observable of the whole campaign.

``fig4_approach``
    The same data as ``1 - u*`` against the viscous clock ``t*`` on log-log
    axes, where the approach to ``V_TC`` is a power law and the fitted
    exponent is read off per case. This is the figure that shows the exponent
    is NOT constant across the sweep.

Typography is real LaTeX (``text.usetex``), not mathtext. The animations
cannot use it -- LaTeX spawns a subprocess per unique string and deadlocks
under multiprocessing -- but these are rendered serially and the docs should
look like the papers they support.

Run from the repository root with the campaign volumes mounted::

    uv run python postProcess/make_docs_figures.py --out docs/planar-campaign/figures

Simulation output is NOT in this repository. ``--sweep`` and ``--archive``
point at the run roots; any case whose data is missing is skipped with a
warning rather than failing the figure.
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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_planar import free_surface, load_frame, read_pvd  # noqa: E402
from animate_stacked import load_control  # noqa: E402

V_TC = np.sqrt(2.0)

INTERFACE_COLOR = "#FF00C8"

# Ohnesorge numbers are quoted in the Savva & Bush / JFM convention throughout
# the docs. See docs/planar-campaign/README.md; the coded value is a factor sqrt(2) larger.
PANELS = [
    ("0.035", "Oh_SB_0.035_inertial_log", "archive"),
    ("0.2377", "Oh_SB_0.2377_u995", "sweep"),
    ("0.61945", "Oh_SB_0.61945_u995", "sweep"),
    ("1.0", "Oh_SB_1_u995", "sweep"),
]

# The eight-point log-spaced production sweep, in ascending Oh_SB. Oh_SB=0.035
# lost its field output to the 2026-08-08 restart defect but its control
# history survived, so it appears in figures 3 and 4 and not in figure 2.
SWEEP = ["Oh_SB_0.035_u995", "Oh_SB_0.0565_u995", "Oh_SB_0.09121_u995",
         "Oh_SB_0.14724_u995", "Oh_SB_0.2377_u995", "Oh_SB_0.38372_u995",
         "Oh_SB_0.61945_u995", "Oh_SB_1_u995"]

STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {out / (stem + '.png')}")


# ==========================================================================
# figure 1 -- schematic
# ==========================================================================

SCHEMATIC_SOURCE = ("Oh_SB_0.035_inertial_log", "archive", 15.0)


def _drawn_profile(L: float, half: float, rim: float):
    """Fallback sheet outline when no run is mounted.

    Nose is a quarter ellipse from the pinned tip to the apex, arriving with
    zero slope; tail is a Gaussian relaxation from the apex to the film, which
    also has zero slope at the apex, so the two meet without a kink. A decaying
    ripple stands in for the capillary-wave train.
    """
    phi = np.linspace(0.0, np.pi / 2, 200)
    xn = rim * (1.0 - np.cos(phi))
    yn = rim * np.sin(phi)
    xt = np.linspace(rim, L, 500)
    s = xt - rim
    yt = half + (rim - half) * np.exp(-(s / (1.35 * rim)) ** 2)
    yt = yt + 0.16 * np.exp(-s / 2.6) * np.sin(1.9 * s)
    return np.concatenate([xn, xt]), np.concatenate([yn, yt])


def _measured_profile(case_dir: Path, t_target: float, x_max_draw: float):
    """Ordered upper-half interface polyline from a real frame.

    ``free_surface`` returns unordered edges. Sorting them by ``x`` is enough
    here because the free surface of a retracting sheet is single-valued in
    ``x`` everywhere except across the nose, and the nose is picked up by the
    same sort as a near-vertical run of points at small ``x``.
    """
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
    if not segs:
        return None
    node = np.vstack([np.vstack(s) for s in segs])
    node = np.unique(np.round(node, 9), axis=0)
    node = node[node[:, 0] <= x_max_draw]
    node = node[np.argsort(node[:, 0])]
    return node[:, 0], node[:, 1]


def fig1_schematic(out: Path, sources: dict[str, Path]) -> None:
    """Geometry, co-moving frame and boundary conditions."""
    case, src, t_want = SCHEMATIC_SOURCE
    L = 16.0                       # drawn window length, deliberately not to scale
    got = _measured_profile(sources[src] / case, t_want, L)
    if got is None:
        print(f"  NOTE: {case} not mounted; figure 1 uses the drawn fallback",
              file=sys.stderr)
        xs, ys = _drawn_profile(L, 0.5, 2.5)
    else:
        xs, ys = got

    rim = float(ys.max())
    half = float(np.median(ys[xs > 0.75 * L]))

    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    fig.patch.set_facecolor("white")

    ax.fill_between(xs, -ys, ys, color="#bfe3ea", zorder=1)
    ax.plot(xs, ys, color="#12343b", lw=2.2, zorder=3, solid_capstyle="round")
    ax.plot(xs, -ys, color="#12343b", lw=2.2, zorder=3, solid_capstyle="round")

    # Midplane symmetry line, labelled with a leader so nothing sits on the sheet.
    ax.plot([0, L], [0, 0], color="0.45", lw=1.0, ls=(0, (7, 5)), zorder=5)
    ax.annotate(r"midplane symmetry", xy=(L * 0.55, 0.0),
                xytext=(L * 0.60, -rim - 1.55), fontsize=8.5, color="0.35",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))

    # Co-moving window.
    top = rim + 0.75
    ax.add_patch(Rectangle((0, -top), L, 2 * top, fill=False, ec="0.6", lw=1.0,
                           ls=(0, (2, 2)), zorder=4))
    ax.annotate("", xy=(0, top + 0.85), xytext=(L, top + 0.85),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.1))
    ax.text(L / 2, top + 1.0, r"co-moving window $L$", ha="center", va="bottom",
            fontsize=9, color="0.3")

    # Far-field inflow at -U, imposed through a Lagrange multiplier. The
    # arrows span the FILM only: the inlet is a liquid boundary, and the
    # earlier version drew them across the full window height, through
    # several thicknesses of nothing.
    for yy in np.linspace(-0.62 * half, 0.62 * half, 3):
        ax.add_patch(FancyArrowPatch((L + 2.1, yy), (L + 0.18, yy),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color="#8a1c1c", lw=1.3))
    ax.text(L + 2.4, 0.0, r"$u_x = -U$" "\n" r"\texttt{EnforcedDirichlet}",
            fontsize=8.5, color="#8a1c1c", va="center", ha="left")

    # Pinned tip: the constraint that defines U.
    ax.plot([0], [0], marker="o", ms=6.5, color="#8a1c1c", zorder=6)
    ax.annotate(r"tip pinned $\Rightarrow$ global unknown $U(t)$",
                xy=(0.0, -0.10), xytext=(0.15, -rim - 1.55), fontsize=9,
                ha="left", va="top", color="#8a1c1c",
                arrowprops=dict(arrowstyle="->", color="#8a1c1c", lw=1.1))

    # Thickness annotation, on the flat film clear of everything else. The
    # label goes ABOVE the film, not beside the arrow: the film is one unit
    # thick here and the midplane dash runs straight through the glyph.
    xa = L * 0.78
    ax.annotate("", xy=(xa, -half), xytext=(xa, half),
                arrowprops=dict(arrowstyle="<->", color="#12343b", lw=1.2))
    ax.text(xa, half + 0.30, r"$h_0$", fontsize=10.5, va="bottom", ha="center",
            color="#12343b")

    ax.text(0.15, rim + 0.30, r"free surface: $\sigma$, no gas phase",
            fontsize=9, color="#12343b", va="bottom")
    ax.annotate(r"capillary waves (low $Oh$)",
                xy=(2.55 * rim, half + 0.22), xytext=(2.9 * rim, rim * 0.80),
                fontsize=8.5, color="#12343b",
                arrowprops=dict(arrowstyle="->", color="#12343b", lw=1.0))

    ax.set_xlim(-1.0, L + 6.0)
    ax.set_ylim(-rim - 2.6, rim + 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, out, "fig1_schematic")


# ==========================================================================
# figure 2 -- phenomenology, now with the fields
# ==========================================================================

def _frame_at(case_dir: Path, t_target: float):
    """Nearest frame to ``t_target``: points, sub-triangles, velocity, phi."""
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
    pts, sub, vel, phi = load_frame(path)
    return t, pts, sub, vel, phi, meta


def _u_at(case_dir: Path, t: float) -> float:
    """Frame velocity at time ``t``, for the shift into the laboratory frame."""
    try:
        h = load_control(case_dir)
    except SystemExit:
        return 0.0
    return float(np.interp(t, h[:, 0], h[:, 2]))


def fig2_phenomenology(out: Path, sources: dict[str, Path], t_target: float,
                       x_max: float = 26.0, y_max: float = 4.6) -> None:
    """Velocity and dissipation across Oh at a common inertio-capillary time.

    Both colour scales are SHARED across the panels. That is the whole point:
    the dissipation rate at ``Oh_SB = 1`` is orders above that at 0.035, and a
    per-panel rescale would hide the one comparison the figure exists to make.
    """
    loaded = []
    for label, case, src in PANELS:
        got = _frame_at(sources[src] / case, t_target)
        if got is None:
            print(f"  WARNING: no data for {case}", file=sys.stderr)
            loaded.append((label, case, None))
            continue
        t, pts, sub, vel, phi, meta = got
        mu = float(meta.get("mu_coded", np.sqrt(2.0) * float(label)))
        u_now = _u_at(sources[src] / case, t)
        vel_lab = vel.copy()
        vel_lab[:, 0] += u_now
        umag = np.linalg.norm(vel_lab, axis=1) / V_TC
        loaded.append((label, case, (t, pts, sub, umag, mu * phi,
                                     float(meta.get("domain_length",
                                                    pts[:, 0].max())))))

    live = [d for _, _, d in loaded if d is not None]
    if not live:
        print("  WARNING: no case data found; figure 2 is a placeholder",
              file=sys.stderr)
        phi_lim = (1e-6, 1e-2)
    else:
        hi = max(float(np.nanpercentile(d[4], 99.8)) for d in live)
        phi_lim = (hi * 1e-5, hi)

    n = len(PANELS)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.30 * n + 1.35), sharex=True)
    c1 = c2 = None

    for ax, (label, case, data) in zip(axes, loaded):
        ax.set_ylabel(rf"$Oh={label}$", fontsize=9)
        if data is None:
            ax.text(0.5, 0.5, rf"data unavailable: \texttt{{{case}}}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="0.5")
            ax.set_xlim(0, x_max)
            ax.set_ylim(-y_max, y_max)
            continue
        t, pts, sub, umag, phi, L0 = data
        tri_up = Triangulation(pts[:, 0], pts[:, 1], sub)
        tri_lo = Triangulation(pts[:, 0], -pts[:, 1], sub)
        c1 = ax.tripcolor(tri_up, umag, cmap="viridis", shading="gouraud",
                          vmin=0.0, vmax=1.0, rasterized=True)
        c2 = ax.tripcolor(tri_lo, phi, cmap="hot_r", shading="flat",
                          norm=LogNorm(*phi_lim), rasterized=True)
        segs = free_surface(pts, sub, L0)
        mirrored = [s * np.array([1, -1]) for s in segs]
        ax.add_collection(LineCollection(segs + mirrored,
                                         colors=INTERFACE_COLOR,
                                         linewidths=0.9, zorder=5))
        ax.axhline(0, color="0.45", lw=0.6, ls=(0, (5, 4)), zorder=6)
        ax.set_xlim(0, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_aspect("equal")
        ax.text(0.992, 0.86, rf"$t={t:.1f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="w",
                bbox=dict(boxstyle="round,pad=0.2", fc="k", ec="none",
                          alpha=0.55))

    axes[-1].set_xlabel(r"distance from the pinned tip, $x/h_0$")
    fig.align_ylabels(axes)

    if c1 is not None:
        fig.subplots_adjust(bottom=0.19, top=0.985, hspace=0.10)
        cax_u = fig.add_axes([0.13, 0.105, 0.34, 0.017])
        cax_e = fig.add_axes([0.55, 0.105, 0.34, 0.017])
        cb1 = fig.colorbar(c1, cax=cax_u, orientation="horizontal")
        cb1.set_label(r"$|u|_{\mathrm{lab}}/V_{\mathrm{TC}}$ \quad (upper half)",
                      fontsize=8)
        cb1.ax.tick_params(labelsize=7)
        cb2 = fig.colorbar(c2, cax=cax_e, orientation="horizontal")
        cb2.set_label(r"$\varepsilon\,/\,(\sigma/h_0)\sqrt{\sigma/\rho h_0}$"
                      r" \quad (lower half)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    _save(fig, out, "fig2_phenomenology")


# ==========================================================================
# figures 3 and 4 -- the retraction velocity
# ==========================================================================

def _sweep_histories(sweep: Path):
    """``(Oh_SB, t, u*, tstar)`` per mounted sweep case, ascending in Oh."""
    out = []
    for case in SWEEP:
        d = sweep / case
        meta_path = d / "case.json"
        if not meta_path.exists():
            print(f"  WARNING: no case.json for {case}", file=sys.stderr)
            continue
        try:
            h = load_control(d)
        except SystemExit:
            print(f"  WARNING: no control history for {case}", file=sys.stderr)
            continue
        oh = float(json.loads(meta_path.read_text())["Oh_savva_bush"])
        out.append((oh, h[:, 0], h[:, 3], h[:, 1]))
    out.sort(key=lambda r: r[0])
    return out


def _colours(n: int):
    return plt.cm.viridis(np.linspace(0.05, 0.85, n))


def fig3_velocity(out: Path, sweep: Path) -> None:
    """``u*(t)`` for the sweep against the inertio-capillary clock."""
    hist = _sweep_histories(sweep)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    if not hist:
        ax.text(0.5, 0.5, "no sweep data mounted", transform=ax.transAxes,
                ha="center", va="center", color="0.5")
    for (oh, t, u, _), col in zip(hist, _colours(len(hist))):
        m = t > 0
        ax.plot(t[m], u[m], color=col, lw=1.6,
                label=rf"${oh:g}$" + rf"\ \ ({u.max():.4f})")
    ax.axhline(1.0, color="k", lw=1.2, ls="--", zorder=1)
    ax.text(0.015, 1.0, r"$V_{\mathrm{TC}}$", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, None)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"$t\,\sqrt{\sigma/\rho h_0^3}$")
    ax.set_ylabel(r"$u^{*} = U/V_{\mathrm{TC}}$")
    ax.minorticks_on()
    leg = ax.legend(title=r"$Oh_{SB}$\ \ (final $u^{*}$)", fontsize=7.5,
                    title_fontsize=8, loc="lower right", frameon=True,
                    framealpha=0.95, ncol=2, handlelength=1.4,
                    columnspacing=1.0)
    leg.get_frame().set_linewidth(0.7)
    _save(fig, out, "fig3_velocity")


def fig4_approach(out: Path, sweep: Path) -> None:
    """``1 - u*`` against the viscous clock, where the approach is a power law.

    The exponent is fitted on the LAST DECADE of each record, not on a common
    absolute ``t*`` window. Fitting the whole record folds in the early
    inertial transient, which is not a power law and drags the exponent toward
    zero; fixing an absolute window instead excludes the two highest-``Oh``
    cases entirely, since their viscous clock is fast and they never reach it.
    """
    hist = _sweep_histories(sweep)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    if not hist:
        ax.text(0.5, 0.5, "no sweep data mounted", transform=ax.transAxes,
                ha="center", va="center", color="0.5")
    for (oh, _, u, tstar), col in zip(hist, _colours(len(hist))):
        gap = 1.0 - u
        m = (tstar > 0) & (gap > 0)
        ts, gp = tstar[m], gap[m]
        ax.plot(ts, gp, color=col, lw=1.6)
        fit = ts >= ts.max() / 10.0
        if fit.sum() >= 8:
            a, _b = np.polyfit(np.log10(ts[fit]), np.log10(gp[fit]), 1)
            ax.plot(ts[fit], 10 ** _b * ts[fit] ** a, color=col, lw=0.9,
                    ls=(0, (4, 3)), zorder=1)
            lbl = rf"${oh:g}$: $a={a:.3f}$"
        else:
            lbl = rf"${oh:g}$"
        ax.plot([], [], color=col, lw=1.6, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t^{*} = t/\tau_{\mathrm{vis}}$")
    ax.set_ylabel(r"$1 - u^{*}$")
    # Upper right: the curves run down the diagonal and the lower left is
    # where the highest-Oh cases end, so a legend there sits on the data.
    leg = ax.legend(title=r"$Oh_{SB}$: fitted $1-u^{*}\propto t^{*\,a}$",
                    fontsize=7.5, title_fontsize=8, loc="upper right",
                    frameon=True, framealpha=0.95, ncol=2, handlelength=1.4,
                    columnspacing=1.0)
    leg.get_frame().set_linewidth(0.7)
    _save(fig, out, "fig4_approach")


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/planar-campaign/figures"))
    ap.add_argument("--sweep", type=Path,
                    default=Path("/mnt/stokesInt1/running-simulations/"
                                 "elastic-taylor-culick/u995-sweep"))
    ap.add_argument("--archive", type=Path,
                    default=Path("/home/vatsal/running-simulations/"
                                 "elastic-taylor-culick/"
                                 "newtonian-comoving-pyoomph/validation"))
    ap.add_argument("--time", type=float, default=25.0,
                    help="inertio-capillary time at which to compare fields")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of 1 2 3 4; default is all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ["1", "2", "3", "4"])
    sources = {"sweep": args.sweep, "archive": args.archive}

    with plt.rc_context(STYLE):
        if "1" in want:
            fig1_schematic(args.out, sources)
        if "2" in want:
            fig2_phenomenology(args.out, sources, args.time)
        if "3" in want:
            fig3_velocity(args.out, args.sweep)
        if "4" in want:
            fig4_approach(args.out, args.sweep)


if __name__ == "__main__":
    main()
