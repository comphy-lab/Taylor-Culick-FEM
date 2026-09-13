"""Animate the planar co-moving Taylor-Culick run.

Two panels, one row:

  left   the rim in the co-moving frame. The simulation solves the half sheet
         above the midplane, so the panel mirrors it: velocity magnitude in the
         upper half, viscous dissipation rate per unit volume in the lower half,
         with the free surface drawn over both.
  right  the recovered control velocity U(t)/V_TC, drawn from 0 to the current
         time only, against the Taylor-Culick target.

Dissipation is not written by the solver, so it is reconstructed here from the
velocity field:

    phi = 2 mu E:E,   E = sym(grad u),   E:E = exx^2 + eyy^2 + 2 exy^2

Each quadratic (``triangle6``) element is split into its four linear
sub-triangles, on which the velocity gradient is exact for a P1 field, giving a
piecewise-constant dissipation. That is a visualisation-grade reconstruction,
not a quantitative one -- it is one order lower than the P2 velocity field.

Frame rendering uses mathtext rather than usetex: LaTeX spawns a subprocess per
unique string and deadlocks under multiprocessing.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

# Parallel-safe typography (see module docstring).
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

V_TC = np.sqrt(2.0)

# Interface overlay colour. Absent from both viridis and hot_r, so it reads
# against either without needing an outline that would obscure the near-surface
# boundary layer.
INTERFACE_COLOR = "#FF00C8"

# VTK triangle6 node order: 0,1,2 vertices; 3=mid(0,1), 4=mid(1,2), 5=mid(2,0).
SUBTRIS = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def read_pvd(pvd: Path) -> list[tuple[float, Path]]:
    """Return (time, vtu path) pairs from a ParaView collection file."""
    root = ET.parse(pvd).getroot()
    out = []
    for ds in root.iter("DataSet"):
        out.append((float(ds.attrib["timestep"]), pvd.parent / ds.attrib["file"]))
    out.sort(key=lambda p: p[0])
    return out


def load_frame(path: Path):
    """Load a frame and build its linear sub-triangulation plus dissipation."""
    m = meshio.read(path)
    pts = m.points[:, :2]
    tri6 = np.vstack([c.data for c in m.cells if c.type == "triangle6"])
    vel = m.point_data["velocity"][:, :2]

    # Split each P2 triangle into four P1 sub-triangles.
    sub = tri6[:, SUBTRIS].reshape(-1, 3)

    # Exact P1 gradient on each sub-triangle.
    p0, p1, p2 = pts[sub[:, 0]], pts[sub[:, 1]], pts[sub[:, 2]]
    det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    det = np.where(np.abs(det) < 1e-300, np.nan, det)

    def grad(f):
        f0, f1, f2 = f[sub[:, 0]], f[sub[:, 1]], f[sub[:, 2]]
        dfdx = ((f1 - f0) * (p2[:, 1] - p0[:, 1]) - (f2 - f0) * (p1[:, 1] - p0[:, 1])) / det
        dfdy = ((f2 - f0) * (p1[:, 0] - p0[:, 0]) - (f1 - f0) * (p2[:, 0] - p0[:, 0])) / det
        return dfdx, dfdy

    dudx, dudy = grad(vel[:, 0])
    dvdx, dvdy = grad(vel[:, 1])
    exy = 0.5 * (dudy + dvdx)
    phi = 2.0 * (dudx**2 + dvdy**2 + 2.0 * exy**2)  # times mu, applied by caller
    # The velocity is returned as a vector, not a magnitude: the caller shifts it
    # into the laboratory frame before taking the norm.
    return pts, sub, vel, np.nan_to_num(phi)


def free_surface(pts, sub, x_max) -> list[np.ndarray]:
    """Boundary edges that are neither the midplane nor the inlet."""
    e = np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, idx, cnt = np.unique(e, axis=0, return_index=True, return_counts=True)
    bnd = e[idx[cnt == 1]]
    keep = []
    for a, b in bnd:
        ya, yb = pts[a, 1], pts[b, 1]
        xa, xb = pts[a, 0], pts[b, 0]
        if max(ya, yb) < 1e-9:  # midplane symmetry
            continue
        if min(xa, xb) > x_max - 1e-6:  # inlet
            continue
        keep.append(np.array([[xa, ya], [xb, yb]]))
    return keep


DPI = 110


def _even_inches(inches: float, dpi: int = DPI) -> float:
    """Round a figure dimension up to an EVEN pixel count.

    Deriving the canvas from the data aspect gives arbitrary sizes, and an odd
    pixel dimension makes libx264 with yuv420p fail outright -- chroma is
    subsampled 2x2, so both dimensions must be even. The frames render fine and
    only the encode dies, with a bare broken pipe that says nothing about the
    cause.
    """
    px = int(round(inches * dpi))
    return (px + (px % 2)) / dpi


def render(
    item, *, outdir: Path, mu: float, xlim, ylim, phi_lim, hist, hist_abs, x_max: float
) -> None:
    i, (t, path) = item
    dest = outdir / f"frame_{i:05d}.png"
    if dest.exists():
        return

    pts, sub, vel, phi = load_frame(path)
    phi = mu * phi

    # Laboratory frame: the solver works in a frame translating at +U e_x, so
    # u_lab = u_frame + U e_x. The undisturbed film is then at rest, and the rim
    # travels at the retraction speed -- which tends to V_TC.
    th_all, u_all = hist_abs
    u_now = float(np.interp(t, th_all, u_all))
    vel_lab = vel.copy()
    vel_lab[:, 0] += u_now
    umag = np.linalg.norm(vel_lab, axis=1)

    # Layout is derived from the DATA aspect, not fixed in figure fractions.
    # A fixed wide box plus set_aspect("equal") shrinks the drawn field to a
    # square inside it and pads the remainder with blank canvas -- which is
    # where the old figure lost roughly half its width. Sizing the box to
    # dx:dy means equal aspect fills it exactly, and the colorbars inherit the
    # same width so nothing has to be nudged by hand when the window changes.
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    PH = 7.2                                        # field panel height, inches
    PW = min(max(PH * dx / dy, 3.0), 13.0)          # width at equal aspect
    RW = 6.2                                        # control-trace panel width
    ml, gap, mr = 1.30, 1.70, 0.30                  # margins and inter-panel gap
    # The dissipation label is a nested fraction under a radical and is far
    # taller than the velocity label above, so the bottom margin is not the
    # mirror of the top one.
    mb, mt = 1.45, 0.85                             # room for the outer cbar labels
    cb_h = 0.30
    # Asymmetric on purpose: the gap BELOW the panel carries the x tick labels
    # and the x axis label as well as clearing the colorbar, while the gap above
    # clears nothing but the panel spine. Equal gaps put the lower colorbar on
    # top of the "x/h_0" label.
    gap_lo, gap_hi = 1.20, 0.32

    FW = _even_inches(ml + PW + gap + RW + mr)
    FH = _even_inches(mb + cb_h + gap_lo + PH + gap_hi + cb_h + mt)
    y0 = (mb + cb_h + gap_lo) / FH                  # shared bottom of both panels

    fig = plt.figure(figsize=(FW, FH))
    axL = fig.add_axes([ml / FW, y0, PW / FW, PH / FH])
    cax_e = fig.add_axes([ml / FW, mb / FH, PW / FW, cb_h / FH])
    cax_u = fig.add_axes([ml / FW, (mb + cb_h + gap_lo + PH + gap_hi) / FH,
                          PW / FW, cb_h / FH])
    axR = fig.add_axes([(ml + PW + gap) / FW, y0, RW / FW, PH / FH])

    # ---- left: mirrored field panel --------------------------------------
    tri_up = Triangulation(pts[:, 0], pts[:, 1], sub)
    tri_lo = Triangulation(pts[:, 0], -pts[:, 1], sub)

    # Normalised by V_TC, not by the capillary velocity sqrt(sigma/rho h0). The
    # right-hand panel plots V/V_TC, so scaling the field by anything else left
    # the two halves of the same figure in different units -- the colorbar ran
    # to sqrt(2) while the trace ran to 1, for the same physical speed.
    c1 = axL.tripcolor(tri_up, umag / V_TC, cmap="viridis", shading="gouraud",
                       vmin=0.0, vmax=1.0)
    c2 = axL.tripcolor(tri_lo, phi, cmap="hot_r", shading="flat", norm=LogNorm(*phi_lim))

    # The interface is drawn on both halves as a single thin magenta line.
    #
    # Deliberately no halo: a contrasting outline would mask a few pixels either
    # side of the interface, which is exactly where the viscous boundary layer
    # and the dissipation peak sit. Magenta is used because neither colormap
    # contains it -- viridis runs purple/teal/yellow and hot_r white/red/black --
    # so a bare line stays legible against both without hiding anything.
    segs = free_surface(pts, sub, x_max)
    mirrored = [np.column_stack([s[:, 0], -s[:, 1]]) for s in segs]
    axL.add_collection(
        LineCollection(
            segs + mirrored, colors=INTERFACE_COLOR, linewidths=1.8,
            zorder=5, capstyle="round",
        )
    )
    axL.axhline(0.0, color="0.45", lw=1.0, ls=(0, (6, 4)), zorder=6)

    axL.set_xlim(*xlim)
    axL.set_ylim(*ylim)
    axL.set_aspect("equal")
    axL.set_xlabel(r"$x/h_0$", fontsize=28, labelpad=8)
    axL.set_ylabel(r"$y/h_0$", fontsize=28, labelpad=8)
    axL.tick_params(which="both", direction="out", width=2, labelsize=21, pad=5)
    for sp in axL.spines.values():
        sp.set_linewidth(2)

    box = dict(boxstyle="round,pad=0.28", fc="k", ec="none", alpha=0.55)
    axL.text(0.012, 0.90, r"$|u|$", transform=axL.transAxes, fontsize=26,
             color="w", bbox=box, zorder=8)
    axL.text(0.012, 0.045, r"$\varepsilon$", transform=axL.transAxes, fontsize=26,
             color="w", bbox=box, zorder=8)
    # Time lives inside the panel: a title would sit under the top colorbar.
    axL.text(0.985, 0.90, rf"$t = {t:5.2f}$", transform=axL.transAxes, ha="right",
             fontsize=26, color="w", bbox=box, zorder=8)

    cb1 = fig.colorbar(c1, cax=cax_u, orientation="horizontal")
    cb1.set_label(r"$|u|_{\mathrm{lab}}\,/\,V_{\mathrm{TC}}$", fontsize=25, labelpad=12)
    cb1.ax.tick_params(labelsize=20, pad=5)
    cb1.ax.xaxis.set_label_position("top")
    cb1.ax.xaxis.set_ticks_position("top")
    cb1.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb1.set_ticklabels(["0", "0.25", "0.50", "0.75", "1"])

    cb2 = fig.colorbar(c2, cax=cax_e, orientation="horizontal")
    cb2.set_label(r"$\varepsilon\,/\,(\sigma/h_0)\sqrt{\sigma/\rho h_0}$",
                  fontsize=25, labelpad=10)
    cb2.ax.tick_params(labelsize=20, pad=5)

    # ---- right: growing control trace ------------------------------------
    th, uh = hist
    mask = th <= t + 1e-12
    axR.axhline(1.0, color="k", lw=2.5, ls="--", zorder=2)
    axR.text(
        0.98, 1.0 + 0.012, r"$V_{\mathrm{TC}} = \sqrt{2}$", transform=axR.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=24,
    )
    axR.plot(th[mask], uh[mask], color="#c1121f", lw=3.5, zorder=3)
    if mask.any():
        axR.plot(th[mask][-1], uh[mask][-1], "o", ms=14, color="#c1121f",
                 mec="k", mew=2, zorder=4)

    axR.set_xlim(0, th.max())
    axR.set_ylim(0, 1.15)
    axR.set_xlabel(r"$t\,\sqrt{\sigma/\rho h_0^3}$", fontsize=28, labelpad=8)
    axR.set_ylabel(r"$V/V_{\mathrm{TC}}$", fontsize=28, labelpad=8)
    axR.tick_params(which="both", direction="out", width=2, labelsize=21, pad=5)
    axR.tick_params(which="major", length=10)
    axR.tick_params(which="minor", length=5)
    for sp in axR.spines.values():
        sp.set_linewidth(2)
    axR.minorticks_on()

    fig.savefig(dest, dpi=DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvd", required=True, type=Path)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--out", default=Path("taylor_culick_planar.mp4"), type=Path)
    ap.add_argument("--frames-dir", default=Path("frames"), type=Path)
    ap.add_argument("--mu", type=float, default=0.05,
                    help="coded dynamic viscosity mu = sqrt(2) Oh; NOT Oh itself")
    ap.add_argument("--nframes", type=int, default=220)
    ap.add_argument("--fps", type=int, default=25)
    # --xmax-domain is the INLET MARKER, not a view setting: free_surface() uses
    # it to discard the far boundary edge, so it must equal the case's true
    # domain length. The view window is separate, below.
    ap.add_argument("--xmax-domain", type=float, default=12.0)
    # The rim is pinned near x = 0 in the co-moving frame and is the only part
    # worth looking at; everything beyond it is undisturbed film. Framing a
    # 110-unit domain in full would render the rim a few pixels wide.
    ap.add_argument("--xlim", type=float, nargs=2, default=(-1.0, 9.0),
                    help="view window in x; default is 10 units at the rim")
    ap.add_argument("--ylim", type=float, nargs=2, default=(-5.0, 5.0),
                    help="view window in y, mirrored about the midplane")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    entries = read_pvd(args.pvd)
    # Subsample uniformly in time, not in file index: early steps are tiny.
    times = np.array([t for t, _ in entries])
    targets = np.linspace(times.min(), times.max(), args.nframes)
    picks = sorted({int(np.argmin(np.abs(times - tt))) for tt in targets})
    entries = [entries[i] for i in picks]
    print(f"{len(entries)} frames, t = {times.min():.3f} .. {times.max():.3f}", flush=True)

    hist_raw = np.loadtxt(args.control, comments="#")
    # control.txt columns are  0:t  1:tstar  2:U  3:U/V_TC  -- both of these
    # were off by one. The right panel drew raw U under a "V/V_TC" label, and
    # the lab-frame shift added TSTAR to the x velocity instead of U, which at
    # t* = 30 is a factor of 23 and saturated the entire colormap. That is why
    # every rim rendered as flat yellow.
    hist = (hist_raw[:, 0], hist_raw[:, 3])  # t, U/V_TC   (right panel)
    hist_abs = (hist_raw[:, 0], hist_raw[:, 2])  # t, U     (lab-frame shift)

    # Velocity is bounded on [0, sqrt(2)] by construction: V_TC is the physical
    # ceiling for this problem, so a fixed scale is meaningful rather than
    # merely convenient. Only the dissipation range is sampled.
    phi_hi = 0.0
    for _, p in entries[:: max(1, len(entries) // 12)]:
        _, _, _, ph = load_frame(p)
        phi_hi = max(phi_hi, float(np.nanpercentile(args.mu * ph, 99.5)))
    phi_lim = (max(phi_hi * 1e-4, 1e-8), max(phi_hi, 1e-6))
    print(f"dissipation range {phi_lim[0]:.2e} .. {phi_lim[1]:.2e}", flush=True)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    work = partial(
        render,
        outdir=args.frames_dir,
        mu=args.mu,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        phi_lim=phi_lim,
        hist=hist,
        hist_abs=hist_abs,
        x_max=args.xmax_domain,
    )
    with Pool(args.procs) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, list(enumerate(entries)), 1), 1):
            if n % 20 == 0:
                print(f"  rendered {n}/{len(entries)}", flush=True)

    import imageio.v2 as imageio

    files = sorted(args.frames_dir.glob("frame_*.png"))
    print(f"encoding {len(files)} frames -> {args.out}", flush=True)
    with imageio.get_writer(
        args.out, fps=args.fps, codec="libx264", quality=8,
        macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as w:
        for f in files:
            w.append_data(imageio.imread(f))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
