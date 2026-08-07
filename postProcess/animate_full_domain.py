"""Animate the WHOLE co-moving domain, x from 0 to L0.

Companion to ``animate_planar.py``, which frames a fixed window at the rim.
This one shows the entire computational domain, so the two answer different
questions:

  animate_planar      what is the rim doing?
  animate_full_domain is the domain long enough, and is the far field actually
                      undisturbed?

The second question is why this exists. The co-moving formulation is only valid
while the inlet stays quiescent, and the domain-length sweep (u* = 0.819 at
L = 12 rising to 0.950 at L = 40 and 60) says truncation dominated the answer
until L was large enough. That is visible directly here: watch the disturbance
travel toward the inlet and see whether it arrives before the run ends.

One panel, mirrored about the midplane as in the companion script -- velocity
magnitude above, viscous dissipation rate below, free surface over both. The
control trace is deliberately absent: at L = 110 the domain is ~9x wider than
tall, and a square panel bolted to its side would shrink the field to a sliver.

The loaders, the P2 -> P1 dissipation reconstruction and the free-surface edge
extraction are imported from animate_planar rather than copied. That
reconstruction is subtle enough that two divergent copies would be a liability,
and any fix to the physics should reach both animations at once.

Conventions inherited deliberately, each of which was a bug once:

  * velocity is normalised by V_TC, not by the capillary velocity, so a value
    of 1 means "moving at the Taylor-Culick speed" everywhere it appears;
  * the lab-frame shift uses control.txt column 2 (U), not column 1 (tstar);
  * figure dimensions round to a whole macro block, or libx264 fails outright
    on odd sizes and imageio silently rescales anything not divisible by 16.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_planar import (  # noqa: E402
    INTERFACE_COLOR,
    V_TC,
    free_surface,
    load_frame,
    read_pvd,
)

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

DPI = 110


def _block_inches(inches: float, block: int = 16, dpi: int = DPI) -> float:
    """Round a figure dimension up to a whole number of macro blocks.

    Even pixels are enough for libx264 to accept the stream, but imageio then
    resizes anything not divisible by 16 up to the next block and says so only
    in a warning. That rescale is small -- 2306 -> 2320 here -- but it is a
    ~0.6% anisotropic stretch applied after the field has been drawn at equal
    aspect, which is exactly the thing this view is meant to be trusted on.
    """
    px = int(np.ceil(inches * dpi / block) * block)
    return px / dpi


def render(item, *, outdir: Path, mu: float, xlim, ylim, phi_lim, hist_abs,
           x_max: float) -> None:
    i, (t, path) = item
    dest = outdir / f"frame_{i:05d}.png"
    if dest.exists():
        return

    pts, sub, vel, phi = load_frame(path)
    phi = mu * phi

    th, uh = hist_abs
    u_now = float(np.interp(t, th, uh))
    vel_lab = vel.copy()
    vel_lab[:, 0] += u_now                      # co-moving -> laboratory frame
    umag = np.linalg.norm(vel_lab, axis=1) / V_TC

    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

    # Panel width follows the domain aspect but is capped: at L = 110 and
    # y = +-12 the true aspect is over 9:1, and an unclamped canvas would be
    # 60 inches wide for a 7-inch-tall strip. Past the cap the panel is drawn
    # to the cap width and the field is no longer at equal aspect, which is
    # stated on the axis rather than hidden.
    PH_MAX, PW_MAX = 7.0, 26.0
    PH = min(PH_MAX, PW_MAX * dy / dx)
    PW = min(PW_MAX, PH * dx / dy)
    equal_aspect = PW >= PH * dx / dy - 1e-9

    ml, mr = 1.25, 0.45
    mb, mt = 1.45, 0.85
    cb_h, gap_lo, gap_hi = 0.28, 1.15, 0.30

    FW = _block_inches(ml + PW + mr)
    FH = _block_inches(mb + cb_h + gap_lo + PH + gap_hi + cb_h + mt)

    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([ml / FW, (mb + cb_h + gap_lo) / FH, PW / FW, PH / FH])
    cax_e = fig.add_axes([ml / FW, mb / FH, PW / FW, cb_h / FH])
    cax_u = fig.add_axes([ml / FW, (mb + cb_h + gap_lo + PH + gap_hi) / FH,
                          PW / FW, cb_h / FH])

    tri_up = Triangulation(pts[:, 0], pts[:, 1], sub)
    tri_lo = Triangulation(pts[:, 0], -pts[:, 1], sub)
    c1 = ax.tripcolor(tri_up, umag, cmap="viridis", shading="gouraud",
                      vmin=0.0, vmax=1.0)
    c2 = ax.tripcolor(tri_lo, phi, cmap="hot_r", shading="flat",
                      norm=LogNorm(*phi_lim))

    segs = free_surface(pts, sub, x_max)
    mirrored = [np.column_stack([s[:, 0], -s[:, 1]]) for s in segs]
    ax.add_collection(LineCollection(segs + mirrored, colors=INTERFACE_COLOR,
                                     linewidths=1.4, zorder=5, capstyle="round"))
    ax.axhline(0.0, color="0.45", lw=0.9, ls=(0, (6, 4)), zorder=6)

    # The inlet is the boundary the co-moving formulation assumes is quiescent.
    # Marking it makes "did the disturbance reach it?" a question you can answer
    # by looking, which is the entire point of this view.
    ax.axvline(x_max, color="0.25", lw=2.0, ls=(0, (2, 2)), zorder=7)
    ax.text(x_max, ylim[1], " inlet", ha="right", va="top", rotation=90,
            fontsize=17, color="0.25", zorder=8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal_aspect:
        ax.set_aspect("equal")
    ax.set_xlabel(r"$x/h_0$", fontsize=25, labelpad=7)
    ax.set_ylabel(r"$y/h_0$", fontsize=25, labelpad=7)
    ax.tick_params(which="both", direction="out", width=1.8, labelsize=19, pad=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)

    box = dict(boxstyle="round,pad=0.26", fc="k", ec="none", alpha=0.55)
    ax.text(0.006, 0.88, r"$|u|$", transform=ax.transAxes, fontsize=22,
            color="w", bbox=box, zorder=9)
    ax.text(0.006, 0.05, r"$\varepsilon$", transform=ax.transAxes, fontsize=22,
            color="w", bbox=box, zorder=9)
    label = rf"$t = {t:.2f}$" + ("" if equal_aspect else r"   (y stretched)")
    ax.text(0.994, 0.88, label, transform=ax.transAxes, ha="right",
            fontsize=22, color="w", bbox=box, zorder=9)

    cb1 = fig.colorbar(c1, cax=cax_u, orientation="horizontal")
    cb1.set_label(r"$|u|_{\mathrm{lab}}\,/\,V_{\mathrm{TC}}$", fontsize=23, labelpad=11)
    cb1.ax.tick_params(labelsize=18, pad=4)
    cb1.ax.xaxis.set_label_position("top")
    cb1.ax.xaxis.set_ticks_position("top")
    cb1.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb1.set_ticklabels(["0", "0.25", "0.50", "0.75", "1"])

    cb2 = fig.colorbar(c2, cax=cax_e, orientation="horizontal")
    cb2.set_label(r"$\varepsilon\,/\,(\sigma/h_0)\sqrt{\sigma/\rho h_0}$",
                  fontsize=23, labelpad=9)
    cb2.ax.tick_params(labelsize=18, pad=4)

    fig.savefig(dest, dpi=DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pvd", required=True, type=Path)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--out", default=Path("full_domain.mp4"), type=Path)
    ap.add_argument("--frames-dir", default=Path("frames_full"), type=Path)
    ap.add_argument("--mu", type=float, required=True, help="coded viscosity")
    ap.add_argument("--domain-length", type=float, required=True,
                    help="L0; sets both the view and the inlet marker")
    ap.add_argument("--nframes", type=int, default=220)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    entries = read_pvd(args.pvd)
    times = np.array([t for t, _ in entries])
    targets = np.linspace(times.min(), times.max(), args.nframes)
    picks = sorted({int(np.argmin(np.abs(times - tt))) for tt in targets})
    entries = [entries[i] for i in picks]
    print(f"{len(entries)} frames, t = {times.min():.3f} .. {times.max():.3f}", flush=True)

    hist_raw = np.loadtxt(args.control, comments="#")
    hist_abs = (hist_raw[:, 0], hist_raw[:, 2])   # t, U -- column 2, see module docstring

    # y extent from the LAST frame: the rim grows monotonically, so that frame
    # bounds every earlier one and one read beats scanning the series.
    pts_last, _, _, _ = load_frame(entries[-1][1])
    y_hi = float(np.nanmax(np.abs(pts_last[:, 1])))
    ylim = (-(y_hi + 0.6), y_hi + 0.6)
    xlim = (-0.5, args.domain_length + 0.5)

    phi_hi = 0.0
    for _, p in entries[:: max(1, len(entries) // 12)]:
        _, _, _, ph = load_frame(p)
        phi_hi = max(phi_hi, float(np.nanpercentile(args.mu * ph, 99.5)))
    phi_lim = (max(phi_hi * 1e-4, 1e-8), max(phi_hi, 1e-6))
    print(f"domain x[0,{args.domain_length:g}]  y+-{y_hi + 0.6:.2f}  "
          f"dissipation {phi_lim[0]:.2e} .. {phi_lim[1]:.2e}", flush=True)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    work = partial(render, outdir=args.frames_dir, mu=args.mu, xlim=xlim,
                   ylim=ylim, phi_lim=phi_lim, hist_abs=hist_abs,
                   x_max=args.domain_length)
    with Pool(args.procs) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, list(enumerate(entries)), 1), 1):
            if n % 25 == 0:
                print(f"  {n}/{len(entries)}", flush=True)

    import imageio.v2 as imageio

    frames = sorted(args.frames_dir.glob("frame_*.png"))
    print(f"encoding {len(frames)} frames -> {args.out}", flush=True)
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=None, ffmpeg_params=["-crf", "12", "-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
