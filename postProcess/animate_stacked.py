"""Animate a co-moving run as two stacked rows: the whole domain, then ``u*(t)``.

This is the production layout. It supersedes the side-by-side arrangement of
``animate_planar.py`` for presentation use, and differs from both predecessors
in what it puts where:

  row 1  the ENTIRE co-moving domain, ``x`` from 0 to ``L0`` -- the framing of
         ``animate_full_domain.py``. Velocity magnitude above the midplane,
         viscous dissipation rate below, free surface over both, inlet marked.
  row 2  the recovered control velocity ``U(t)/V_TC``, drawn up to the current
         frame. CENTRED and deliberately narrow: a trace stretched to the full
         width of a 200-thickness domain is a flat line with no readable
         structure, and the approach to ``V_TC`` is the point of it.

Why the whole domain rather than a window at the rim: the co-moving
formulation is only valid while the inlet stays quiescent, and the
domain-length sweep (``u* = 0.819`` at ``L = 12`` against ``0.950`` at
``L >= 40``) says truncation dominated the answer until the window was long
enough. Row 1 makes that a question you can answer by looking.

The time axis of row 2 is LOGARITHMIC. The solver writes on a geometric output
schedule, so successive frames are spaced by a growing ``dt``; on a linear axis
the marker crawls for most of the run and then bolts. Log time makes the marker
travel at roughly constant speed and shows the asymptotic approach to
``V_TC``, which a linear axis compresses into the last few pixels.

The loaders, the P2 -> P1 dissipation reconstruction and the free-surface edge
extraction are imported from ``animate_planar`` rather than copied.

Conventions inherited deliberately, each of which was a bug once:

  * velocity is normalised by ``V_TC``, not by the capillary velocity, so a
    value of 1 means "moving at the Taylor-Culick speed" wherever it appears;
  * the lab-frame shift uses ``control.txt`` column 2 (``U``), not column 1
    (``tstar``);
  * figure dimensions round to a whole macro block, or libx264 fails on odd
    sizes and imageio silently rescales anything not divisible by 16;
  * frames render with mathtext, never usetex: LaTeX spawns a subprocess per
    unique string and deadlocks under multiprocessing. ``mathtext.fontset =
    "cm"`` is Computer Modern, so the glyphs match the docs figures anyway.

Restart-damaged runs
--------------------
``--restart-from`` truncates ``control.txt`` and rewrites ``liquid.pvd``, and
the post-restart run overwrites ``liquid_000000.vtu`` onwards. ``--reconstruct``
rebuilds the pre-restart index-to-time map from the launcher log's ``OUTPUT at
t=`` lines, which this repository has verified against an intact ``liquid.pvd``
to be exactly the emitted series (see ``_reconstructed_frames``).
"""

from __future__ import annotations

import argparse
import json
import re
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

TRACE_COLOR = "#c1121f"
GHOST_COLOR = "0.72"


def _block_inches(inches: float, block: int = 16, dpi: int = DPI) -> float:
    """Round a figure dimension up to a whole number of macro blocks.

    Even pixels are enough for libx264 to accept the stream, but imageio then
    resizes anything not divisible by 16 up to the next block and says so only
    in a warning -- a small anisotropic stretch applied after the field has
    been drawn at equal aspect, which is the thing this view is meant to be
    trusted on.
    """
    px = int(np.ceil(inches * dpi / block) * block)
    return px / dpi


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_control(case_dir: Path) -> np.ndarray:
    """Merged ``(t, tstar, U, U/V_TC)`` history across a restart.

    ``control.txt`` is truncated on restart and ``control_part1.txt`` holds the
    pre-restart record, which for the three restarted cases runs FURTHER in
    time than the post-restart continuation does -- the restart resumed from an
    earlier checkpoint. Neither file alone is the trajectory. The union sorted
    by ``t`` is, and the two branches agree to ~5e-6 in ``u*`` where they
    overlap, so the join is invisible.
    """
    rv = case_dir / "run_validation"
    rows = []
    for name in ("control_part1.txt", "control.txt"):
        p = rv / name
        if p.exists() and p.stat().st_size > 0:
            a = np.atleast_2d(np.loadtxt(p, comments="#"))
            if a.ndim == 2 and a.shape[0] and a.shape[1] >= 4:
                rows.append(a)
    if not rows:
        raise SystemExit(f"no usable control history under {rv}")
    a = np.vstack(rows)
    a = a[np.argsort(a[:, 0])]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    return a[keep]


def _reconstructed_frames(case_dir: Path, log: Path) -> list[tuple[float, Path]]:
    """Pre-restart ``(time, vtu)`` pairs recovered from the launcher log.

    The launcher log prints ``OUTPUT at t=<t>`` once per emitted VTU, in order,
    so log line ``i`` names the time of ``liquid_{i:06d}.vtu``. Verified
    exactly (max difference 0.0 over 2120 entries) against the intact
    ``liquid.pvd`` of a case that was never restarted. The log is truncated by
    a couple of dozen lines at the tail -- the launcher's buffer was never
    flushed -- so the reconstruction stops where the log does.

    Indices below ``n_clobbered`` are skipped: the post-restart run rewrote
    ``liquid_000000.vtu`` upward, so those files no longer hold the
    pre-restart state their log times claim.
    """
    liquid = case_dir / "run_validation" / "liquid"
    times = [float(m) for m in re.findall(r"^OUTPUT at t=([0-9.eE+-]+)",
                                          log.read_text(errors="ignore"), re.M)]
    pvd = case_dir / "run_validation" / "liquid.pvd"
    n_clobbered = len(read_pvd(pvd)) if pvd.exists() else 0
    out = []
    for i, t in enumerate(times):
        if i < n_clobbered:
            continue
        f = liquid / f"liquid_{i:06d}.vtu"
        if f.exists():
            out.append((t, f))
    if not out:
        raise SystemExit(f"reconstruction from {log} recovered no frames")
    print(f"reconstructed {len(out)} pre-restart frames from {log.name} "
          f"(skipped {n_clobbered} clobbered indices)", flush=True)
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(item, *, outdir: Path, mu: float, xlim, ylim, phi_lim, hist,
           hist_abs, x_max: float, tlim, oh_label: str) -> None:
    i, (t, path) = item
    dest = outdir / f"frame_{i:05d}.png"
    if dest.exists():
        return

    pts, sub, vel, phi = load_frame(path)
    phi = mu * phi

    th_abs, u_abs = hist_abs
    u_now = float(np.interp(t, th_abs, u_abs))
    vel_lab = vel.copy()
    vel_lab[:, 0] += u_now                      # co-moving -> laboratory frame
    umag = np.linalg.norm(vel_lab, axis=1) / V_TC

    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

    # Row 1 follows the domain aspect but is capped: at L = 340 the true aspect
    # is far past 10:1 and an unclamped canvas would be metres wide. Past the
    # cap the field is no longer at equal aspect, which is stated on the panel
    # rather than hidden.
    PH_MAX, PW_MAX = 6.8, 26.0
    PH = min(PH_MAX, PW_MAX * dy / dx)
    PW = min(PW_MAX, PH * dx / dy)
    equal_aspect = PW >= PH * dx / dy - 1e-9

    # Row 2 is centred and narrow on purpose (see module docstring). It is
    # sized against the row-1 panel so the proportion holds whatever the
    # domain aspect turns out to be, with a floor so short domains still get a
    # legible trace.
    RW = min(max(0.40 * PW, 6.0), 10.0)
    RH = 4.5

    ml, mr = 1.30, 0.50
    # The row-2 x label is a fraction under a radical and is roughly twice the
    # height of a bare tick label, so the bottom margin is not the mirror of
    # the top one. At 0.95 in it was clipped.
    mb, mt = 1.75, 0.80
    cb_h = 0.28
    gap_hi = 0.30          # panel spine -> velocity colorbar
    gap_lo = 1.15          # x tick labels and x label -> dissipation colorbar
    gap_mid = 1.45         # dissipation colorbar label -> row 2

    # Colorbars are centred on the panel and capped rather than spanning it.
    # A 26-inch bar 0.28 in tall reads as a decorative stripe and puts its
    # ticks metres from the field they describe.
    CW = min(PW, 13.0)
    cb_x = (ml + 0.5 * (PW - CW)) / 1.0

    FW = _block_inches(ml + PW + mr)
    FH = _block_inches(mb + RH + gap_mid + cb_h + gap_lo + PH + gap_hi + cb_h + mt)

    fig = plt.figure(figsize=(FW, FH))

    y_row1 = (mb + RH + gap_mid + cb_h + gap_lo) / FH
    ax = fig.add_axes([ml / FW, y_row1, PW / FW, PH / FH])
    cax_e = fig.add_axes([cb_x / FW, (mb + RH + gap_mid) / FH, CW / FW, cb_h / FH])
    cax_u = fig.add_axes([cb_x / FW, y_row1 + (PH + gap_hi) / FH, CW / FW, cb_h / FH])
    # Centred on the FIELD panel, not on the canvas: the left margin carries the
    # y label and the right one does not, so canvas-centring would sit visibly
    # off to the right of the thing it describes.
    axR = fig.add_axes([(ml + 0.5 * (PW - RW)) / FW, mb / FH, RW / FW, RH / FH])

    # ---- row 1: the whole domain -----------------------------------------
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
    label = rf"$Oh = {oh_label}$   $t = {t:.2f}$"
    if not equal_aspect:
        label += r"   ($y$ stretched)"
    # Bottom right, not top right: the top right is where the inlet marker's
    # rotated label lives, and at large L the two sit on top of each other.
    # The lower half far from the rim is undisturbed film, so nothing is
    # covered here.
    ax.text(0.994, 0.06, label, transform=ax.transAxes, ha="right",
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

    # ---- row 2: the control trace ----------------------------------------
    th, uh = hist
    # The whole trajectory in grey behind the drawn portion: without it the
    # y range appears to change as the curve grows, and the viewer cannot see
    # how much of the approach is still to come.
    axR.plot(th, uh, color=GHOST_COLOR, lw=2.0, zorder=1)
    axR.axhline(1.0, color="k", lw=2.0, ls="--", zorder=2)
    # Left, where the curve is still far below the line. On the right the
    # marker arrives at V_TC and sits under the label.
    axR.text(0.015, 1.0, r"$V_{\mathrm{TC}}$", transform=axR.get_yaxis_transform(),
             ha="left", va="bottom", fontsize=20)

    mask = th <= t + 1e-12
    axR.plot(th[mask], uh[mask], color=TRACE_COLOR, lw=3.0, zorder=3)
    if mask.any():
        axR.plot(th[mask][-1], uh[mask][-1], "o", ms=12, color=TRACE_COLOR,
                 mec="k", mew=1.8, zorder=4)

    axR.set_xscale("log")
    axR.set_xlim(*tlim)
    axR.set_ylim(0.0, 1.12)
    axR.set_xlabel(r"$t\,\sqrt{\sigma/\rho h_0^3}$", fontsize=24, labelpad=6)
    axR.set_ylabel(r"$V/V_{\mathrm{TC}}$", fontsize=24, labelpad=6)
    axR.tick_params(which="both", direction="out", width=1.8, labelsize=18, pad=4)
    axR.tick_params(which="major", length=8)
    axR.tick_params(which="minor", length=4)
    for sp in axR.spines.values():
        sp.set_linewidth(1.8)

    fig.savefig(dest, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", required=True, type=Path,
                    help="case root holding case.json and run_validation/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="default: <out>.frames alongside the video")
    ap.add_argument("--reconstruct", type=Path, default=None,
                    help="launcher log to rebuild a restart-truncated pvd from")
    ap.add_argument("--nframes", type=int, default=240)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.case_dir / "case.json").read_text())
    mu = float(meta["mu_coded"])
    L0 = float(meta["domain_length"])
    oh_sb = float(meta["Oh_savva_bush"])
    oh_label = f"{oh_sb:g}"

    if args.reconstruct:
        entries = _reconstructed_frames(args.case_dir, args.reconstruct)
    else:
        entries = read_pvd(args.case_dir / "run_validation" / "liquid.pvd")
    if not entries:
        raise SystemExit("no frames")

    # Subsample geometrically in time, not uniformly and not by file index.
    # The output schedule is geometric, so uniform-in-t sampling spends almost
    # every frame in the last decade and skips the entire early transient --
    # which at low Oh is where the capillary-wave train forms.
    times = np.array([t for t, _ in entries])
    t_lo = max(times[times > 0].min(), times.max() * 1e-5)
    targets = np.geomspace(t_lo, times.max(), args.nframes)
    picks = sorted({int(np.argmin(np.abs(times - tt))) for tt in targets})
    entries = [entries[i] for i in picks]
    print(f"Oh_SB={oh_label}  L0={L0:g}  {len(entries)} frames, "
          f"t = {times.min():.4g} .. {times.max():.4g}", flush=True)

    hist_raw = load_control(args.case_dir)
    hist = (hist_raw[:, 0], hist_raw[:, 3])       # t, U/V_TC   (row 2)
    hist_abs = (hist_raw[:, 0], hist_raw[:, 2])   # t, U        (lab-frame shift)
    tlim = (t_lo, max(times.max(), hist_raw[-1, 0]) * 1.05)

    # y extent from the LAST frame: the rim grows monotonically, so that frame
    # bounds every earlier one and one read beats scanning the series.
    pts_last, _, _, _ = load_frame(entries[-1][1])
    y_hi = float(np.nanmax(np.abs(pts_last[:, 1])))
    ylim = (-(y_hi + 0.6), y_hi + 0.6)
    xlim = (-0.5, L0 + 0.5)

    phi_hi = 0.0
    for _, p in entries[:: max(1, len(entries) // 12)]:
        _, _, _, ph = load_frame(p)
        phi_hi = max(phi_hi, float(np.nanpercentile(mu * ph, 99.5)))
    phi_lim = (max(phi_hi * 1e-4, 1e-8), max(phi_hi, 1e-6))
    print(f"domain x[0,{L0:g}]  y+-{y_hi + 0.6:.2f}  "
          f"dissipation {phi_lim[0]:.2e} .. {phi_lim[1]:.2e}", flush=True)

    frames_dir = args.frames_dir or args.out.with_suffix(".frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    work = partial(render, outdir=frames_dir, mu=mu, xlim=xlim, ylim=ylim,
                   phi_lim=phi_lim, hist=hist, hist_abs=hist_abs, x_max=L0,
                   tlim=tlim, oh_label=oh_label)
    with Pool(args.procs) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, list(enumerate(entries)), 1), 1):
            if n % 25 == 0:
                print(f"  {n}/{len(entries)}", flush=True)

    import imageio.v2 as imageio

    files = sorted(frames_dir.glob("frame_*.png"))
    print(f"encoding {len(files)} frames -> {args.out}", flush=True)
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=None,
                            ffmpeg_params=["-crf", "14", "-pix_fmt", "yuv420p"]) as w:
        for f in files:
            w.append_data(imageio.imread(f))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)

    if not args.keep_frames:
        for f in files:
            f.unlink()
        frames_dir.rmdir()


if __name__ == "__main__":
    main()
