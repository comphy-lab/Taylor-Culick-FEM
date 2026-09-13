"""Animate a co-moving run as two stacked rows: the whole domain, then ``u*(t)``.

This is the production layout. It supersedes the side-by-side arrangement of
``animate_planar.py`` for presentation use, and differs from both predecessors
in what it puts where:

  row 1  the DISTURBED part of the co-moving domain: ``x`` from 0 out to just
         beyond where the film stops noticing the rim. Velocity magnitude
         above the midplane, viscous dissipation rate below, free surface over
         both.
  row 2  left, the recovered control velocity ``U(t)/V_TC`` drawn up to the
         current frame; right, the two colourbars.

Why not the entire domain to ``L0``: the production windows run to
``L0 = 140-340`` while the disturbance reaches a few tens of thicknesses, so a
full-domain frame is 80-90% undisturbed film and the rim renders a few pixels
wide. ``--xmax auto`` measures the disturbance front on the LAST frame -- the
furthest it ever reaches -- and frames slightly beyond it. The window is fixed
for the whole animation, so the view does not breathe; the caption states
``L0`` so the truncation is visible even when the inlet is off-screen.

The front is measured from the lab-frame velocity and from the interface
together. Either alone misses a case: at high ``Oh`` the film is set in motion
well ahead of any visible thickening, and at low ``Oh`` the capillary-wave
train deforms the interface where the velocity perturbation is already tiny.

The time axis of row 2 is LOGARITHMIC and spans exactly the first to the last
recorded point, with no padding. The solver writes on a geometric output
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


def disturbance_front(path: Path, u_frame: float, L0: float,
                      u_tol: float = 2e-3, h_tol: float = 5e-3) -> float:
    """Largest ``x`` at which the film still notices the rim.

    Two independent detectors, because either alone under-reads a limit of the
    problem:

    * **velocity.** In the laboratory frame the undisturbed film is at rest by
      construction -- the inlet imposes ``u_x = -U`` and the shift adds ``+U``
      -- so any ``|u|`` above the tolerance is disturbance. This is the one
      that binds at high ``Oh``, where the viscous footprint ``sqrt(4 nu t)``
      sets fluid moving far ahead of any visible thickening.
    * **interface.** The half-thickness relaxes to ``h0/2``, and a departure
      from it is disturbance. This is the one that binds at low ``Oh``, where
      the capillary-wave train ripples the surface out past the point where
      the velocity perturbation has dropped below ``u_tol``.

    The reference half-thickness is taken as the MEDIAN over the outer tenth
    of the domain rather than the nominal 0.5, so a small systematic offset in
    a particular run does not read as disturbance everywhere.
    """
    pts, sub, vel, _ = load_frame(path)
    lab = vel.copy()
    lab[:, 0] += u_frame
    mag = np.linalg.norm(lab, axis=1) / V_TC
    x_u = float(pts[mag > u_tol, 0].max()) if np.any(mag > u_tol) else 0.0

    segs = free_surface(pts, sub, L0)
    x_h = 0.0
    if segs:
        node = np.unique(np.round(np.vstack([np.vstack(s) for s in segs]), 9),
                         axis=0)
        far = node[node[:, 0] > 0.9 * L0]
        h_inf = float(np.median(far[:, 1])) if len(far) else 0.5
        off = np.abs(node[:, 1] - h_inf) > h_tol
        if np.any(off):
            x_h = float(node[off, 0].max())

    return max(x_u, x_h)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(item, *, outdir: Path, mu: float, xlim, ylim, phi_lim, hist,
           hist_abs, x_max: float, tlim, oh_label: str, L0: float) -> None:
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

    ml, mr = 1.30, 0.50
    # The row-2 x label is a fraction under a radical and is roughly twice the
    # height of a bare tick label, so the bottom margin is not the mirror of
    # the top one. At 0.95 in it was clipped.
    mb, mt = 1.55, 0.55
    cb_h = 0.30
    gap_rows = 1.30        # row-1 x label -> row 2

    # Row 2 splits its width: the trace on the left, the two colourbars on the
    # right. The two are PARTITIONED, not sized independently -- when the x
    # crop makes the field panel narrow, independently floored widths sum to
    # more than the panel and the colourbars land on top of the plot.
    #
    # Row 2 also sets a width floor of its own. The field panel can come out
    # under 9 inches once the view is cropped to the disturbance front, and a
    # 4-inch colourbar cannot carry readable tick labels at this font size.
    ROW2_MIN = 13.0
    row2_w = max(PW, ROW2_MIN)
    gap_cols = 0.05 * row2_w
    RW = 0.40 * (row2_w - gap_cols)
    CW = (row2_w - gap_cols) - RW
    RH = min(RW, 7.0)                 # the trace is square, but bounded

    content_w = max(PW, row2_w)
    FW = _block_inches(ml + content_w + mr)
    FH = _block_inches(mb + RH + gap_rows + PH + mt)

    fig = plt.figure(figsize=(FW, FH))

    # Field panel centred over row 2 when row 2 is the wider of the two.
    ax_x = ml + 0.5 * (content_w - PW)
    y_row1 = (mb + RH + gap_rows) / FH
    ax = fig.add_axes([ax_x / FW, y_row1, PW / FW, PH / FH])
    axR = fig.add_axes([ml / FW, mb / FH, RW / FW, RH / FH])

    # Colourbars stacked and vertically centred against the trace. Velocity on
    # top to match the panel above, where velocity is the upper half.
    cb_left = ml + row2_w - CW
    cb_gap = 0.26 * RH
    cb_lo_y = mb + 0.5 * RH - 0.5 * (2 * cb_h + cb_gap)
    cax_e = fig.add_axes([cb_left / FW, cb_lo_y / FH, CW / FW, cb_h / FH])
    cax_u = fig.add_axes([cb_left / FW, (cb_lo_y + cb_h + cb_gap) / FH,
                          CW / FW, cb_h / FH])

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

    # The inlet is only drawn when the framing actually reaches it. When the
    # view is cropped to the disturbance front the inlet is off-screen, and a
    # marker at the panel edge would claim the domain ends there -- which is
    # exactly the misreading the full-domain view existed to prevent. The
    # caption carries L0 instead.
    if x_max <= xlim[1] + 1e-9:
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
    # rotated label lives when it is drawn at all. The lower half far from the
    # rim is undisturbed film, so nothing is covered here.
    ax.text(0.994, 0.06, label, transform=ax.transAxes, ha="right",
            fontsize=22, color="w", bbox=box, zorder=9)
    if x_max > xlim[1] + 1e-9:
        # State the truncation the framing hides.
        ax.text(0.994, 0.88, rf"domain $L_0/h_0 = {L0:g}$", transform=ax.transAxes,
                ha="right", fontsize=19, color="w", bbox=box, zorder=9)

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
        # clip_on off: the axis ends exactly at the last data point, so on the
        # final frame a clipped marker would be sliced in half by the spine.
        axR.plot(th[mask][-1], uh[mask][-1], "o", ms=12, color=TRACE_COLOR,
                 mec="k", mew=1.8, zorder=4, clip_on=False)

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
    ap.add_argument("--xmax", type=float, default=None,
                    help="fix the view window in x; default measures the "
                         "disturbance front on the last frame")
    ap.add_argument("--xmax-pad", type=float, default=1.18,
                    help="multiple of the measured front to frame out to")
    ap.add_argument("--nframes", type=int, default=240)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.case_dir / "case.json").read_text())
    mu = float(meta["mu_coded"])
    L0 = float(meta["domain_length"])
    oh = float(meta.get("Oh", meta.get("Oh_savva_bush")))
    oh_label = f"{oh:g}"

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
    print(f"Oh={oh_label}  L0={L0:g}  {len(entries)} frames, "
          f"t = {times.min():.4g} .. {times.max():.4g}", flush=True)

    hist_raw = load_control(args.case_dir)
    hist = (hist_raw[:, 0], hist_raw[:, 3])       # t, U/V_TC   (row 2)
    hist_abs = (hist_raw[:, 0], hist_raw[:, 2])   # t, U        (lab-frame shift)
    # Exactly the first and last recorded points, no padding. The axis is
    # logarithmic so the first point must be positive.
    t_hist = hist_raw[hist_raw[:, 0] > 0, 0]
    tlim = (float(t_hist.min()), float(t_hist.max()))

    # y extent from the LAST frame: the rim grows monotonically, so that frame
    # bounds every earlier one and one read beats scanning the series.
    t_last, p_last = entries[-1]
    pts_last, _, _, _ = load_frame(p_last)
    y_hi = float(np.nanmax(np.abs(pts_last[:, 1])))
    ylim = (-(y_hi + 0.6), y_hi + 0.6)

    # The x window is fixed for the whole animation and sized on the LAST
    # frame, where the disturbance has reached furthest. Sizing per frame
    # would make the view breathe and destroy any sense of scale.
    if args.xmax is not None:
        x_hi = args.xmax
    else:
        u_last = float(np.interp(t_last, hist_abs[0], hist_abs[1]))
        front = disturbance_front(p_last, u_last, L0)
        # The floor only stops the panel coming out taller than it is wide,
        # which happens at low Oh where the rim grows large but its influence
        # stays local. It is not meant to pad the window, and at 4 y_hi it was
        # doing exactly that -- doubling a 45-thickness front to 94.
        x_hi = min(L0 + 0.5, args.xmax_pad * max(front, 1.2 * y_hi))
        print(f"disturbance front at x = {front:.1f}; "
              f"framing x <= {x_hi:.1f} of L0 = {L0:g}", flush=True)
    xlim = (-0.5, x_hi)

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
                   tlim=tlim, oh_label=oh_label, L0=L0)
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
