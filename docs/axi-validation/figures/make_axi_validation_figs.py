#!/usr/bin/env python3
r"""Publication figures for the axisymmetric validation document.

STYLE. Project publication-plots standard, matched to
``validation/ahsan_campaign_figs.py`` so the two validation notes typeset as
one system: serif/Computer Modern with usetex, PDF as the deliverable and a
PNG alongside for inspection. Geometry (spines, tick lengths, pads, line
widths) is scaled from the 12x12 inch reference by ``PANEL/12``; TEXT sizes are
absolute, because these panels are drawn ~18.5in wide and included at
``\textwidth`` (6.54in), so preserving the reference text ratios would land the
legend near 4pt on the page. Check any change with ``size * 6.54 / width_in``.

usetex is safe here: six curves, one process, serial. It would deadlock under
multiprocessing, so do not parallelise this script -- switch to
``mathtext.fontset='cm'`` instead if it ever grows a per-frame loop.

LINE STYLE, one system across both figures:

    2D DNS (axisymmetric FEM)   solid, blue
    1D ALE, default closure     dashed, green
    1D ALE, toroidal correction dash-dot, pink
    external reference          open circles, orange
    bound / target              dotted, grey

The bound and the reference are the hypotheses under test and are drawn as the
quiet curves; the numerics are not plotted as deviations from them.

PROVENANCE AND CACHING. The run outputs live outside this repository, on the
compute host's run root. Paths are therefore command-line arguments with no
committed defaults, and every series this script reads is cached as a small
two- or three-column CSV under ``figures/data/``. Once that cache exists the
figures rebuild from the repository alone, with no run tree mounted; pass
``--refresh`` together with the source paths to regenerate it.

The planar-limit ladder of figure 2(b) is not a time series and has no control
file: the deviations are the pass/fail record printed by
``validation/trouton1d_axi_checks.py planar-limit`` and
``validation/trouton1d_ale_axi_checks.py planar-limit``. They are transcribed
into ``LADDER_*`` below, and the transcription is the thing to re-check if
those suites are re-run.

Usage::

    python3 make_axi_validation_figs.py                       # from cache
    python3 make_axi_validation_figs.py --refresh \
        --dns <run>/control.txt \
        --ale-default <run>/control1d_ale_axi.txt \
        --ale-tipcorr <run>/control1d_ale_axi.txt \
        --basilisk <run>/Oh1.000_v1501_2024.csv \
        --planar-reduction <run>/control.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"

# ---------------------------------------------------------------- style ----
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

PANEL = 9.0                     # inches per square panel
S = PANEL / 12.0                # geometry scale against the 12x12 reference
LAB = 32
LABPAD = 15 * S
TICK = 30
TICKPAD = 10 * S
LEG = 24
LW = 3 * S
MLW = 3.0 * S

C_DNS = "#0072B2"               # 2D axisymmetric DNS
C_DEF = "#009E73"               # 1D ALE, default tip closure
C_COR = "#CC79A7"               # 1D ALE, toroidal tip correction
C_REF = "#D55E00"               # external reference (Basilisk 2024)
C_TH = "0.45"                   # bound / target, muted grey
C_MASS = "#0072B2"              # 1D mass-coordinate ladder
C_ALE = "#009E73"               # 1D ALE ladder

# --------------------------------------------------------------- ladder ----
# Planar-limit ladder, Oh_SB = 1, deviation = max |u*_axi - u*_planar| over a
# matched output grid.  Mass coordinate: comparison over the whole record to
# t* = 10.  ALE: at t* = 10.  Both suites match the tip cell in RADIUS rather
# than the element count, so the difference carries no discretisation offset.
# Source: the PASS lines of the two `planar-limit` check runs.
LADDER_MASS = np.array([[50.0, 3.3550e-3],
                        [100.0, 2.7171e-3],
                        [200.0, 1.6745e-3],
                        [400.0, 9.2455e-4]])
LADDER_ALE = np.array([[50.0, 3.3565e-3],
                       [100.0, 2.7180e-3],
                       [200.0, 1.6749e-3]])

DNS_PLANAR_TARGET = 0.9297      # validated planar solver, Oh_SB = 0.5, t* = 30


def axstyle(ax, square=True):
    for s in ax.spines.values():
        s.set_linewidth(LW)
    ax.tick_params(which="major", direction="out", width=LW, length=12 * S,
                   labelsize=TICK, pad=TICKPAD)
    ax.tick_params(which="minor", direction="out", width=LW, length=6 * S,
                   labelsize=TICK * 0.80, pad=TICKPAD * 0.8)
    ax.minorticks_on()
    if square:
        ax.set_box_aspect(1)


def poslog(x, y, what):
    """Keep strictly positive pairs; report what a log axis would have eaten.

    A log axis drops non-positive values silently, so a curve that is wrong
    looks merely short.  Nothing goes onto a log axis here without passing
    through this.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int((~good).sum())
    if n:
        print(f"    {what}: dropped {n}/{len(x)} non-positive points")
    return x[good], y[good]


def posx(x, y):
    """Mask only the abscissa, for panels whose ordinate axis is linear.

    The DNS record at ``R0 = 1`` carries genuinely negative ``u*`` over the
    first ~0.2 viscous times: the tip recedes before it opens.  That is the
    measurement, so it must not be filtered out by a helper written for log
    ordinates.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[good], y[good]


# ----------------------------------------------------------------- data ----
def _cache_path(name):
    return CACHE / f"{name}.csv"


def _write_cache(name, cols, header):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savetxt(_cache_path(name), np.column_stack(cols), delimiter=",",
               header=header, comments="# ", fmt="%.10e")
    print(f"  cached {name}.csv ({len(cols[0])} rows)")


def _read_cache(name):
    p = _cache_path(name)
    if not p.exists():
        raise SystemExit(
            f"missing cache {p}; rerun with --refresh and the source paths")
    return np.loadtxt(p, delimiter=",")


def refresh(args):
    """Extract the plotted columns from the run outputs into the cache."""
    if args.dns:
        d = np.loadtxt(args.dns)          # t tstar R Rdot ustar volume
        _write_cache("dns_oh0707_r0_1", (d[:, 1], d[:, 4], d[:, 2]),
                     "tstar,ustar,R -- 2D axisymmetric DNS, lab frame, "
                     "Oh_SB = 0.707107, R0 = 1")
    if args.ale_default:
        d = np.loadtxt(args.ale_default)  # t tstar R u_tip ustar ...
        _write_cache("ale1d_default", (d[:, 1], d[:, 4], d[:, 2]),
                     "tstar,ustar,R -- 1D axi ALE, default tip closure, "
                     "Oh_SB = 0.707107, R0 = R_eq = 1.1129")
    if args.ale_tipcorr:
        d = np.loadtxt(args.ale_tipcorr)
        _write_cache("ale1d_tipcorr", (d[:, 1], d[:, 4], d[:, 2]),
                     "tstar,ustar,R -- 1D axi ALE, toroidal tip correction, "
                     "Oh_SB = 0.707107, R0 = R_eq = 1.1129")
    if args.basilisk:
        d = np.loadtxt(args.basilisk, delimiter=",", skiprows=1)  # t rTip vTip
        # tau_vis = mu/2 and the reference carries mu = 1, so t* = 2t; the
        # reference reports the dimensional tip speed, u* = vTip / V_TC.
        _write_cache("basilisk2024_oh1", (2.0 * d[:, 0],
                                          d[:, 2] / np.sqrt(2.0), d[:, 1]),
                     "tstar,ustar,rTip -- Basilisk volume-of-fluid, 2024 "
                     "campaign, coded Oh = 1 (Oh_SB = 0.707107), R0 = 1")
    if args.planar_reduction:
        d = np.loadtxt(args.planar_reduction)
        _write_cache("dns_planar_reduction_r0_100", (d[:, 1], d[:, 4]),
                     "tstar,ustar -- 2D axisymmetric DNS, Oh_SB = 0.5, "
                     "R0 = 100 (planar-reduction gate)")


# -------------------------------------------------------------- figure 1 ---
def figure1():
    dns = _read_cache("dns_oh0707_r0_1")
    dfl = _read_cache("ale1d_default")
    cor = _read_cache("ale1d_tipcorr")
    bas = _read_cache("basilisk2024_oh1")

    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL + 1.4, PANEL + 0.6))
    a, b = axes

    # (a) the four records on their own abscissae.
    a.plot(*posx(dns[:, 0], dns[:, 1]), "-", color=C_DNS, lw=MLW,
           label=r"2D DNS (lab frame)", zorder=4)
    a.plot(*posx(dfl[:, 0], dfl[:, 1]), "--", color=C_DEF,
           lw=MLW, label=r"1D ALE, $N_{rr}(R)=-2$", zorder=3)
    a.plot(*posx(cor[:, 0], cor[:, 1]), "-.", color=C_COR,
           lw=MLW, label=r"1D ALE, $-2+\pi h_{\rm tip}/2R$", zorder=3)
    xb, yb = posx(bas[:, 0], bas[:, 1])
    # The reference is written on a UNIFORM time grid and is plotted on a log
    # abscissa, so an every-nth stride would put every marker in the last
    # decade.  Choose the marker positions log-spaced in t* instead.
    want = np.logspace(np.log10(max(xb[0], 2e-2)), np.log10(xb[-1]), 16)
    idx = np.unique(np.searchsorted(xb, want).clip(0, len(xb) - 1))
    a.plot(xb[idx], yb[idx], "o", mfc="none", mec=C_REF, mew=MLW * 0.7,
           ms=13 * S, label=r"Basilisk VoF (2024)", zorder=2)
    a.axhline(1.0, ls=":", color=C_TH, lw=MLW * 0.8, zorder=1)
    a.text(0.13, 1.012, r"$u^{*}=1$", color=C_TH, fontsize=LEG)

    a.set_xscale("log")
    a.set_xlim(1e-2, 1.2e2)
    a.set_ylim(-0.10, 1.10)
    a.set_xlabel(r"$t^{*}=t/\tau_{\rm vis}$", fontsize=LAB, labelpad=LABPAD)
    a.set_ylabel(r"$u^{*}=\dot{R}/V_{TC}$", fontsize=LAB, labelpad=LABPAD)
    a.legend(fontsize=LEG, loc="lower right", frameon=False,
             handlelength=2.4, borderaxespad=0.6)
    axstyle(a)
    a.text(-0.16, 1.02, r"(a)", transform=a.transAxes, fontsize=LAB)

    # (b) departure from the DNS, on the DNS abscissa.  The three records sit
    # on three geometric output schedules, so one of them has to be
    # interpolated; the curves here are smooth and monotone over the window
    # plotted, unlike the planar deceleration knee where linear interpolation
    # manufactures two orders of magnitude of spurious disagreement.
    t = dns[:, 0]
    m = t >= 1.0
    for src, col, ls, lab in ((dfl, C_DEF, "--", r"1D ALE, $N_{rr}(R)=-2$"),
                              (cor, C_COR, "-.",
                               r"1D ALE, $-2+\pi h_{\rm tip}/2R$"),
                              (bas, C_REF, ":", r"Basilisk VoF (2024)")):
        u = np.interp(t, src[:, 0], src[:, 1],
                      left=np.nan, right=np.nan)
        b.plot(t[m], (u - dns[:, 1])[m], ls, color=col, lw=MLW, label=lab)
    b.axhline(0.0, color=C_DNS, lw=MLW, zorder=1)
    b.set_xscale("log")
    b.set_xlim(1.0, 1.05e2)
    b.set_xlabel(r"$t^{*}=t/\tau_{\rm vis}$", fontsize=LAB, labelpad=LABPAD)
    b.set_ylabel(r"$u^{*}-u^{*}_{\rm DNS}$", fontsize=LAB, labelpad=LABPAD)
    # Opaque, unbordered: the legend sits over the tail of the default-closure
    # curve and an unfilled box would read as a crossing.
    b.legend(fontsize=LEG, loc="upper left", frameon=True, framealpha=1.0,
             edgecolor="none", handlelength=2.4, borderaxespad=0.6)
    axstyle(b)
    b.text(-0.20, 1.02, r"(b)", transform=b.transAxes, fontsize=LAB)

    save(fig, "fig1_closure_bracket")


# -------------------------------------------------------------- figure 2 ---
def figure2():
    red = _read_cache("dns_planar_reduction_r0_100")

    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL + 1.4, PANEL + 0.6))
    a, b = axes

    # (a) the DNS at R0 = 100 against the validated planar value.
    a.plot(*posx(red[:, 0], red[:, 1]), "-", color=C_DNS, lw=MLW,
           label=r"2D DNS, $R_{0}=100$", zorder=3)
    a.axhline(DNS_PLANAR_TARGET, ls=":", color=C_TH, lw=MLW,
              label=r"planar solver, $u^{*}(30)=0.9297$", zorder=2)
    u30 = red[-1, 1]
    a.plot([red[-1, 0]], [u30], "o", color=C_DNS, ms=17 * S, zorder=4)
    # The whole point of the panel is a 0.13% gap, which is invisible on an
    # axis that also has to show the transient from rest.  Draw the gap.
    ins = a.inset_axes([0.13, 0.58, 0.44, 0.32])
    ins.plot(red[:, 0], red[:, 1], "-", color=C_DNS, lw=MLW)
    ins.axhline(DNS_PLANAR_TARGET, ls=":", color=C_TH, lw=MLW)
    ins.plot([red[-1, 0]], [u30], "o", color=C_DNS, ms=13 * S)
    ins.set_xlim(20.0, 32.0)
    ins.set_ylim(0.9255, 0.9330)
    ins.set_yticks([0.9297, 0.9309])
    ins.set_yticklabels([r"$0.9297$", r"$0.9309$"])
    ins.set_xticks([20, 25, 30])
    for s in ins.spines.values():
        s.set_linewidth(LW * 0.7)
    ins.tick_params(which="major", direction="out", width=LW * 0.7,
                    length=8 * S, labelsize=TICK * 0.62, pad=TICKPAD * 0.6)
    ins.tick_params(which="minor", length=0)
    ins.text(0.05, 0.78, r"$0.13\%$", transform=ins.transAxes,
             fontsize=LEG * 0.85, color=C_TH)
    a.set_xscale("log")
    a.set_xlim(1e-2, 4e1)
    a.set_ylim(0.0, 1.02)
    a.set_xlabel(r"$t^{*}=t/\tau_{\rm vis}$", fontsize=LAB, labelpad=LABPAD)
    a.set_ylabel(r"$u^{*}=\dot{R}/V_{TC}$", fontsize=LAB, labelpad=LABPAD)
    a.legend(fontsize=LEG, loc="lower right", frameon=False,
             handlelength=2.4, borderaxespad=0.6)
    axstyle(a)
    a.text(-0.16, 1.02, r"(a)", transform=a.transAxes, fontsize=LAB)

    # (b) the 1D planar-limit ladder against the 2/R0 admission bound.
    inv = np.logspace(-3.4, -1.5, 50)
    b.plot(inv, 2.0 * inv, ":", color=C_TH, lw=MLW,
           label=r"bound $2/R_{0}$")
    b.plot(inv, 0.37 * inv, "-", color="0.75", lw=MLW * 0.8,
           label=r"$0.37/R_{0}$")
    b.plot(*poslog(1.0 / LADDER_MASS[:, 0], LADDER_MASS[:, 1], "mass ladder"),
           "s-", color=C_MASS, lw=MLW, ms=15 * S, mfc="none", mew=MLW * 0.7,
           label=r"1D mass coordinate")
    b.plot(*poslog(1.0 / LADDER_ALE[:, 0], LADDER_ALE[:, 1], "ALE ladder"),
           "^--", color=C_ALE, lw=MLW, ms=15 * S, mfc="none", mew=MLW * 0.7,
           label=r"1D ALE")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(1.5e-3, 3e-2)
    # Default log minor labelling puts five overlapping labels on this
    # 1.3-decade axis.  Label the rungs themselves instead.
    b.set_xlabel(r"$1/R_{0}$", fontsize=LAB, labelpad=LABPAD)
    b.set_ylabel(r"$\max|u^{*}_{\rm axi}-u^{*}_{\rm planar}|$",
                 fontsize=LAB, labelpad=LABPAD)
    b.legend(fontsize=LEG, loc="upper left", frameon=False, handlelength=2.4,
             borderaxespad=0.6)
    axstyle(b)
    # After axstyle, which turns minor ticks back on: the default minor
    # labelling puts five overlapping labels on this 1.3-decade axis, so label
    # the rungs themselves instead.
    b.set_xticks([1 / 400, 1 / 200, 1 / 100, 1 / 50])
    b.set_xticklabels([r"$1/400$", r"$1/200$", r"$1/100$", r"$1/50$"])
    b.set_xticks([], minor=True)
    b.text(-0.22, 1.02, r"(b)", transform=b.transAxes, fontsize=LAB)

    save(fig, "fig2_planar_reduction")


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true",
                   help="re-extract the cached series from the run outputs")
    p.add_argument("--dns")
    p.add_argument("--ale-default")
    p.add_argument("--ale-tipcorr")
    p.add_argument("--basilisk")
    p.add_argument("--planar-reduction")
    args = p.parse_args()
    if args.refresh:
        print("refreshing cache")
        refresh(args)
    print("figure 1")
    figure1()
    print("figure 2")
    figure2()


if __name__ == "__main__":
    main()
