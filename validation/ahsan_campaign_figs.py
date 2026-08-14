#!/usr/bin/env python3
r"""Publication figures for the 1D validation document.

STYLE. Project publication-plots standard: serif/Computer Modern with usetex,
PDF as the deliverable and a PNG alongside for inspection. The reference sizing
is quoted for a ~12x12 inch panel (labels 40, ticks 30, legend 30, spines 3,
major ticks 12, minor 6, labelpad 15, tick pad 10); ``PANEL`` below sets the
actual panel size and every one of those numbers is scaled by
``PANEL/12`` so the ratios are preserved rather than re-guessed per figure.

usetex IS SAFE HERE and would not be in a batch job. LaTeX spawns a subprocess
per unique string and deadlocks under multiprocessing; this script renders six
figures in one process, serially, so it is on the right side of that line. If
this ever grows a per-frame loop, switch to ``mathtext.fontset='cm'`` and drop
usetex rather than parallelising around it.

LINE STYLE, one system across every panel:

    TC1D-FEM            solid line
    ABDS-CN             open circles
    asymptotic theory   dotted, grey

Theory is the hypothesis under test, not the reference: it is styled as the
quiet curve, and the numerics are not drawn as deviations from it.

LOG AXES DROP NON-POSITIVE VALUES SILENTLY. Section 3 plots the deficit
1 - V(T) on log-log, and V can exceed 1 numerically -- their own L = infinity
script overshoots by ~5e-3 at the default truncation. A point with 1 - V <= 0
would vanish from a log axis with no error and the curve would look truncated
rather than wrong, so ``poslog`` masks them explicitly and prints what it
dropped. Nothing on a log axis in this file is plotted without going through it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LIT = Path(sys.argv[2]) if len(sys.argv) > 2 else None   # digitised fig. 5

PANEL = 9.0                     # inches per square panel
S = PANEL / 12.0                # scale factor against the 12x12 reference

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

LAB = 40 * S
LABPAD = 15 * S
TICK = 30 * S
TICKPAD = 10 * S
LEG = 30 * S
LW = 3 * S
MLW = 3.0 * S                   # data line width

C_TC = "#0072B2"                # TC1D-FEM
C_AB = "#D55E00"                # ABDS-CN
C_TH = "0.45"                   # theory, muted grey
CYC = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#E69F00"]

DROPPED = []


def axstyle(ax, square=True):
    for s in ax.spines.values():
        s.set_linewidth(LW)
    ax.tick_params(which="major", direction="out", width=LW, length=12 * S,
                   labelsize=TICK, pad=TICKPAD)
    ax.tick_params(which="minor", direction="out", width=LW, length=6 * S)
    ax.minorticks_on()
    if square:
        ax.set_box_aspect(1)


def poslog(x, y, what):
    """Keep only strictly positive (x, y); report what a log axis would eat."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int((~good).sum())
    if n:
        bad = np.flatnonzero(~good)
        msg = (f"    {what}: {n}/{len(x)} points not plottable on a log axis "
               f"(min y = {np.nanmin(y):.3e}); first offending x = "
               f"{x[bad[0]]:.4g}, y = {y[bad[0]]:.4g}")
        print(msg)
        DROPPED.append(msg)
    return x[good], y[good]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load(p):
    """CSV -> record array, tolerant of a leading comment block.

    The digitised fig. 5 files carry a long provenance header of ``#`` lines.
    ``genfromtxt(names=True)`` takes its field names from the FIRST line and
    strips a leading ``#`` from it, so it happily adopts a sentence of English
    prose as the column names and then fails on every data row. Strip the
    comments first and hand it only the real header.
    """
    p = Path(p)
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return None
    return np.genfromtxt(lines, delimiter=",", names=True)


def nearest_profile(d, prefix, Twant, suffix=""):
    """Find the profile file whose recorded T is closest to Twant.

    Profile files are named by the time the solver ACTUALLY reached, not the
    time that was asked for: the output grid is geometric and each requested
    time is snapped to the nearest grid point, and the adaptive stepper then
    lands a hair short of it. So ``..._T12_H.csv`` generally does not exist
    while ``..._T11.9513_H.csv`` does, and an exact-name lookup silently finds
    nothing at all but the one case where the arithmetic came out round.
    """
    pat = re.compile(re.escape(prefix) + r"_T([0-9.eE+-]+)" +
                     re.escape(suffix) + r"\.csv$")
    best, bestd = None, None
    for f in Path(d).glob(f"{prefix}_T*{suffix}.csv"):
        m = pat.search(f.name)
        if not m:
            continue
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        e = abs(t - Twant)
        if bestd is None or e < bestd:
            best, bestd, bt = f, e, t
    if best is None or bestd > 0.05 * max(1.0, Twant):
        return None, None
    return best, bt


# ===========================================================================
print("fig1_earlytime")
fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL + 2.5, PANEL))
for i, (L, lab) in enumerate(((0.1, r"$\mathcal{L}=0.1$"),
                              (1.0, r"$\mathcal{L}=1$"),
                              (20.0, r"$\mathcal{L}=20$"))):
    a = load(ROOT / "analysis" / f"sec1_L{L:g}.csv")
    if a is None:
        continue
    c = CYC[i]
    x, y = poslog(a["T"], a["V_TC1D_FEM"], f"fig1 V TC1D-FEM L={L:g}")
    axs[0].plot(x, y, "-", color=c, lw=MLW, label=f"TC1D-FEM, {lab}", zorder=3)
    k = max(1, len(a["T"]) // 22)
    x, y = poslog(a["T"][::k], a["V_ABDS_CN"][::k], f"fig1 V ABDS-CN L={L:g}")
    axs[0].plot(x, y, "o", mfc="none", mec=c, mew=MLW * 0.7,
                ms=13 * S, ls="none", zorder=4,
                label=f"ABDS-CN, {lab}")
    axs[1].semilogx(a["T"], 100 * a["rel_TC1D_FEM"], "-", color=c, lw=MLW)
    axs[1].semilogx(a["T"][::k], 100 * a["rel_ABDS_CN"][::k], "o", mfc="none",
                    mec=c, mew=MLW * 0.7, ms=13 * S, ls="none")
# Eq.~(12) is ``eq:diffusive`` in docs/validation-1d/validation-1d.tex.
Tt = np.geomspace(1e-4, 1.0, 200)
axs[0].plot(Tt, 2 * np.sqrt(Tt / np.pi), ":", color=C_TH, lw=MLW * 1.2,
            zorder=2, label=r"Eq.~(12), $2\sqrt{T/\pi}$")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$T$", fontsize=LAB, labelpad=LABPAD)
axs[0].set_ylabel(r"$\mathrm{d}X_c/\mathrm{d}T$", fontsize=LAB, labelpad=LABPAD)
axs[0].legend(fontsize=LEG * 0.62, loc="upper left", frameon=False,
              handlelength=1.6, labelspacing=0.25)
axs[1].axhline(0.0, color=C_TH, ls=":", lw=MLW * 1.2)
axs[1].set_ylim(-30, 5)
axs[1].set_xlabel(r"$T$", fontsize=LAB, labelpad=LABPAD)
axs[1].set_ylabel(r"deviation from Eq.~(12)  [\%]", fontsize=LAB * 0.8,
                  labelpad=LABPAD)
for ax in axs:
    axstyle(ax)
axs[0].text(0.95, 0.06, r"(a)", transform=axs[0].transAxes, fontsize=LAB,
            ha="right")
axs[1].text(0.05, 0.06, r"(b)", transform=axs[1].transAxes, fontsize=LAB)
save(fig, "fig1_earlytime")


# ===========================================================================
print("fig2_deceleration")
# Included at textwidth this ~23" figure is scaled by ~0.28, so ticks/legends
# must be oversized here. A stacked dfrac ylabel is wider than the panel gap
# and sits on (a)'s right spine; use a one-line ratio instead.
fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL + 5.0, PANEL),
                        gridspec_kw={"wspace": 0.50})
TICK2, LEG2, LAB2 = 42, 40, 40
for i, L in enumerate((1.0, 5.0, 20.0, 40.0)):
    a = load(ROOT / "analysis" / f"sec2_L{L:g}.csv")
    if a is None:
        continue
    c = CYC[i]
    lab = rf"$\mathcal{{L}}={L:g}$"
    x, y = poslog(a["T"], a["V_TC1D_FEM"], f"fig2 V TC1D-FEM L={L:g}")
    axs[0].plot(x, y, "-", color=c, lw=MLW, label=f"TC1D-FEM, {lab}", zorder=3)
    k = max(1, len(a["T"]) // 20)
    x, y = poslog(a["T"][::k], a["V_ABDS_CN"][::k], f"fig2 V ABDS-CN L={L:g}")
    axs[0].plot(x, y, "o", mfc="none", mec=c, mew=MLW * 0.7, ms=13 * S,
                ls="none", zorder=4)
    x, y = poslog(a["T"], a["U_eq6p10"], f"fig2 theory L={L:g}")
    axs[0].plot(x, y, ":", color=C_TH, lw=MLW * 1.2, zorder=2)
    axs[1].semilogx(a["T"], a["ratio_TC1D_FEM"], "-", color=c, lw=MLW,
                    label=lab)
    axs[1].semilogx(a["T"][::k], a["ratio_ABDS_CN"][::k], "o", mfc="none",
                    mec=c, mew=MLW * 0.7, ms=13 * S, ls="none")
axs[0].plot([], [], ":", color=C_TH, lw=MLW * 1.2,
            label=r"Eq.~(6.10), $\mathcal{L}/(1+T)^2$")
axs[0].plot([], [], "o", mfc="none", mec="k", mew=MLW * 0.7, ms=13 * S,
            ls="none", label="ABDS-CN")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$T$", fontsize=LAB2, labelpad=LABPAD)
axs[0].set_ylabel(r"$\mathrm{d}X_c/\mathrm{d}T$", fontsize=LAB2, labelpad=LABPAD)
axs[0].legend(fontsize=LEG2, loc="lower left", frameon=False,
              handlelength=1.8, labelspacing=0.35)
axs[1].axhline(1.0, color=C_TH, ls=":", lw=MLW * 1.2)
axs[1].set_ylim(0.9, 2.2)
axs[1].set_xlabel(r"$T$", fontsize=LAB2, labelpad=LABPAD)
axs[1].set_ylabel(r"$(\mathrm{d}X_c/\mathrm{d}T)\,/\,[\mathcal{L}(1+T)^{-2}]$",
                  fontsize=LAB2, labelpad=LABPAD * 1.2)
axs[1].legend(fontsize=LEG2, loc="upper right", frameon=False,
              handlelength=1.8, labelspacing=0.35)
for ax in axs:
    axstyle(ax)
    ax.tick_params(which="major", labelsize=TICK2, pad=12)
axs[0].text(0.03, 0.965, r"(a)", transform=axs[0].transAxes, fontsize=LAB,
            va="top")
axs[1].text(0.03, 0.965, r"(b)", transform=axs[1].transAxes, fontsize=LAB,
            va="top")
save(fig, "fig2_deceleration")


# ===========================================================================
# THE KEY FIGURE. Panel (a) is the deficit against (7.11); panel (b) is the
# same data as (1+T)^2 (1-V), which is the constant (7.11) asserts is 1.
#
# The ABDS-CN curve drawn here is the RESOLVED one (dx = 0.0025). Drawing their
# scheme at their own stated grid (dx = L/N = 10/500 = 0.02) would be drawing a
# discretisation error: at dx = 0.02 the deficit at T = 40 comes out NEGATIVE.
# That is not a criticism hidden in a footnote -- it is the whole reason the
# published coefficient is 1 -- so the dx sequence is shown explicitly in (b)
# rather than replaced by a single converged curve.
print("fig3_latetime_V")
a = load(ROOT / "analysis" / "sec3_deficit.csv")
ref = load(ROOT / "sec3-refine" / "ref_dx0p0025_series.csv")
tcf = load(ROOT / "sec3-main" / "tcfem_Linf_dt2_series.csv")
if a is not None and tcf is not None:
    fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL + 2.5, PANEL))
    T, d = tcf["T"], 1.0 - tcf["V"]
    m = T >= 1.0
    x, y = poslog(T[m], d[m], "fig3a deficit TC1D-FEM")
    axs[0].plot(x, y, "-", color=C_TC, lw=MLW, label="TC1D-FEM", zorder=4)
    if ref is not None:
        mr = ref["T"] >= 1.0
        k = max(1, int(mr.sum()) // 22)
        x, y = poslog(ref["T"][mr][::k], 1.0 - ref["V"][mr][::k],
                      "fig3a deficit ABDS-CN dx=0.0025")
        axs[0].plot(x, y, "o", mfc="none", mec=C_AB, mew=MLW * 0.7, ms=14 * S,
                    ls="none", zorder=5,
                    label=r"ABDS-CN, $\Delta X=0.0025$")
    Tg = np.geomspace(1.0, 55, 200)
    axs[0].plot(Tg, 1.0 / (1 + Tg) ** 2, ":", color=C_TH, lw=MLW * 1.4,
                zorder=2, label=r"Eq.~(7.11), $(1+T)^{-2}$")
    axs[0].plot(Tg, 1.5 / (1 + Tg) ** 2, "--", color="#8B5E00", lw=MLW * 1.1,
                zorder=3, label=r"$\tfrac{3}{2}(1+T)^{-2}$ (this work)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$T$", fontsize=LAB, labelpad=LABPAD)
    axs[0].set_ylabel(r"$1-\mathrm{d}X_c/\mathrm{d}T$", fontsize=LAB,
                      labelpad=LABPAD)
    axs[0].legend(fontsize=LEG * 0.66, loc="lower left", frameon=False,
                  handlelength=1.8, labelspacing=0.3)

    # panel (b): the coefficient, and how the ABDS-CN grid controls it
    axs[1].semilogx(T[m], (1 + T[m]) ** 2 * d[m], "-", color=C_TC, lw=MLW,
                    zorder=5, label="TC1D-FEM")
    greys = ["#BBBBBB", "#999999", "#777777", "#555555"]
    for i, (dxs, lab) in enumerate(
            (("0p02", r"$\Delta X=0.02$ (their grid)"),
             ("0p01", r"$\Delta X=0.01$"),
             ("0p005", r"$\Delta X=0.005$"),
             ("0p0025", r"$\Delta X=0.0025$"))):
        r = load(ROOT / "sec3-refine" / f"ref_dx{dxs}_series.csv")
        if r is None:
            continue
        mr = r["T"] >= 1.0
        axs[1].semilogx(r["T"][mr], (1 + r["T"][mr]) ** 2 * (1 - r["V"][mr]),
                        "-" if i < 3 else "--", color=C_AB, lw=MLW * 0.75,
                        alpha=min(1.0, 0.35 + 0.2 * i), zorder=3, label="ABDS-CN, " + lab)
    axs[1].axhline(1.0, color=C_TH, ls=":", lw=MLW * 1.4, zorder=2,
                   label=r"Eq.~(7.11): $-U_{-2}=1$")
    axs[1].axhline(1.5, color="#8B5E00", ls="--", lw=MLW * 1.1, zorder=2,
                   label=r"$-U_{-2}=3/2$ (this work)")
    axs[1].set_ylim(0.0, 1.8)
    axs[1].set_xlabel(r"$T$", fontsize=LAB, labelpad=LABPAD)
    axs[1].set_ylabel(r"$(1+T)^2\left(1-\mathrm{d}X_c/\mathrm{d}T\right)$",
                      fontsize=LAB * 0.85, labelpad=LABPAD)
    axs[1].legend(fontsize=LEG * 0.52, loc="lower left", frameon=False,
                  handlelength=1.8, labelspacing=0.22)
    for ax in axs:
        axstyle(ax)
    axs[0].text(0.95, 0.92, r"(a)", transform=axs[0].transAxes, fontsize=LAB,
                ha="right")
    axs[1].text(0.95, 0.92, r"(b)", transform=axs[1].transAxes, fontsize=LAB,
                ha="right")
    save(fig, "fig3_latetime_V")


# ===========================================================================
print("fig3_latetime_profiles")
D3 = ROOT / "sec3-main"
want = [2, 5, 12, 28, 50]
fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL + 2.5, PANEL))
drew = 0
for i, Tw in enumerate(want):
    fh, Tt = nearest_profile(D3, "tcfem_Linf_profile", Tw, "_H")
    fv, _ = nearest_profile(D3, "tcfem_Linf_profile", Tw, "_V")
    if fh is None or fv is None:
        continue
    drew += 1
    c = CYC[i % len(CYC)]
    H, V = load(fh), load(fv)
    XM = 12.0
    m = H["Xhat"] <= XM
    mv = V["Xhat"] <= XM
    axs[0].plot(H["Xhat"][m], H["H"][m], "-", color=c, lw=MLW,
                label=rf"$T={Tt:.1f}$", zorder=3)
    # theory at FULL resolution -- subsampling a dotted line makes it look
    # like a broken curve rather than a dotted one
    axs[0].plot(H["Xhat"][m], H["H_eq7p13"][m], ":", color=C_TH,
                lw=MLW * 1.3, zorder=2)
    axs[1].plot(V["Xhat"][mv], V["V"][mv], "-", color=c, lw=MLW, zorder=3)
    axs[1].plot(V["Xhat"][mv], V["V_eq7p14"][mv], ":", color=C_TH,
                lw=MLW * 1.3, zorder=2)
    fa, _ = nearest_profile(D3, "abds_Linf_profile", Tt)
    if fa is not None:
        A = load(fa)
        ma = A["Xhat"] <= XM
        kk = max(1, int(ma.sum()) // 11)
        axs[0].plot(A["Xhat"][ma][::kk], A["H"][ma][::kk], "o", mfc="none",
                    mec=c, mew=MLW * 0.7, ms=13 * S, ls="none", zorder=4)
        axs[1].plot(A["Xhat"][ma][::kk], A["V"][ma][::kk], "o", mfc="none",
                    mec=c, mew=MLW * 0.7, ms=13 * S, ls="none", zorder=4)
if drew:
    axs[0].plot([], [], ":", color=C_TH, lw=MLW * 1.3, label=r"Eq.~(7.13)")
    axs[0].plot([], [], "o", mfc="none", mec="k", mew=MLW * 0.7, ms=13 * S,
                ls="none", label="ABDS-CN")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\hat{X}=X-X_c$", fontsize=LAB, labelpad=LABPAD)
    axs[0].set_ylabel(r"$H$", fontsize=LAB, labelpad=LABPAD)
    axs[0].legend(fontsize=LEG * 0.6, loc="upper right", frameon=False,
                  handlelength=1.6, labelspacing=0.22, ncol=2)
    axs[1].plot([], [], ":", color=C_TH, lw=MLW * 1.3, label=r"Eq.~(7.14)")
    axs[1].plot([], [], "o", mfc="none", mec="k", mew=MLW * 0.7, ms=13 * S,
                ls="none", label="ABDS-CN")
    axs[1].set_xlabel(r"$\hat{X}=X-X_c$", fontsize=LAB, labelpad=LABPAD)
    axs[1].set_ylabel(r"$V$", fontsize=LAB, labelpad=LABPAD)
    axs[1].legend(fontsize=LEG * 0.7, loc="upper right", frameon=False,
                  handlelength=1.6)
    for ax in axs:
        axstyle(ax)
    axs[0].text(0.05, 0.06, r"(a)", transform=axs[0].transAxes, fontsize=LAB)
    axs[1].text(0.05, 0.06, r"(b)", transform=axs[1].transAxes, fontsize=LAB)
    save(fig, "fig3_latetime_profiles")
else:
    plt.close(fig)
    print("  no profiles found")


# ===========================================================================
print("fig4_Tc_vs_L")
t = load(ROOT / "analysis" / "sec4_fig5_tcfem.csv")
b = load(ROOT / "analysis" / "sec4_fig5_abds.csv")
if t is not None:
    fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL + 2.5, PANEL))
    Lg = np.geomspace(2.5, 60, 50)
    axs[0].plot(Lg, Lg, ":", color=C_TH, lw=MLW * 1.3,
                label=r"$T_c=\mathcal{L}$", zorder=2)
    x, y = poslog(t["L"], t["Tc"], "fig4 Tc TC1D-FEM")
    axs[0].plot(x, y, "-", color=C_TC, lw=MLW, label="TC1D-FEM", zorder=3)
    if b is not None:
        x, y = poslog(b["L"], b["Tc"], "fig4 Tc ABDS-CN")
        axs[0].plot(x, y, "o", mfc="none", mec=C_AB, mew=MLW * 0.7, ms=14 * S,
                    ls="none", label="ABDS-CN", zorder=4)
    axs[1].semilogx(t["L"], t["dTc_code"], "-", color=C_TC, lw=MLW,
                    label="TC1D-FEM", zorder=3)
    if b is not None:
        axs[1].semilogx(b["L"], b["dTc_code"], "o", mfc="none", mec=C_AB,
                        mew=MLW * 0.7, ms=14 * S, ls="none", label="ABDS-CN",
                        zorder=4)
    if LIT is not None:
        da = load(LIT / "ahsan_fig5a_Tc_vs_L.csv")
        db = load(LIT / "ahsan_fig5b_dTc_vs_L.csv")
        if da is not None:
            axs[0].plot(da["L"], da["Tc"], "s", mfc="none", mec="k",
                        mew=MLW * 0.55, ms=12 * S, ls="none", zorder=5,
                        label=r"their Fig.~5(a), digitised")
        if db is not None:
            axs[1].plot(db["L"], db["dTc"], "s", mfc="none", mec="k",
                        mew=MLW * 0.55, ms=12 * S, ls="none", zorder=5,
                        label=r"their Fig.~5(b), digitised")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\mathcal{L}$", fontsize=LAB, labelpad=LABPAD)
    axs[0].set_ylabel(r"$T_c$", fontsize=LAB, labelpad=LABPAD)
    axs[0].legend(fontsize=LEG * 0.7, loc="upper left", frameon=False,
                  handlelength=1.6, labelspacing=0.3)
    axs[1].set_xlabel(r"$\mathcal{L}$", fontsize=LAB, labelpad=LABPAD)
    axs[1].set_ylabel(r"$\Delta T_c$", fontsize=LAB, labelpad=LABPAD)
    axs[1].legend(fontsize=LEG * 0.7, loc="lower right", frameon=False,
                  handlelength=1.6, labelspacing=0.3)
    for ax in axs:
        axstyle(ax)
    axs[0].text(0.95, 0.06, r"(a)", transform=axs[0].transAxes, fontsize=LAB,
                ha="right")
    axs[1].text(0.05, 0.92, r"(b)", transform=axs[1].transAxes, fontsize=LAB)
    save(fig, "fig4_Tc_vs_L")


# ===========================================================================
print("figA1_port_fidelity")
pfr = ROOT / "appendix-port-fidelity" / "port_fidelity.csv"
if pfr.exists():
    import csv
    rows = [r for r in csv.DictReader(pfr.open())
            # THE FRAME-TIME COLUMN IS EXCLUDED, and deliberately. Their save
            # blocks tag each frame with a uniform linspace rather than the
            # actual time of the state, so `t` differs by exactly one frame by
            # construction. That is a documented bookkeeping difference in the
            # port, not a difference in the scheme, and plotting it would put
            # two 1e-2 bars next to twenty-three 5e-9 ones.
            if r["quantity"] != "t"]
    # NORMALISED, max|o-p| / max|o|, not absolute and not pointwise relative.
    # Absolute is not comparable across quantities spanning H ~ 121 down to
    # V ~ 1e-3. Pointwise relative is worse: hh(:,end) and the V profile pass
    # through zero in the far field, so it divides a write-precision difference
    # by ~0 and reports 1e-5 where nothing is wrong. Octave's `save -ascii`
    # keeps a fixed number of SIGNIFICANT digits, so the normalised difference
    # has one meaningful floor for every row.
    labs = []
    vals = []
    for r in rows:
        c = r["case"]
        tag = (r"$\infty$, $N{=}$" + c.split("N=")[1].split()[0]
               if "L=infty" in c else
               r"$\mathcal{L}{=}$" + c.split("L=")[1].split()[0])
        labs.append(f"{tag}\n{r['quantity']}")
        vals.append(max(float(r["max_norm_diff"]), 1e-16))
    fig, ax = plt.subplots(figsize=(2 * PANEL + 2.5, PANEL * 0.85))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=C_TC, edgecolor="k", linewidth=LW * 0.5)
    ax.axhline(5e-9, color=C_AB, ls="--", lw=MLW,
               label=r"Octave \texttt{save -ascii} resolution "
                     r"($\sim$9 significant digits)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=90, fontsize=TICK * 0.40)
    ax.set_ylabel(r"normalised $|\,$ABDS-CN $-$ Octave$\,|$",
                  fontsize=LAB * 0.8, labelpad=LABPAD)
    ax.set_ylim(1e-11, 1e-6)
    ax.legend(fontsize=LEG * 0.6, loc="upper left", frameon=False)
    axstyle(ax, square=False)
    ax.tick_params(axis="x", labelsize=TICK * 0.40, pad=TICKPAD * 0.4)
    save(fig, "figA1_port_fidelity")

print("\nlog-axis points dropped:")
print("  none" if not DROPPED else "\n".join(DROPPED))
(FIG / "_log_axis_dropped.txt").write_text(
    "\n".join(DROPPED) if DROPPED else "none\n")
