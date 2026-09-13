#!/usr/bin/env python3
r"""Reduce the campaign CSVs to the numbers RESULTS.md quotes.

Reads only the files the campaign wrote; runs no solver. Everything printed
here is a number, not a verdict -- the verdicts live in RESULTS.md and can be
checked against this output.

WHAT IS COMPARED, AND HOW

Section 1  V(T) against (5.10) 2 sqrt(T/pi). Reported as the relative
           deviation V/(2 sqrt(T/pi)) - 1, and as the T at which that
           deviation first exceeds 1%, 5% and 10% going up in T. The two
           windows of each case are spliced at T = 1e-2.
Section 2  V(T) against (6.10) L/(1+T)^2 and L - Xc against L/(1+T). Because
           (6.10) is exact in the limit L - Xc -> 0 rather than an
           approximation with a free constant, the useful number is the RATE
           at which the ratio approaches 1, not just its value.
Section 3  THE DEFICIT, NOT A POWER LAW. 1 - V(T) is compared directly with
           1/(1+T)^2 from (7.11), and both the ratio (1-V)/(1+T)^-2 and the
           absolute difference are tabulated against T. A pure-power fit in T
           is also reported, but only to show how slowly it converges: (7.11)
           expands as 1 - V = T^-2 - 2 T^-3 + O(T^-4), so a local slope
           d ln(1-V)/d ln T reaches -2 only as T -> infinity and reads about
           -2 - 2/T at finite T. Fitting exponents here measures the fit
           window, not the physics.
           Also: H(Xc,T) against (3.10a) 1+T, Xc(T) against (7.17) T+1, and
           the profiles against (7.13)/(7.14).
Section 4  T_c and Delta T_c by their own estimator, both codes, against
           T_c = L.

THE U_{-2} QUESTION. Sec. VII derives U_0 = 1 and U_{-1} = 0 and states that
U_{-2} "cannot be obtained by extending our analysis"; (7.11) is then an
empirical resummation equivalent to choosing U_{-2} = -1, "for the initial
conditions in our numerical solution". That constant is therefore a NUMBER TO
MEASURE, not a result to reproduce, and this script measures it three ways:
a plateau in T^2 (1 - V), a two-term regression 1 - V = A/T^2 + B/T^3, and a
one-parameter shifted-origin fit 1 - V = (1 + T + c)^-2. The third is the
sharpest test of whether (7.11)'s particular resummation is the right one.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# np.trapezoid is NumPy >= 2.0; np.trapz is the pre-2.0 spelling and is removed
# in 2.x. Bind once so this runs on either without a version pin.
_trapz = getattr(np, "trapezoid", None) or np.trapz

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
S = {}                       # everything printed, also dumped to summary.json


def load(p):
    """CSV -> record array, tolerant of a leading ``#`` comment block."""
    p = Path(p)
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    return np.genfromtxt(lines, delimiter=",", names=True) if lines else None


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


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def splice(a, b, Tsplit=1e-2):
    """Join the two early-time windows at Tsplit, always as a (T, V) tuple.

    Returning the raw record array when one window is missing looks harmless
    and then unpacks wrong three lines later, so both branches are normalised.
    """
    if a is None and b is None:
        return None
    if a is None:
        return np.asarray(b["T"], float), np.asarray(b["V"], float)
    if b is None:
        return np.asarray(a["T"], float), np.asarray(a["V"], float)
    T = np.concatenate([a["T"][a["T"] < Tsplit], b["T"][b["T"] >= Tsplit]])
    V = np.concatenate([a["V"][a["T"] < Tsplit], b["V"][b["T"] >= Tsplit]])
    o = np.argsort(T)
    return T[o], V[o]


def first_exceed(T, rel, thr):
    """Smallest T at which |rel| exceeds thr and stays above it."""
    bad = np.abs(rel) > thr
    if not bad.any():
        return float("nan")
    # Walk back from the LAST above-threshold sample, not from the end of the
    # array, so a single noisy early point does not win. Anchoring at
    # len(bad)-1 is wrong whenever the final sample is BELOW threshold: the
    # backward scan then starts inside a good run, the while loop exits at
    # once, and the function returns the last time in the record instead of
    # the start of the final bad run.
    i = int(np.flatnonzero(bad)[-1])
    while i > 0 and bad[i - 1]:
        i -= 1
    return float(T[i])


# ===========================================================================
rule("SECTION 1 -- early time, eq (5.10)  dXc/dT ~ 2 sqrt(T/pi)")
sec1 = {}
for L, tag in ((0.1, "L0p1"), (1.0, "L1"), (20.0, "L20")):
    d = ROOT / "sec1-early"
    tc = splice(load(d / f"tcfem_{tag}_wA_series.csv"),
                load(d / f"tcfem_{tag}_wB_series.csv"))
    ab = splice(load(d / f"abds_{tag}_wA_series.csv"),
                load(d / f"abds_{tag}_wB_series.csv"))
    if tc is None or ab is None:
        print(f"  L = {L}: MISSING")
        continue
    Tt, Vt = tc
    Ta, Va = ab
    # common grid: the TC1D-FEM output times, which are the sparser set
    m = (Tt >= 1e-4) & (Tt <= 1.0) & (Tt >= Ta.min()) & (Tt <= Ta.max())
    T = Tt[m]
    Vt2 = Vt[m]
    Va2 = np.interp(T, Ta, Va)
    th = 2.0 * np.sqrt(T / math.pi)
    rt, ra = Vt2 / th - 1.0, Va2 / th - 1.0
    dcode = np.abs(Vt2 - Va2)
    np.savetxt(OUT / f"sec1_L{L:g}.csv",
               np.column_stack([T, Vt2, Va2, th, rt, ra, dcode]),
               delimiter=",", comments="",
               header="T,V_TC1D_FEM,V_ABDS_CN,U_eq5p10,rel_TC1D_FEM,"
                      "rel_ABDS_CN,abs_diff_codes")
    e = {"L": L,
         "T_dev_1pct_tcfem": first_exceed(T, rt, 0.01),
         "T_dev_5pct_tcfem": first_exceed(T, rt, 0.05),
         "T_dev_10pct_tcfem": first_exceed(T, rt, 0.10),
         "T_dev_1pct_abds": first_exceed(T, ra, 0.01),
         "T_dev_5pct_abds": first_exceed(T, ra, 0.05),
         "T_dev_10pct_abds": first_exceed(T, ra, 0.10),
         "max_abs_code_diff": float(dcode.max()),
         "max_abs_code_diff_at_T": float(T[int(dcode.argmax())]),
         "rel_tcfem_at_1e-3": float(np.interp(1e-3, T, rt)),
         "rel_tcfem_at_1e-2": float(np.interp(1e-2, T, rt)),
         "rel_tcfem_at_1e-1": float(np.interp(1e-1, T, rt)),
         "rel_abds_at_1e-3": float(np.interp(1e-3, T, ra)),
         "rel_abds_at_1e-2": float(np.interp(1e-2, T, ra)),
         "rel_abds_at_1e-1": float(np.interp(1e-1, T, ra))}
    sec1[f"L{L:g}"] = e
    print(f"\n  L = {L:g}   ({len(T)} points, T in [{T.min():.3g}, {T.max():.3g}])")
    print(f"    rel. dev. from (5.10)   T=1e-3   T=1e-2   T=1e-1")
    print(f"      TC1D-FEM            {e['rel_tcfem_at_1e-3']:+8.3%} "
          f"{e['rel_tcfem_at_1e-2']:+8.3%} {e['rel_tcfem_at_1e-1']:+8.3%}")
    print(f"      ABDS-CN             {e['rel_abds_at_1e-3']:+8.3%} "
          f"{e['rel_abds_at_1e-2']:+8.3%} {e['rel_abds_at_1e-1']:+8.3%}")
    print(f"    (5.10) departs by 1% / 5% / 10% above T = "
          f"{e['T_dev_1pct_abds']:.3g} / {e['T_dev_5pct_abds']:.3g} / "
          f"{e['T_dev_10pct_abds']:.3g}   [ABDS-CN]")
    print(f"    code-code max |dV| = {e['max_abs_code_diff']:.3e} at "
          f"T = {e['max_abs_code_diff_at_T']:.3g}")
S["section1"] = sec1


# ===========================================================================
rule("SECTION 2 -- late time, finite L, eq (6.10)  dXc/dT ~ L/(1+T)^2")
sec2 = {}
for L in (1.0, 5.0, 20.0, 40.0):
    d = ROOT / "sec2-late-finite"
    tc, ab = load(d / f"tcfem_L{L:g}_series.csv"), load(d / f"abds_L{L:g}_series.csv")
    if tc is None or ab is None:
        print(f"  L = {L}: MISSING")
        continue
    T = tc["T"]
    Vt, Va = tc["V"], np.interp(T, ab["T"], ab["V"])
    th = L / (1.0 + T) ** 2
    Lm_t, Lm_a = tc["L_minus_Xc"], np.interp(T, ab["T"], ab["L_minus_Xc"])
    Lm_th = L / (1.0 + T)
    np.savetxt(OUT / f"sec2_L{L:g}.csv",
               np.column_stack([T, Vt, Va, th, Vt / th, Va / th,
                                Lm_t, Lm_a, Lm_th, np.abs(Vt - Va)]),
               delimiter=",", comments="",
               header="T,V_TC1D_FEM,V_ABDS_CN,U_eq6p10,ratio_TC1D_FEM,"
                      "ratio_ABDS_CN,LmXc_TC1D_FEM,LmXc_ABDS_CN,"
                      "LmXc_eq6p10,abs_diff_codes")
    pts = [T[-1] * f for f in (0.25, 0.5, 1.0)]
    e = {"L": L, "Tend": float(T[-1]),
         "LmXc_end": float(Lm_t[-1]),
         "max_abs_code_diff_T_gt_1": float(np.max(np.abs(Vt - Va)[T > 1.0])),
         "ratio_at": {f"{p:.3g}": float(np.interp(p, T, Vt / th)) for p in pts},
         "ratio_abds_at": {f"{p:.3g}": float(np.interp(p, T, Va / th)) for p in pts},
         "LmXc_ratio_end": float(Lm_t[-1] / Lm_th[-1])}
    sec2[f"L{L:g}"] = e
    print(f"\n  L = {L:g}, Tend = {T[-1]:g}, L - Xc(Tend) = {Lm_t[-1]:.4f}")
    print("    V / (6.10)    " + "  ".join(
        f"T={p:.3g}: {e['ratio_at'][f'{p:.3g}']:.5f}" for p in pts))
    print("    ABDS-CN       " + "  ".join(
        f"T={p:.3g}: {e['ratio_abds_at'][f'{p:.3g}']:.5f}" for p in pts))
    print(f"    (L-Xc)/(L/(1+T)) at Tend = {e['LmXc_ratio_end']:.6f}")
    print(f"    code-code max |dV| (T>1) = {e['max_abs_code_diff_T_gt_1']:.3e}")

# the deep case: how fast does the ratio approach 1?
d = ROOT / "sec2-late-finite"
tc, ab = load(d / "tcfem_L5_deep_series.csv"), load(d / "abds_L5_deep_series.csv")
if tc is not None:
    T, V = tc["T"], tc["V"]
    r = V / (5.0 / (1.0 + T) ** 2)
    rows = []
    print("\n  L = 5 pushed to T = 400 (approach rate of (6.10)):")
    print(f"    {'T':>7} {'L-Xc':>10} {'V/(6.10)':>11} {'(V/th - 1)*(1+T)':>18}")
    for p in (10, 20, 40, 80, 160, 320, 400):
        if p <= T.max():
            rr = float(np.interp(p, T, r))
            lm = float(np.interp(p, T, tc["L_minus_Xc"]))
            rows.append([p, lm, rr, (rr - 1) * (1 + p)])
            print(f"    {p:7g} {lm:10.5f} {rr:11.6f} {(rr-1)*(1+p):18.5f}")
    np.savetxt(OUT / "sec2_L5_deep_approach.csv", np.array(rows), delimiter=",",
               comments="", header="T,LmXc,ratio_V_over_eq6p10,scaled_error")
    if ab is not None:
        dd = np.abs(V - np.interp(T, ab["T"], ab["V"]))
        print(f"    code-code max |dV| (T>1) = {np.max(dd[T > 1]):.3e}")
        S["sec2_deep_code_diff"] = float(np.max(dd[T > 1]))
    S["sec2_deep"] = rows
S["section2"] = sec2


# ===========================================================================
rule("SECTION 3 -- main regime, L >> T >> 1;  eqs (7.11), (3.10a), (7.17)")
d = ROOT / "sec3-main"
tc = load(d / "tcfem_Linf_series.csv")
ai = load(d / "abds_Linf_series.csv")
af = load(d / "abds_L200_series.csv")
if tc is not None and ai is not None:
    T = tc["T"]
    m = T >= 1.0
    T = T[m]
    Vt = tc["V"][m]
    Va = np.interp(T, ai["T"], ai["V"])
    Vf = np.interp(T, af["T"], af["V"]) if af is not None else np.full_like(T, np.nan)
    dfc_t, dfc_a = 1.0 - Vt, 1.0 - Va
    th = 1.0 / (1.0 + T) ** 2
    np.savetxt(OUT / "sec3_deficit.csv",
               np.column_stack([T, Vt, Va, Vf, dfc_t, dfc_a, th,
                                dfc_t / th, dfc_a / th, dfc_t - th, dfc_a - th,
                                np.abs(Vt - Va)]),
               delimiter=",", comments="",
               header="T,V_TC1D_FEM,V_ABDS_CN_inf,V_ABDS_CN_L200,"
                      "deficit_TC1D_FEM,deficit_ABDS_CN,deficit_eq7p11,"
                      "ratio_TC1D_FEM,ratio_ABDS_CN,absdiff_TC1D_FEM,"
                      "absdiff_ABDS_CN,codediff")
    print(f"\n  {'T':>6} {'V(TC1D-FEM)':>13} {'V(ABDS-CN)':>12} "
          f"{'1-V':>11} {'1/(1+T)^2':>11} {'ratio':>9} {'abs diff':>11}")
    for p in (1, 2, 5, 10, 15, 20, 30, 40, 50):
        if p < T.max():
            i = int(np.argmin(np.abs(T - p)))
            print(f"  {T[i]:6.2f} {Vt[i]:13.7f} {Va[i]:12.7f} "
                  f"{dfc_t[i]:11.3e} {th[i]:11.3e} {dfc_t[i]/th[i]:9.4f} "
                  f"{dfc_t[i]-th[i]:+11.3e}")
    hi = T > 10
    S["section3"] = {
        "code_diff_max_T_gt_1": float(np.max(np.abs(Vt - Va))),
        "code_diff_max_T_gt_10": float(np.max(np.abs(Vt - Va)[hi])),
        "ratio_TC_T20": float(np.interp(20, T, dfc_t / th)),
        "ratio_TC_T50": float(np.interp(50, T, dfc_t / th)),
        "ratio_AB_T20": float(np.interp(20, T, dfc_a / th)),
        "ratio_AB_T50": float(np.interp(50, T, dfc_a / th)),
        "absdiff_TC_max_T_gt_10": float(np.max(np.abs(dfc_t - th)[hi])),
        "absdiff_AB_max_T_gt_10": float(np.max(np.abs(dfc_a - th)[hi])),
        "V_TC_at_50": float(Vt[-1]), "V_AB_at_50": float(Va[-1]),
    }
    print(f"\n  code-code max |dV|: {S['section3']['code_diff_max_T_gt_1']:.3e} "
          f"(T>1), {S['section3']['code_diff_max_T_gt_10']:.3e} (T>10)")
    print(f"  max |(1-V) - (1+T)^-2| for T>10: TC1D-FEM "
          f"{S['section3']['absdiff_TC_max_T_gt_10']:.3e}, ABDS-CN "
          f"{S['section3']['absdiff_AB_max_T_gt_10']:.3e}")

    # ---- U_{-2}: the constant (7.11) fits rather than derives -------------
    print("\n  U_-2, the constant Sec. VII leaves undetermined and (7.11) sets"
          " to -1")
    print("  (7.11) <=> 1 - V = (1+T)^-2 = T^-2 - 2 T^-3 + O(T^-4), i.e. "
          "U_-2 = -1, U_-3 = +2")
    print(f"    {'T':>6} {'T^2 (1-V)':>12} {'(1+T)^2 (1-V)':>15} "
          f"{'d ln(1-V)/d ln T':>18}")
    slope = np.gradient(np.log(dfc_t), np.log(T))
    fit = []
    for p in (5, 10, 20, 30, 40, 50):
        if p < T.max():
            i = int(np.argmin(np.abs(T - p)))
            fit.append([T[i], T[i] ** 2 * dfc_t[i],
                        (1 + T[i]) ** 2 * dfc_t[i], slope[i]])
            print(f"    {T[i]:6.2f} {T[i]**2*dfc_t[i]:12.6f} "
                  f"{(1+T[i])**2*dfc_t[i]:15.6f} {slope[i]:18.4f}")
    np.savetxt(OUT / "sec3_Um2.csv", np.array(fit), delimiter=",", comments="",
               header="T,T2_times_deficit,one_plus_T2_times_deficit,loglog_slope")
    # two-term regression 1-V = A/T^2 + B/T^3 over T in [20, Tmax]
    w = T > 20
    A = np.column_stack([T[w] ** -2, T[w] ** -3])
    coef, *_ = np.linalg.lstsq(A, dfc_t[w], rcond=None)
    print(f"    regression 1-V = A/T^2 + B/T^3 over T>20: "
          f"A = {coef[0]:.5f} (=> U_-2 = {-coef[0]:.5f}), B = {coef[1]:.4f} "
          f"(=> U_-3 = {-coef[1]:.4f});  (7.11) implies A=1, B=-2")
    # one-parameter shifted-origin fit 1-V = (1+T+c)^-2
    c = np.mean((dfc_t[w] ** -0.5) - 1.0 - T[w])
    resid_c = np.max(np.abs(dfc_t[w] - (1 + T[w] + c) ** -2))
    resid_0 = np.max(np.abs(dfc_t[w] - (1 + T[w]) ** -2))
    print(f"    shifted-origin fit 1-V = (1+T+c)^-2 over T>20: c = {c:+.5f}, "
          f"max resid {resid_c:.3e} vs {resid_0:.3e} at c = 0 "
          f"(factor {resid_0/max(resid_c,1e-300):.1f})")
    S["section3"].update({"A_T2": float(coef[0]), "B_T3": float(coef[1]),
                          "shift_c": float(c), "resid_shift": float(resid_c),
                          "resid_c0": float(resid_0)})

    # ---- (3.10a) and (7.17) ---------------------------------------------
    Ht = tc["H_max"][m]
    Xc = tc["Xc"][m]
    e310 = np.max(np.abs(Ht / (1.0 + T) - 1.0))
    print(f"\n  (3.10a) H(Xc,T) = 1+T   TC1D-FEM max rel dev = {e310:.3e}")
    print(f"  (7.17)  Xc ~ T+1        Xc - (T+1) at T = 10/30/50: "
          f"{np.interp(10,T,Xc)-11:+.4f} / {np.interp(30,T,Xc)-31:+.4f} / "
          f"{np.interp(50,T,Xc)-51:+.4f}")
    S["section3"].update({"eq3p10a_max_rel": float(e310),
                          "Xc_minus_T1_at50": float(Xc[-1] - (T[-1] + 1))})

    # ---- numerical self-consistency -------------------------------------
    print(f"\n  TC1D-FEM invariant (3.1) residual: max "
          f"{np.max(tc['invariant_max']):.3e}, rms "
          f"{np.max(tc['invariant_rms']):.3e}")
    print(f"  TC1D-FEM far-end thickness drift (must stay ~0 here): "
          f"{np.max(tc['drift_H_far']):.3e}")
    S["section3"].update({
        "invariant_max": float(np.max(tc["invariant_max"])),
        "drift_H_far_max": float(np.max(tc["drift_H_far"]))})

# resolution twins
for a, b, lab in (("tcfem_Linf", "tcfem_Linf_dt2", "TC1D-FEM dt/2"),
                  ("tcfem_Linf", "tcfem_Linf_mesh2", "TC1D-FEM mesh x2"),
                  ("abds_Linf", "abds_Linf_coarse", "ABDS-CN dx,dt x2")):
    x, y = load(d / f"{a}_series.csv"), load(d / f"{b}_series.csv")
    if x is None or y is None:
        continue
    mm = x["T"] > 1.0
    dd = np.abs(x["V"][mm] - np.interp(x["T"][mm], y["T"], y["V"]))
    print(f"  resolution twin, {lab}: max |dV| = {dd.max():.3e} (T > 1)")
    S.setdefault("resolution", {})[lab] = float(dd.max())

# Oh independence
base = load(d.parent / "sec2-late-finite" / "tcfem_L5_series.csv")
for oh in ("0p25", "4"):
    x = load(d / f"tcfem_L5_oh{oh}_series.csv")
    if x is None or base is None:
        continue
    mm = (base["T"] > 0.1) & (base["T"] <= x["T"].max())
    dd = np.abs(base["V"][mm] - np.interp(base["T"][mm], x["T"], x["V"]))
    print(f"  Oh-independence at L=5, Oh={oh.replace('p','.')} vs 1: "
          f"max |dV| = {dd.max():.3e}, max rel = "
          f"{np.max(dd/base['V'][mm]):.3e}")
    S.setdefault("oh_independence", {})[oh] = float(dd.max())

# profiles vs (7.13)/(7.14)
print("\n  profiles vs (7.13) H and (7.14) V, TC1D-FEM, over 0 <= Xhat <= T:")
prow = []
for Tp in (1, 2, 5, 10, 12, 20, 28, 40, 50):
    fh, Tt = nearest_profile(d, "tcfem_Linf_profile", Tp, "_H")
    fv, _ = nearest_profile(d, "tcfem_Linf_profile", Tp, "_V")
    if fh is None or fv is None:
        continue
    Tp = Tt
    H, Vv = load(fh), load(fv)
    mh = H["Xhat"] <= Tp
    mv = Vv["Xhat"] <= Tp
    eH = np.max(np.abs(H["H"][mh] / H["H_eq7p13"][mh] - 1.0))
    eV = np.max(np.abs(Vv["V"][mv] - Vv["V_eq7p14"][mv]))
    # RELATIVE deviation only where the theory profile is not asymptotically
    # zero. (7.14) decays to 0 in the far field, so a relative error there is
    # 100% by construction and says nothing; restrict it to V > 0.01.
    big = mv & (Vv["V_eq7p14"] > 0.01)
    eVr = (np.max(np.abs(Vv["V"][big] / Vv["V_eq7p14"][big] - 1.0))
           if big.any() else float("nan"))
    prow.append([Tp, eH, eV, eVr])
    print(f"    T = {Tp:3g}   max rel dev H {eH:8.3%}   max abs dev V "
          f"{eV:9.3e}   max rel dev V {eVr:8.3%}")
if prow:
    np.savetxt(OUT / "sec3_profiles_vs_theory.csv", np.array(prow),
               delimiter=",", comments="",
               header="T,max_rel_dev_H_eq7p13,max_abs_dev_V_eq7p14,"
                      "max_rel_dev_V_eq7p14")
    S["section3_profiles"] = prow

# profile agreement between the two codes
print("\n  profile agreement TC1D-FEM vs ABDS-CN (interpolated onto Xhat):")
prow2 = []
for Tp in (1, 2, 5, 10, 12, 20, 28, 40, 50):
    fh, Tt = nearest_profile(d, "tcfem_Linf_profile", Tp, "_H")
    fa, _ = nearest_profile(d, "abds_Linf_profile", Tp)
    if fh is None or fa is None:
        continue
    Tp = Tt
    H, A = load(fh), load(fa)
    mh = H["Xhat"] <= min(Tp, A["Xhat"].max())
    eH = np.max(np.abs(H["H"][mh] - np.interp(H["Xhat"][mh], A["Xhat"], A["H"])))
    prow2.append([Tp, eH])
    print(f"    T = {Tp:3g}   max |dH| = {eH:.3e}")
if prow2:
    np.savetxt(OUT / "sec3_profiles_code_vs_code.csv", np.array(prow2),
               delimiter=",", comments="", header="T,max_abs_dH")


# ===========================================================================
rule("SECTION 3b -- resolution of the deficit, and the value of U_-2")
# The deficit at T = 50 is 5.8e-4. Any scheme whose error in V is larger than
# that cannot see it at all, so the deficit MUST be resolution-tested before it
# is interpreted. It turns out that ABDS-CN at their own stated grid
# (dx = L/N = 10/500 = 0.02) is nowhere near, and that this is what makes
# (7.11)'s coefficient look like 1.
d3 = ROOT / "sec3-main"
dr = ROOT / "sec3-refine"
TS = (10, 20, 30, 40, 50)


def kappa(f):
    """(1+T)^2 (1-V) at the sample times: -U_-2 if (7.11)'s grouping is right."""
    a = load(f)
    if a is None:
        return None
    out = []
    for p_ in TS:
        i = int(np.argmin(np.abs(a["T"] - p_)))
        out.append((1 + a["T"][i]) ** 2 * (1 - a["V"][i]))
    return np.array(out)


cases = [
    ("ABDS-CN dx=0.02 (their grid)", dr / "ref_dx0p02_series.csv"),
    ("ABDS-CN dx=0.01", dr / "ref_dx0p01_series.csv"),
    ("ABDS-CN dx=0.005", dr / "ref_dx0p005_series.csv"),
    ("ABDS-CN dx=0.0025", dr / "ref_dx0p0025_series.csv"),
    ("ABDS-CN dx=0.005, dt/2", dr / "ref_dx0p005_dt2_series.csv"),
    ("ABDS-CN dx=0.005, Lhat=60", dr / "ref_dx0p005_L60_series.csv"),
    ("TC1D-FEM L=200", d3 / "tcfem_Linf_series.csv"),
    ("TC1D-FEM dt/2", d3 / "tcfem_Linf_dt2_series.csv"),
    ("TC1D-FEM mesh x2", d3 / "tcfem_Linf_mesh2_series.csv"),
]
print("  (1+T)^2 (1 - dXc/dT).  (7.11) says this is 1 at large T.")
print("  " + f"{'case':>30} " + " ".join(f"{'T=' + str(p_):>10}" for p_ in TS))
K, tab = {}, []
for nm, f in cases:
    v = kappa(f)
    if v is None:
        continue
    K[nm] = v
    tab.append([nm] + list(v))
    print("  " + f"{nm:>30} " + " ".join(f"{q:10.5f}" for q in v))
if tab:
    with (OUT / "sec3_resolution.csv").open("w") as fh:
        fh.write("case," + ",".join(f"kappa_T{p_}" for p_ in TS) + "\n")
        for r in tab:
            fh.write(r[0] + "," + ",".join(f"{q:.8g}" for q in r[1:]) + "\n")

need = ("ABDS-CN dx=0.01", "ABDS-CN dx=0.005", "ABDS-CN dx=0.0025")
if all(k in K for k in need):
    a_, b_, c_ = (K[k] for k in need)
    with np.errstate(all="ignore"):
        order = np.log2(np.abs((a_ - b_) / (b_ - c_)))
        ext = c_ + (c_ - b_) / (2 ** order - 1)
    print("\n  Richardson on the ABDS-CN dx sequence (their scheme, our port):")
    print("  " + f"{'observed order':>30} " + " ".join(f"{q:10.3f}" for q in order))
    print("  " + f"{'extrapolated dx -> 0':>30} " + " ".join(f"{q:10.5f}" for q in ext))
    if "TC1D-FEM dt/2" in K:
        dd = np.abs(ext - K["TC1D-FEM dt/2"])
        print("  " + f"{'|extrap - TC1D-FEM|':>30} " + " ".join(f"{q:10.2e}" for q in dd))
        S["Um2_extrap_vs_tcfem"] = float(dd.max())
    S["Um2_abds_extrap"] = ext.tolist()
    S["Um2_abds_order"] = order.tolist()

x = load(d3 / "tcfem_Linf_dt2_series.csv")
if x is not None:
    T, dfc = x["T"], 1.0 - x["V"]
    w = (T > 20) & (T <= 50)
    y = (1 + T[w]) ** 2 * dfc[w]
    for basis, lab in ((["1", "T^-2"], "A + B/T^2"),
                       (["1", "T^-1"], "A + B/T"),
                       (["1", "T^-1", "T^-2"], "A + B/T + C/T^2")):
        cols = [np.ones_like(T[w])]
        if "T^-1" in basis:
            cols.append(1.0 / T[w])
        if "T^-2" in basis:
            cols.append(T[w] ** -2.0)
        M = np.column_stack(cols)
        cf, *_ = np.linalg.lstsq(M, y, rcond=None)
        print(f"    fit (1+T)^2(1-V) = {lab:<18} over 20<T<=50:  A = {cf[0]:.6f}"
              f"   max resid {np.max(np.abs(M @ cf - y)):.2e}")
        S.setdefault("Um2_fits", {})[lab] = float(cf[0])

    print("\n  candidate closed forms, max RELATIVE error in the deficit:")
    forms = {
        "(7.11)   1 - 1/(1+T)^2": lambda t: 1.0 / (1 + t) ** 2,
        "         1 - (3/2)/(1+T)^2": lambda t: 1.5 / (1 + t) ** 2,
        "         1 - (3/2)/(1+T)^2 + 1/(1+T)^3":
            lambda t: 1.5 / (1 + t) ** 2 - 1.0 / (1 + t) ** 3,
    }
    for lo, hi in ((5, 50), (10, 50), (20, 50)):
        w2 = (T >= lo) & (T <= hi)
        print(f"    {lo} <= T <= {hi}:")
        for nm, fn in forms.items():
            e = np.max(np.abs(fn(T[w2]) / dfc[w2] - 1.0))
            print(f"      {nm:<40} {e:8.2%}")
            S.setdefault("form_error", {})[f"{nm.strip()}|{lo}-{hi}"] = float(e)

    # The constant in Xc ~ T - C is C = int_0^inf (1-V) dT, and (7.8)+(7.10)
    # give C = 1 exactly. Any global form for the deficit must integrate to 1;
    # a bare (3/2)(1+T)^-2 integrates to 3/2 and therefore cannot be used down
    # to T = 0 even though it is the correct large-T tail.
    C = float(_trapz(dfc, T)) + 1.5 / (1 + T[-1]) - 0.5 / (1 + T[-1]) ** 2
    print(f"\n  C = int_0^inf (1-V) dT = {C:.5f}   [Xc -> T - C; (7.8)+(7.10) "
          f"give C = 1 exactly]")
    print(f"  measured Xc(T={T[-1]:g}) - T = {x['Xc'][-1] - T[-1]:+.5f}; "
          f"(7.13)+(7.16) give -1 + 1/(1+T) = {-1 + 1 / (1 + T[-1]):+.5f}; "
          f"(7.17) AS PRINTED says +1")
    S["C_integral"] = C
    S["Xc_minus_T_end"] = float(x["Xc"][-1] - T[-1])


# ===========================================================================
rule("SECTION 4 -- their fig. 5:  T_c(L) and Delta T_c(L)")
for code, pref in (("TC1D-FEM", "tcfem"), ("ABDS-CN", "abds")):
    f = OUT / f"sec4_fig5_{pref}.csv"
    if not f.exists():
        print(f"  {code}: {f.name} not built yet (run the fig5 subcommand)")
        continue
    a = load(f)
    print(f"\n  {code}")
    print(f"    {'L':>7} {'Tc':>8} {'Tc/L':>8} {'dTc(code)':>10} "
          f"{'FWHM':>8} {'Umax':>8}")
    for r in np.atleast_1d(a):
        print(f"    {r['L']:7.3g} {r['Tc']:8.3f} {r['Tc_over_L']:8.4f} "
              f"{r['dTc_code']:10.4f} {r['fwhm']:8.4f} {r['Umax']:8.5f}")
    S.setdefault("section4", {})[code] = {
        "L": np.atleast_1d(a)["L"].tolist(),
        "Tc": np.atleast_1d(a)["Tc"].tolist(),
        "Tc_over_L": np.atleast_1d(a)["Tc_over_L"].tolist(),
        "dTc_code": np.atleast_1d(a)["dTc_code"].tolist(),
        "fwhm": np.atleast_1d(a)["fwhm"].tolist(),
        "Umax": np.atleast_1d(a)["Umax"].tolist()}

t = S.get("section4", {}).get("TC1D-FEM")
b = S.get("section4", {}).get("ABDS-CN")
if t and b:
    # Align on the L values both codes actually produced before subtracting:
    # a sweep that drops a case in one code would otherwise broadcast-error
    # or, worse, silently pair mismatched L.
    _Lt = np.asarray(t["L"], float); _Lb = np.asarray(b["L"], float)
    _c = np.intersect1d(_Lt, _Lb)
    _it = np.searchsorted(_Lt, _c); _ib = np.searchsorted(_Lb, _c)
    dt_ = np.abs(np.asarray(t["Tc"])[_it] - np.asarray(b["Tc"])[_ib])
    dd_ = np.abs(np.asarray(t["dTc_code"])[_it] - np.asarray(b["dTc_code"])[_ib])
    print(f"\n  code-code: max |dTc| = {dt_.max():.4f}, "
          f"max |d(Delta Tc)| = {dd_.max():.4f}")
    r = np.array(t["Tc_over_L"])
    print(f"  Tc/L over the sweep: {r.min():.4f} to {r.max():.4f} "
          f"(their fig. 5a dashed line is Tc = L)")
    print(f"  Delta Tc (their code's 2xFWHM) spans "
          f"{min(t['dTc_code']):.3f} to {max(t['dTc_code']):.3f}; "
          f"their fig. 5b ordinate runs 2.4 to 3.1")
    S["section4_summary"] = {"max_dTc_codes": float(dt_.max()),
                             "max_ddTc_codes": float(dd_.max()),
                             "Tc_over_L_min": float(r.min()),
                             "Tc_over_L_max": float(r.max())}

(OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
print(f"\nwrote {OUT}/summary.json and the per-section CSVs")
