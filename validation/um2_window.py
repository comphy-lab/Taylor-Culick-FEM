#!/usr/bin/env python3
"""Adversarial window analysis of the late-time deficit coefficient U_{-2}.

The claim under test is  1 - dXc/dT ~ a (1+T)^{-2}  with a = 3/2 rather than
the a = 1 printed in Ahsan et al. (7.11).  The quantity

    R(T) = (1+T)^2 (1 - V(T))

is measured, and the question is whether R -> 3/2 or whether the apparent
plateau near 1.4986 is a pre-asymptotic transient on its way to 1.

The intercept is basis-independent: if the true expansion is in powers of 1/T
rather than 1/(1+T), then T^2 (1-V) and (1+T)^2 (1-V) share the same T -> inf
limit, so nothing here depends on choosing (1+T) as the small parameter.

Method.  With s = 1/(1+T),

    R(s) = a_2 + a_3 s + a_4 s^2 + ...

is analytic at s = 0 if the deficit has a descending-power expansion.  So a_2
is the s -> 0 intercept of a polynomial fit to R(s).  Fitting polynomials of
degree p = 0..4 on windows [Tmin, Tmax] and watching the intercept as Tmin
moves outward is the drift test: a genuine limit gives an intercept that is
stable in BOTH p and Tmin; a transient gives an intercept that slides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_series(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(d["T"], float), np.asarray(d["V"], float)


def R_of_T(T, V):
    return (1.0 + T) ** 2 * (1.0 - V)


def poly_intercept(T, V, Tmin, Tmax, deg):
    """Least-squares intercept of R as a polynomial in s = 1/(1+T)."""
    m = (T >= Tmin) & (T <= Tmax)
    if m.sum() < deg + 2:
        return np.nan
    s = 1.0 / (1.0 + T[m])
    R = R_of_T(T[m], V[m])
    c = np.polyfit(s, R, deg)
    return float(c[-1])


def window_table(T, V, Tmaxes, degs, label, Tmax_global=None):
    Tmax_global = Tmax_global if Tmax_global is not None else T.max()
    print(f"\n### {label}   (data to T = {T.max():.1f})")
    print("intercept a_2 of R(s) = a_2 + a_3 s + ... fitted on [Tmin, Tmax]")
    hdr = "  Tmin  Tmax " + "".join(f"   deg{d}    " for d in degs)
    print(hdr)
    for Tmin in Tmaxes:
        row = f"{Tmin:6.1f} {Tmax_global:5.1f} "
        for d in degs:
            v = poly_intercept(T, V, Tmin, Tmax_global, d)
            row += f" {v:10.6f}" if np.isfinite(v) else "        n/a"
        print(row)


def local_two_point(T, V):
    """Two-point elimination: assume R = a + b s exactly on adjacent decades.

    Uses the pair (T, 2T+1) so that s halves; then a = 2 R(s/2) - R(s).
    Completely fit-free, so it cannot hide a bad window choice.
    """
    print("\n### fit-free two-point elimination  a = 2 R(s/2) - R(s)")
    print("     T1        T2      R(T1)      R(T2)      a_2")
    Ts = [T.max() / 2 ** k for k in range(6, -1, -1)]
    for T1 in Ts:
        T2 = 2 * T1 + 1.0
        if T2 > T.max():
            continue
        R1 = np.interp(T1, T, R_of_T(T, V))
        R2 = np.interp(T2, T, R_of_T(T, V))
        print(f"{T1:9.3f} {T2:9.3f} {R1:10.6f} {R2:10.6f} {2 * R2 - R1:10.6f}")


def constraint(T, V, label, Xc=None):
    """C = int_0^inf (1-V) dT, which equals lim (T - Xc) and should be 1.

    The constraint is NOT evidence for or against a_2.  Within the paper's own
    expansion, Xc = T - 1 - U_{-2}/T + O(1/T^2): the O(1) term is -1 whatever
    U_{-2} is, so C = 1 holds identically and carries no information about the
    coefficient.  What is reported here is therefore a consistency check on the
    numerics, not a test of a_2.

    The quadrature is stated as three separately labelled pieces -- the head
    T < T[0], the measured record, and the analytic tail -- because only the
    middle one is measured.  The head uses the early-time result (5.10),
    1 - V = 1 - 2 sqrt(T/pi), which is valid only for T << 1 and is reported
    with the T[0] at which it was applied so the reader can judge it.  The tail
    assumes the fitted deficit a/(1+T)^2 continues, with a taken from the data
    rather than hard-coded.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    D = 1.0 - V
    C_rec = trapz(D, T)
    T0, Tend = T[0], T[-1]
    # head: (5.10) integrated, int_0^T0 (1 - 2 sqrt(T/pi)) dT
    head = T0 - (4.0 / 3.0) * T0 ** 1.5 / np.sqrt(np.pi)
    head_valid = T0 < 1e-2
    # tail: a from the record's own plateau, not assumed
    a = float(np.median(R_of_T(T, V)[T > 0.6 * Tend]))
    tail = a / (1 + Tend)
    print(f"\n### constraint C = int_0^inf (1-V) dT = lim (T - Xc), {label}")
    print(f"  head   0 .. {T0:<8.4g} from (5.10) = {head:.6f}"
          f"   [{'valid, T0 << 1' if head_valid else 'NOT VALID: T0 is not << 1'}]")
    print(f"  record {T0:.4g} .. {Tend:<8.4g} measured   = {C_rec:.6f}")
    print(f"  tail   T > {Tend:<8.4g} a={a:.4f}      = {tail:.6f}")
    print(f"  C = {head + C_rec + tail:.6f}   (exact value 1; a_2-independent)")
    if Xc is not None:
        print(f"  direct: T - Xc at T = {Tend:.4g} is {Tend - Xc[-1]:.6f},"
              f"  + tail {tail:.6f}  ->  {Tend - Xc[-1] + tail:.6f}")


def candidate_forms(T, V):
    print("\n### candidate closed forms, max relative error in (1-V)")
    forms = {
        "(7.11)  1/(1+T)^2": lambda t: 1.0 / (1 + t) ** 2,
        "(3/2)/(1+T)^2": lambda t: 1.5 / (1 + t) ** 2,
        "(3/2)/(1+T)^2 - 1/(1+T)^3": lambda t: 1.5 / (1 + t) ** 2 - 1.0 / (1 + t) ** 3,
        "(3/2)/(1+T)^2 - 3/(1+T)^3": lambda t: 1.5 / (1 + t) ** 2 - 3.0 / (1 + t) ** 3,
        "(3/2)/(1+T)^2 - 3/(1+T)^3 + 3/(1+T)^4":
            lambda t: 1.5 / (1 + t) ** 2 - 3.0 / (1 + t) ** 3 + 3.0 / (1 + t) ** 4,
        "(3/2)(1+T)/(1+ (3/2) T)^2 ": lambda t: 1.5 * (1 + t) / (1 + 1.5 * t) ** 2,
    }
    wins = [(5, 50), (10, 50), (20, 50), (50, 1e9), (100, 1e9)]
    names = list(forms)
    print("form".ljust(40) + "".join(f"  {a}-{b if b < 1e8 else 'end':>5}" for a, b in wins))
    D = 1.0 - V
    for n in names:
        row = n.ljust(40)
        for a, b in wins:
            m = (T >= a) & (T <= b)
            if m.sum() == 0:
                row += "      -"
                continue
            e = np.max(np.abs(forms[n](T[m]) - D[m]) / D[m])
            row += f" {100 * e:6.2f}%"
        print(row)
    # integrals of the candidate forms over 0..inf
    print("\n  int_0^inf of each form (must be 1 to give Xc ~ T - 1):")
    print("    1/(1+T)^2                          -> 1")
    print("    (3/2)/(1+T)^2                      -> 3/2")
    print("    (3/2)/(1+T)^2 - 1/(1+T)^3          -> 3/2 - 1/2 = 1")
    print("    (3/2)/(1+T)^2 - 3/(1+T)^3          -> 3/2 - 3/2 = 0")
    print("    (3/2)/(1+T)^2 - 3/(1+T)^3 + 3/(1+T)^4 -> 3/2 - 3/2 + 1 = 1")


def main(paths):
    for p in paths:
        p = Path(p)
        T, V = load_series(p)
        label = p.stem
        window_table(T, V, [1, 2, 5, 10, 15, 20, 25, 30, 40], [0, 1, 2, 3, 4], label)
        local_two_point(T, V)
        d = np.genfromtxt(p, delimiter=",", names=True)
        Xc = np.asarray(d["Xc"], float) if "Xc" in d.dtype.names else None
        constraint(T, V, label, Xc)
        candidate_forms(T, V)
        if Xc is not None:
            # Independent estimate of a_2 from the DISPLACEMENT rather than the
            # speed: with a_3 = 0 the resummed result is Xc = T - 1 + a_2/(1+T).
            print("\n### a_2 from the displacement, (Xc - T + 1)(1 + T)")
            for Tq in (5, 10, 20, 30, 50, 100, 150, 200, 300, 400):
                if Tq > T[-1]:
                    continue
                i = int(np.argmin(np.abs(T - Tq)))
                print(f"   T={T[i]:8.3f}  Xc-T={Xc[i]-T[i]:10.6f}   "
                      f"a_2 = {(Xc[i] - T[i] + 1) * (1 + T[i]):.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
