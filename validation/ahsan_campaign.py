#!/usr/bin/env python3
r"""Validation campaign: TC1D-FEM vs ABDS-CN vs the Ahsan et al. asymptotics.

Two 1D solvers and one asymptotic theory, on ONE continuous problem.

  TC1D-FEM   ours.   pyoomph FEM in a Lagrangian mass coordinate, staggered
                     h=D0 / u=C1, BDF2, with_capillarity=False. Driven here
                     through ``simulationCases/trouton1d_pyoomph.py``.
  ABDS-CN    theirs. Our numpy port (``validation/ahsan_thinfilm.py``) of the
                     Crank-Nicolson MATLAB released with Ahsan, Brandao,
                     Davidovitch & Stone, arXiv:2507.04603. Two independent
                     schemes: ``reduced-inf`` (tip frame, L = infinity) and
                     ``reduced-fin`` (static frame, finite L, their appendix D).
  theory             their eqs (5.10), (6.10), (7.11), (3.10a).

UNIT CONVERSION (re-derived, not assumed; see ahsan_thinfilm.py for the algebra)

    Oh_paper = 2 Oh          their Oh uses the HALF thickness h0 = H_full/2
    T        = t*/4             tI = 4 mu h0/gamma = 4 tau_vis
    V        = u*               their V and our u/V_TC are the same number
    H        = h                their H is h_half/h0, ours is h_full/H_full
    lI       = 4 h0 Oh_paper = 4 Oh     in our code units (H_full = 1)
    L        = l0/lI = M/(4 Oh)         M our Lagrangian mass domain
    X - X_c  = (x - x_tip)/(4 Oh)

Pushed through our momentum equation u_t = (4 mu/h)(h u_x)_x this gives exactly
H V_T + H V V_X = (H V_X)_X, their (2.16), with no leftover constant. That is
the re-verification the brief asks for and it is checked numerically in
``check-units``.

THE FINITE SHEET IS REPRESENTABLE IN TC1D-FEM, and this is the point the brief
flags as uncertain. Their finite-L problem (appendix D) puts a ZERO-SLOPE
thickness condition and V = 0 at X = L: that is a symmetry plane, i.e. half of
a sheet of length 2 l0 retracting from both ends. Our staggered scheme applies
DirichletBC(u=0) at m = M with h cell-constant and therefore Neumann -- the
same symmetry plane. So a TC1D-FEM run with mass M and the disturbance ALLOWED
to reach m = M is a finite sheet of L = M/(4 Oh). The co-located form pins
h = 1 there as well and is NOT a finite sheet; it is not used here.

AN EXACT INVARIANT, USED AS A DIAGNOSTIC. Their (3.1) H V + H_X = 0 is an exact
invariant of (2.15)-(2.16) starting from H = 1, V = 0 -- differentiating H V +
H_X in time and substituting gives identically zero, inertia included. In our
mass coordinate H_X = lI H H_m, so the invariant reads

    V = -lI * dh/dm        (lI = 4 Oh)

which involves no reconstruction of the physical coordinate at all. TC1D-FEM
never uses this relation, so its residual is an independent correctness check.

USAGE
    ahsan_campaign.py tcfem --oh 1 --L 20 --Tend 60 --nsteps 60000 --out DIR
    ahsan_campaign.py tcfem --oh 1 --semi-infinite --Tend 50 ... --out DIR
    ahsan_campaign.py abds --mode fin --L 20 --Tend 60 --N 4000 --NT 400000
    ahsan_campaign.py abds --mode inf --Tend 50 --Lhat 40 --dx 0.02 --dt 1e-4
    ahsan_campaign.py check-units
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "simulationCases"))
sys.path.insert(0, str(REPO / "validation"))

V_TC = math.sqrt(2.0)
# Tip cell in mass units validated by the mesh study at M=120, N=400, grade=2.
TIP_CELL = 120.0 / 400.0 ** 2


# ---------------------------------------------------------------------------
# theory
# ---------------------------------------------------------------------------

def U_early(T):
    """(5.10)  dXc/dT ~ 2 sqrt(T/pi),  T << 1."""
    return 2.0 * np.sqrt(np.asarray(T, dtype=float) / math.pi)


def U_late_inf(T):
    """(7.11)  dXc/dT ~ 1 - 1/(1+T)^2,  L = infinity, T >> 1.

    NOTE the deficit is 1/(1+T)^2, not a pure power T^-2. Everything downstream
    compares 1 - U against 1/(1+T)^2 directly.
    """
    return 1.0 - 1.0 / (1.0 + np.asarray(T, dtype=float)) ** 2


def U_late_fin(T, L):
    """(6.10)  dXc/dT ~ L/(1+T)^2,  finite L, L - Xc << 1.

    This is mass conservation plus a uniform thickness: the half-sheet carries
    mass L, its thickness is 1+T by (3.10a), so its length is L - Xc = L/(1+T)
    and dXc/dT = L/(1+T)^2. Stated here so the comparison is against an
    identity of the model rather than against a fitted curve.
    """
    return L / (1.0 + np.asarray(T, dtype=float)) ** 2


def LmXc_late_fin(T, L):
    """L - Xc = L/(1+T), the integral of (6.10) with c = 0 from (3.14)."""
    return L / (1.0 + np.asarray(T, dtype=float))


def Xc_late_inf(T):
    """(7.17) AS PRINTED, Xc ~ T + 1. THE SIGN IS WRONG IN THE PAPER.

    Kept as printed so the ``Xc_eq7p17`` column is what the paper says and the
    discrepancy is visible in the data rather than silently corrected. Their own
    (7.8a-b) with (7.10) give H - 1 = T e^{-Xhat} - Xhat e^{-Xhat}, so (7.16)
    Xc = int_0^inf (H-1) dXhat = T - 1. Their (7.13) gives T^2/(1+T) =
    T - 1 + 1/(1+T). Integrating their own (7.11) gives T^2/(1+T) again. All
    three routes give T - 1, and both codes here measure Xc(50) - 50 = -0.971.
    Nothing downstream depends on it: Sec. VIII uses only Xc ~ T.
    """
    return np.asarray(T, dtype=float) + 1.0


def H_prof_7p13(Xhat, T):
    """(7.13)  H ~ 1 + T exp(-((1+T)/T) Xhat),  Xhat = X - Xc, uniformly
    valid to Xhat = O(T). Equals 1+T at the tip, -> 1 far field."""
    return 1.0 + T * np.exp(-(1.0 + T) / T * np.asarray(Xhat, dtype=float))


def V_prof_7p14(Xhat, T):
    """(7.14)  V ~ (T+1) e^{-a Xhat} / (1 + T e^{-a Xhat}), a = (1+T)/T.
    This is (3.1) applied to (7.13). Equals 1 at the tip, -> 0 far field."""
    e = np.exp(-(1.0 + T) / T * np.asarray(Xhat, dtype=float))
    return (T + 1.0) * e / (1.0 + T * e)


def H_prof_6p3(X, T, L, Xc):
    """(6.3)+(6.6)  H ~ 1 + T + (1/2)(X + Xc - 2L)(X - Xc),  L - Xc << 1."""
    X = np.asarray(X, dtype=float)
    return 1.0 + T + 0.5 * (X + Xc - 2.0 * L) * (X - Xc)


def V_prof_6p7(X, T, L):
    """(6.7)  V ~ (L - X)/(1 + T),  L - Xc << 1. Linear in X, zero at the wall."""
    return (L - np.asarray(X, dtype=float)) / (1.0 + T)


# ---------------------------------------------------------------------------
# fig. 5 diagnostics -- THEIR definition, read off their released MATLAB
# ---------------------------------------------------------------------------
# thin_film_Oh_large.m, run_code_array:
#     a = D(v(N,:)', t(2)-t(1));            % dU/dT on the UNIFORM frame grid
#     a = -a.*(a<0);                        % keep deceleration only
#     [amax,itrans] = max(a);
#     transition(i) = t(itrans);            % -> T_c
#     halfMax  = max(amax)/2;
#     index1 = find(a >= halfMax, 1, 'first');
#     index2 = find(a >= halfMax, 1, 'last');
#     duration(i) = 2*(t(index2) - t(index1));   % -> Delta T_c
# so T_c is the time of PEAK DECELERATION and Delta T_c is TWICE the full width
# at half maximum of that deceleration peak. Both are applied verbatim to both
# codes below. The `find(...,'first'/'last')` scan is GLOBAL, so any unrelated
# deceleration episode anywhere in the record widens Delta T_c; the contiguous
# variant is reported alongside as `dTc_local`.

def _D_onesided(f, dx):
    """thin_film_Oh_large.m ``D(f,dx)``, the derivative their fig. 5 uses."""
    f = np.asarray(f, dtype=float)
    N = f.size
    y = np.zeros(N)
    y[1:N - 1] = (f[2:N] - f[0:N - 2]) / 2.0 / dx
    y[0] = 2.0 * (f[1] - f[0]) / dx - y[1]
    y[N - 1] = 2.0 * (f[N - 1] - f[N - 2]) / dx - y[N - 2]
    return y


def tc_dtc(T, U) -> dict:
    """T_c and Delta T_c, by their definition, applied identically to both codes.

    THE PAPER AND THE CODE DISAGREE BY A FACTOR OF TWO, and this matters for
    reading fig. 5(b). Sec. VIII says Delta T_c "is estimated as the full width
    at half maximum (FWHM) of this minimum"; the released
    ``thin_film_Oh_large.m`` computes ``duration = 2*(t(index2)-t(index1))``,
    which is TWICE the FWHM. Fig. 5(b)'s ordinate runs 2.4 to 3.1, so the
    plotted quantity is the code's ``duration``, i.e. 2 x FWHM. Both are
    returned; ``dTc_code`` is the one that reproduces their figure.

    T must be UNIFORMLY spaced -- their D() assumes it and so does this.
    """
    T = np.asarray(T, dtype=float)
    U = np.asarray(U, dtype=float)
    dT = T[1] - T[0]
    a = _D_onesided(U, dT)
    a = -a * (a < 0)                      # deceleration only, their line
    imax = int(np.argmax(a))
    amax = float(a[imax])
    if amax <= 0.0:
        return {"Tc": float("nan"), "dTc_code": float("nan"),
                "fwhm": float("nan"), "dTc_local": float("nan"),
                "peak_decel": 0.0, "Umax": float(np.max(U)),
                "T_Umax": float(T[int(np.argmax(U))])}
    half = amax / 2.0
    idx = np.flatnonzero(a >= half)       # GLOBAL scan, exactly as they do
    fwhm = float(T[idx[-1]] - T[idx[0]])
    lo = hi = imax                        # contiguous run through the peak
    while lo > 0 and a[lo - 1] >= half:
        lo -= 1
    while hi < a.size - 1 and a[hi + 1] >= half:
        hi += 1
    # Sub-grid variants. The definition above reads Tc and the half-maximum
    # crossings off grid points, so dTc_code is quantised in steps of 2*dT and
    # Tc in steps of dT. With a fixed sample count over a window that grows
    # with L, that quantum grows with L too and shows as a sawtooth in
    # dTc(L) -- every returned value is an exact multiple of 2*dT. These
    # interpolated versions remove it; they are reported ALONGSIDE the verbatim
    # ones, never in place of them, because the comparison against the
    # published figure has to use the published definition.
    def _cross(i_in, i_out):
        """Linear interpolation of the T at which a crosses `half`."""
        a_in, a_out = a[i_in], a[i_out]
        if a_in == a_out:
            return float(T[i_in])
        return float(T[i_in] + (T[i_out] - T[i_in]) * (a_in - half) / (a_in - a_out))

    t_lo = _cross(idx[0], idx[0] - 1) if idx[0] > 0 else float(T[idx[0]])
    t_hi = _cross(idx[-1], idx[-1] + 1) if idx[-1] < a.size - 1 else float(T[idx[-1]])
    # Parabola through the three highest samples for the peak location.
    if 0 < imax < a.size - 1:
        y0, y1, y2 = a[imax - 1], a[imax], a[imax + 1]
        den = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
        tc_interp = float(T[imax] + shift * dT)
    else:
        tc_interp = float(T[imax])

    return {"Tc": float(T[imax]), "dTc_code": 2.0 * fwhm, "fwhm": fwhm,
            "dTc_local": 2.0 * float(T[hi] - T[lo]), "peak_decel": amax,
            "Tc_interp": tc_interp, "dTc_interp": 2.0 * (t_hi - t_lo),
            "Umax": float(np.max(U)), "T_Umax": float(T[int(np.argmax(U))])}


# ---------------------------------------------------------------------------
# TC1D-FEM
# ---------------------------------------------------------------------------

def auto_nelem(mass: float) -> int:
    return int(min(4000, max(400, round(math.sqrt(mass / TIP_CELL)))))


def read_state(p, lI: float) -> dict:
    """Pull (m, h, u) off the mesh and build the paper's (Xhat, H, V).

    ``to_numpy(tesselate_tri, nondimensional, history_index, discontinuous)``
    with discontinuous=True duplicates the nodes per element and returns the
    cell-constant D0 thickness in ``D0_data``. That is the only way to see a D0
    field: it has no nodal value, which is why TextFileOutput writes u alone in
    the staggered scheme.
    """
    msh = p.get_mesh("sheet")
    nodal, einds, _etypes, nfi, D0, _DL, efi = msh.to_numpy(False, True, 0, True)
    m_all = nodal[:, nfi["coordinate_x"]]
    u_all = nodal[:, nfi["u"]]
    h_all = D0[:, efi["h"]]

    ia, ib = einds[:, 0], einds[:, 1]
    ma, mb = m_all[ia], m_all[ib]
    swap = ma > mb
    ia2 = np.where(swap, ib, ia)
    ib2 = np.where(swap, ia, ib)
    m0, m1 = m_all[ia2], m_all[ib2]
    order = np.argsort(m0)
    ia2, ib2, m0, m1 = ia2[order], ib2[order], m0[order], m1[order]

    h = h_all[ia2]                        # cell-constant, same on both copies
    dm = m1 - m0
    dx = dm / h                           # physical width of each cell
    x_node = np.concatenate([[0.0], np.cumsum(dx)])   # distance from the tip
    u_node = np.concatenate([u_all[ia2], [u_all[ib2[-1]]]])
    m_node = np.concatenate([m0, [m1[-1]]])
    m_cell = 0.5 * (m0 + m1)
    x_cell = 0.5 * (x_node[:-1] + x_node[1:])

    # exact invariant (3.1) in the mass coordinate: V = -lI dh/dm, evaluated at
    # the interior nodes where u lives and where the cell-constant h has a
    # well-defined jump.
    dh_dm = (h[1:] - h[:-1]) / (m_cell[1:] - m_cell[:-1])
    V_int = u_node[1:-1] / V_TC
    resid = V_int + lI * dh_dm

    return {
        "Xhat_node": x_node / lI, "V_node": u_node / V_TC, "m_node": m_node,
        "Xhat_cell": x_cell / lI, "H_cell": h, "m_cell": m_cell,
        "length": x_node[-1] / lI,          # current sheet length, in X units
        "H_max": float(h.max()),
        "V_tip": float(u_node[0] / V_TC),
        "inv_max": float(np.max(np.abs(resid))),
        "inv_rms": float(math.sqrt(float(np.mean(resid ** 2)))),
        "V_far": float(u_node[-1] / V_TC),
        "H_far": float(h[-1]),
    }


def run_tcfem(oh, Tend, nsteps, n_outputs, spacing, out, tag,
              L=None, semi_infinite=False, nelem=None, mass=None,
              profile_T=(), t0_frac=1e-4, max_newton=None, gcn=False):
    from trouton1d_pyoomph import Trouton1DProblem

    # ``--out`` MUST BE UNIQUE PER CASE, and this is not a tidiness preference.
    # pyoomph generates, compiles and dlopens a shared object per equation into
    # ``<--out>/_ccode``, regardless of what set_output_directory is given.
    # Eleven concurrent cases pointed at one directory therefore compile over
    # each other's objects and segfault at load: the first launch of this
    # campaign lost all six concurrent TC1D-FEM cases inside four seconds to
    # SIGSEGV, while every pure-numpy ABDS-CN case in the same batch survived.
    # The job list gives each case its own ``--out``; the campaign runner
    # promotes the CSVs up to the section directory afterwards and deletes the
    # scratch, which is reproducible from them and is otherwise ~1e4 files.
    # .resolve() BEFORE the chdir: every later `out / ...` write would
    # otherwise be re-resolved against the NEW working directory and duplicate
    # the path segment, which fails at the very end of a long run when the
    # results are written. Absolute --out hides this; a relative one does not.
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)

    lI = 4.0 * oh          # = 4 h0 Oh_paper in code units (H_full = 1)
    t0 = time.time()
    with Trouton1DProblem() as p:
        p.Oh = oh
        tstar_end = 4.0 * Tend
        t_end = tstar_end * p.tau_vis
        if mass is not None:
            M = mass
        elif semi_infinite:
            M = max(120.0, 2.5 * V_TC * t_end)
        elif L is None:
            # Fail here rather than three lines later with a TypeError on
            # None * float, which is raised only after pyoomph has compiled.
            raise ValueError("tcfem needs exactly one of --L, --mass or "
                             "--semi-infinite to fix the sheet length")
        else:
            M = L * lI
        p.mass = M
        p.nelem = nelem if nelem is not None else auto_nelem(M)
        L_eff = M / lI
        p.set_output_directory(str(out))
        p.initialise()
        p.write_states = False

        dt = t_end / nsteps
        if spacing == "geom":
            Tg = np.geomspace(Tend * t0_frac, Tend, n_outputs)
        else:
            Tg = np.linspace(Tend / n_outputs, Tend, n_outputs)
        # SNAP the profile times onto the existing grid rather than inserting
        # them. Inserting breaks the uniform spacing that their fig. 5
        # estimator requires (it differentiates on the frame index), and
        # np.unique does not remove float-noise duplicates, which silently
        # shifted every later output by one interval the first time this ran.
        want = set()
        for v in np.asarray(profile_T, dtype=float):
            if 0 < v <= Tend * (1 + 1e-12):
                want.add(float(Tg[int(np.argmin(np.abs(Tg - v)))]))
        Tg = Tg[(Tg > 0) & (Tg <= Tend * (1 + 1e-12))]

        meta = {
            "code": "TC1D-FEM", "tag": tag, "Oh": oh,
            "Oh_paper": 2.0 * oh, "mu": p.mu, "tau_vis": p.tau_vis,
            "lI": lI, "mass": M, "L": L_eff,
            "semi_infinite": bool(semi_infinite),
            "nelem": p.nelem, "grade": p.grade, "tip_cell_mass": M / p.nelem ** 2,
            "Tend": Tend, "tstar_end": tstar_end, "t_end": t_end,
            "nsteps": nsteps, "dt_code": dt, "dtstar": dt / p.tau_vis,
            "dT": dt / p.tau_vis / 4.0, "n_outputs": len(Tg),
            "spacing": spacing, "scheme": p.default_timestepping_scheme,
            "staggered": p.staggered, "variable": p.variable,
            "with_capillarity": p.with_capillarity, "V_TC": V_TC,
        }
        print(f"[{tag}] Oh={oh:g} L={L_eff:.6g} M={M:.6g} N={p.nelem} "
              f"Tend={Tend:g} dT={meta['dT']:.3g} nout={len(Tg)}", flush=True)

        rows = []
        prev = 0.0
        for Tt in Tg:
            t = Tt * 4.0 * p.tau_vis
            start = t_end * 1e-6 if prev == 0.0 else min(dt, t - prev)
            extra = {}
            if max_newton is not None:
                extra["max_newton_iterations"] = max_newton
            if gcn:
                extra["globally_convergent_newton"] = True
            p.run(t, startstep=start, maxstep=dt, temporal_error=1e-3,
                  outstep=False, out_initially=False, **extra)
            now = float(p.get_current_time(as_float=True))
            tstar = now / p.tau_vis
            T = tstar / 4.0
            s = read_state(p, lI)
            Xc = L_eff - s["length"]
            rows.append([T, tstar, s["V_tip"], Xc, s["length"], s["H_max"],
                         1.0 + T, s["inv_max"], s["inv_rms"],
                         abs(s["V_far"]), abs(s["H_far"] - 1.0),
                         U_early(T), U_late_inf(T), U_late_fin(T, L_eff),
                         LmXc_late_fin(T, L_eff), Xc_late_inf(T)])
            for v in list(want):
                if abs(Tt - v) <= 1e-9 * max(1.0, abs(v)):
                    _write_profiles(out, tag, T, s, L_eff, Xc)
                    want.discard(v)
            prev = t

        tab = np.array(rows)
        hdr = ("T,tstar,V,Xc,L_minus_Xc,H_max,H_tip_exact,"
               "invariant_max,invariant_rms,drift_V_far,drift_H_far,"
               "U_eq5p10,U_eq7p11,U_eq6p10,L_minus_Xc_eq6p10,Xc_eq7p17")
        np.savetxt(out / f"{tag}_series.csv", tab, delimiter=",",
                   header=hdr, comments="")
        meta["wall_s"] = time.time() - t0
        meta["V_final"] = float(tab[-1, 2])
        meta["worst_drift_V_far"] = float(np.max(tab[:, 9]))
        meta["worst_drift_H_far"] = float(np.max(tab[:, 10]))
        meta["worst_invariant"] = float(np.max(tab[:, 7]))
        (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{tag}] done in {meta['wall_s']:.0f}s  V(Tend)={meta['V_final']:.6f} "
              f"invariant<={meta['worst_invariant']:.2e} "
              f"drift_H_far<={meta['worst_drift_H_far']:.2e}", flush=True)
    return meta


def _write_profiles(out: Path, tag: str, T: float, s: dict, L: float,
                    Xc: float) -> None:
    """Two files per (case, T): V lives on nodes, H on cells. They are NOT on
    the same grid and interpolating one onto the other would invent structure
    the discretisation does not have, so they are kept apart.

    Both carry the paper's asymptotic profiles evaluated on the SAME abscissa,
    so a comparison needs no re-derivation downstream:
      (7.13)/(7.14)  the L = infinity, T >> 1 tip-frame profiles;
      (6.3)/(6.7)    the L - Xc << 1 finite-sheet profiles.
    X = Xc + Xhat is the lab-frame position, the abscissa of their fig. 2.
    """
    base = out / f"{tag}_profile_T{T:.6g}"
    xh = s["Xhat_node"]
    X = Xc + xh
    np.savetxt(f"{base}_V.csv",
               np.column_stack([xh, X, s["V_node"], s["m_node"],
                                V_prof_7p14(xh, T), V_prof_6p7(X, T, L)]),
               delimiter=",", header="Xhat,X,V,m_code,V_eq7p14,V_eq6p7",
               comments="")
    xhc = s["Xhat_cell"]
    Xc_c = Xc + xhc
    np.savetxt(f"{base}_H.csv",
               np.column_stack([xhc, Xc_c, s["H_cell"], s["m_cell"],
                                H_prof_7p13(xhc, T),
                                H_prof_6p3(Xc_c, T, L, Xc)]),
               delimiter=",", header="Xhat,X,H,m_code,H_eq7p13,H_eq6p3",
               comments="")


# ---------------------------------------------------------------------------
# ABDS-CN
# ---------------------------------------------------------------------------

def run_abds(mode, Tend, out, tag, L=None, N=None, NT=None, nframe=4000,
             dx=None, dt=None, Lhat=None, profile_T=()):
    import ahsan_thinfilm as A
    A.use_fast_solver(True)

    # The two subcommands take disjoint parameter sets, and argparse cannot
    # express that. Name the missing option instead of letting it reach numpy
    # as None (inf) or the solver as None (fin), where the message is useless.
    need = {"inf": ("dx", "dt", "Lhat"), "fin": ("L", "N", "NT")}[mode]
    got = {"dx": dx, "dt": dt, "Lhat": Lhat, "L": L, "N": N, "NT": NT}
    missing = [k for k in need if got[k] is None]
    if missing:
        raise ValueError(f"abds --mode {mode} requires "
                         + ", ".join("--" + k for k in missing))

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if mode == "inf":
        r = A.simulate_reduced_infinite(T=Tend, dt=dt, dx=dx, L=Lhat,
                                        Nframe=nframe, verbose=True)
        T = r["T"]
        U = r["U"]
        Xhat = r["Xhat"]
        H = r["H"]
        # Xc is not returned; U is on a UNIFORM T grid so cumulative trapezoid
        # is exact to O(dT^2) and there is nothing better available.
        Xc = np.concatenate([[0.0], np.cumsum(0.5 * (U[1:] + U[:-1])
                                              * np.diff(T))])
        LmXc = np.full_like(T, np.nan)
        Lrep = float("inf")
        meta = {"mode": "reduced-inf", "dx": dx, "dt": dt, "Lhat": Lhat,
                "N": int(round(Lhat / dx)), "NT": int(round(Tend / dt))}
    else:
        r = A.simulate_reduced_finite(L=L, N=N, T=Tend, NT=NT, Nframe=nframe,
                                      verbose=True)
        T = r["T"]
        U = r["U"]
        Xc = r["Xc"]
        LmXc = np.sqrt(r["S"])
        H = r["H"]
        Xhat = None                      # X-dependent grid, handled per frame
        Lrep = L
        meta = {"mode": "reduced-fin", "L": L, "N": N, "NT": NT,
                "dX_wall": L / N, "dT": Tend / NT}

    Hmax = H.max(axis=0)
    Htip = H[-1, :] if mode == "fin" else H[0, :]
    cols = [T, 4.0 * T, U, Xc, LmXc, Hmax, Htip, 1.0 + T,
            U_early(T), U_late_inf(T), Xc_late_inf(T)]
    hdr = ["T", "tstar", "V", "Xc", "L_minus_Xc", "H_max", "H_tip",
           "H_tip_exact", "U_eq5p10", "U_eq7p11", "Xc_eq7p17"]
    if mode == "fin":
        cols += [U_late_fin(T, L), LmXc_late_fin(T, L)]
        hdr += ["U_eq6p10", "L_minus_Xc_eq6p10"]
    np.savetxt(out / f"{tag}_series.csv", np.column_stack(cols), delimiter=",",
               header=",".join(hdr), comments="")

    for v in profile_T:
        j = int(np.argmin(np.abs(T - v)))
        Tj = float(T[j])
        Hj = H[:, j]
        if mode == "inf":
            xh = Xhat
            # V = -(ln H)_X, their (3.1). In the tip frame d/dX = d/dXhat.
            Vj = -A._D_uniform(np.log(Hj), xh[1] - xh[0], "interior")
            Xj = Xc[j] + xh
            Lj = np.inf
        else:
            xw = r["xw"][:, j]                      # L - X, wall at 0
            # their v_calculate: V = d(ln H)/d(L-X) = -(ln H)_X, i.e. (3.1)
            Vj = A._D_uniform(np.log(Hj), xw[1] - xw[0], "onesided")
            xh = xw[-1] - xw                        # X - Xc, tip at 0
            o = np.argsort(xh)
            xh, Hj, Vj = xh[o], Hj[o], Vj[o]
            Xj = Xc[j] + xh
            Lj = L
        cc = [xh, Xj, Hj, Vj, H_prof_7p13(xh, Tj), V_prof_7p14(xh, Tj)]
        hh = ["Xhat", "X", "H", "V", "H_eq7p13", "V_eq7p14"]
        if np.isfinite(Lj):
            cc += [H_prof_6p3(Xj, Tj, Lj, Xc[j]), V_prof_6p7(Xj, Tj, Lj)]
            hh += ["H_eq6p3", "V_eq6p7"]
        np.savetxt(out / f"{tag}_profile_T{Tj:.6g}.csv", np.column_stack(cc),
                   delimiter=",", header=",".join(hh), comments="")

    meta.update({"code": "ABDS-CN", "tag": tag, "Tend": Tend,
                 "L_reported": Lrep, "nframe": len(T),
                 "wall_s": time.time() - t0, "V_final": float(U[-1])})
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{tag}] done in {meta['wall_s']:.0f}s V(Tend)={U[-1]:.6f}",
          flush=True)
    return meta


# ---------------------------------------------------------------------------
# fig. 5 assembly
# ---------------------------------------------------------------------------

def build_fig5(series, out_csv, code):
    """Apply their T_c / Delta T_c estimator to a list of (L, series.csv).

    The series MUST be on a uniform T grid: their estimator differentiates on
    the frame index. The grid spacing is written to the output so the
    sensitivity of Delta T_c to sampling is visible rather than hidden.
    """
    rows = []
    for L, path in series:
        a = np.genfromtxt(path, delimiter=",", names=True)
        T, U = np.asarray(a["T"], float), np.asarray(a["V"], float)
        d = np.diff(T)
        skew = float(np.max(np.abs(d - d[0])) / abs(d[0]))
        if skew > 1e-3:
            # The adaptive timestepper lands a hair short of each requested
            # output time, so the recorded grid is uniform only to round-off.
            # Beyond 0.1% resample rather than pretend, and say so.
            print(f"    (resampled {path.name}: T grid non-uniform by "
                  f"{skew:.2e} of dT)")
            Tu = np.linspace(T[0], T[-1], T.size)
            U = np.interp(Tu, T, U)
            T = Tu
        r = tc_dtc(T, U)
        rows.append([L, r["Tc"], r["dTc_code"], r["fwhm"], r["dTc_local"],
                     r["peak_decel"], r["Umax"], r["T_Umax"], d[0],
                     r["Tc"] / L])
        print(f"  {code} L={L:<6g} Tc={r['Tc']:.4g} (Tc/L={r['Tc']/L:.4f}) "
              f"dTc_code={r['dTc_code']:.4g} FWHM={r['fwhm']:.4g} "
              f"Umax={r['Umax']:.5f}", flush=True)
    tab = np.array(rows)
    tab = tab[np.argsort(tab[:, 0])]
    np.savetxt(out_csv, tab, delimiter=",",
               header="L,Tc,dTc_code,fwhm,dTc_local,peak_decel,Umax,T_Umax,"
                      "dT_sample,Tc_over_L", comments="")
    print(f"wrote {out_csv}")
    return tab


# ---------------------------------------------------------------------------
# unit-conversion re-verification
# ---------------------------------------------------------------------------

def check_units() -> int:
    """Re-derive the conversion symbolically and confirm it numerically.

    Written out so the campaign does not inherit it on trust.
    """
    import sympy as sp
    gamma, rho, Hf, X, T = sp.symbols("gamma rho H_full X T", positive=True)
    oh = sp.Symbol("Oh", positive=True)
    # our convention: Oh is fixed by mu = sqrt(2) Oh at rho=gamma=Hf=1,
    # i.e. Oh = mu/sqrt(2 rho gamma H_full).
    mu = oh * sp.sqrt(2 * rho * gamma * Hf)
    h0 = Hf / 2                                  # their half thickness
    Oh_paper = mu / sp.sqrt(rho * h0 * gamma)
    lI = 4 * h0 * Oh_paper
    tI = 4 * mu * h0 / gamma
    tau_vis = mu * Hf / (2 * gamma)
    uTC = sp.sqrt(gamma / (rho * h0))
    V_TC_s = sp.sqrt(2 * gamma / (rho * Hf))
    checks = [
        ("Oh_paper / Oh", sp.simplify(Oh_paper / oh), 2),
        ("tI / tau_vis", sp.simplify(tI / tau_vis), 4),
        ("uTC / V_TC", sp.simplify(uTC / V_TC_s), 1),
        ("lI / (4 Oh)  at rho=gamma=H_full=1",
         sp.simplify((lI / (4 * oh)).subs({rho: 1, gamma: 1, Hf: 1})), 1),
    ]
    ok = True
    for name, got, wants in checks:
        good = sp.simplify(got - wants) == 0
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name} = {got}  (want {wants})")

    # push u_t = (4 mu/h)(h u_x)_x through the substitution and read off (2.16)
    V = sp.Function("V")
    Hh = sp.Function("H")
    sub = {rho: 1, gamma: 1, Hf: 1}
    lIs, tIs = sp.simplify(lI.subs(sub)), sp.simplify(tI.subs(sub))
    mus = sp.simplify(mu.subs(sub))
    # u = V_TC V and h = H, with x = lI X and t = tI T, so d/dx = (1/lI) d/dX
    # and d/dt = (1/tI) d/dT -- the chain-rule factors are written out inline
    # below rather than carried as unused symbols.
    u = sp.sqrt(2) * V(X, T)
    h = Hh(X, T)
    lhs = sp.diff(u, T) / tIs + u * sp.diff(u, X) / lIs
    rhs = 4 * mus / h * sp.diff(h * sp.diff(u, X) / lIs, X) / lIs
    expr = sp.simplify(sp.expand((lhs - rhs) * h * tIs / sp.sqrt(2)))
    target = sp.expand(h * sp.diff(V(X, T), T) + h * V(X, T) * sp.diff(V(X, T), X)
                       - sp.diff(h * sp.diff(V(X, T), X), X))
    good = sp.simplify(expr - target) == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] our u_t = (4 mu/h)(h u_x)_x maps to "
          f"their (2.16) H V_T + H V V_X = (H V_X)_X; residual "
          f"{sp.simplify(expr - target)}")

    # (3.1) H V + H_X = 0 is an EXACT invariant of (2.15)-(2.16), inertia and
    # all, and that is what lets them replace the coupled system by the single
    # heat equation (3.8). It is also the diagnostic ``read_state`` applies to
    # TC1D-FEM, so it is worth confirming rather than quoting.
    #
    # Substituting W = 0 into the time derivative of W and eliminating H_T and
    # V_T with the PDEs is the natural route, but it leaves MIXED derivatives
    # (H_XT from differentiating H_X) that a Derivative(H, T) substitution does
    # not reach, so the result depends on sympy's canonicalisation order. Do it
    # constructively instead: W = 0 plus (2.15) IS the heat equation (3.8), so
    # take an arbitrary two-mode solution of the heat equation -- H = 1 +
    # sum a_i exp(-k_i X + k_i^2 T), a solution for every a_i, k_i -- set
    # V = -H_X/H, and check that the MOMENTUM equation (2.16) is then satisfied
    # identically in a_i, k_i, X, T. Nothing is substituted and nothing is
    # simplified away; if the invariant were not exact, the residual would
    # carry the a_i and k_i and could not vanish.
    a1, a2, k1, k2 = sp.symbols("a1 a2 k1 k2")
    Hs = (1 + a1 * sp.exp(-k1 * X + k1 ** 2 * T)
            + a2 * sp.exp(-k2 * X + k2 ** 2 * T))
    good = sp.simplify(sp.diff(Hs, T) - sp.diff(Hs, X, 2)) == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] test field satisfies the heat "
          f"equation (3.8) H_T = H_XX")
    Vs = -sp.diff(Hs, X) / Hs                                    # (3.1)
    mom = (Hs * (sp.diff(Vs, T) + Vs * sp.diff(Vs, X))
           - sp.diff(Hs * sp.diff(Vs, X), X))                    # (2.16)
    res = sp.simplify(sp.together(sp.expand(sp.simplify(mom))))
    good = res == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] (3.1) H V + H_X = 0 is an exact "
          f"invariant: (2.16) residual under V = -H_X/H and H_T = H_XX is "
          f"{res}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tcfem")
    p.add_argument("--oh", type=float, default=1.0)
    p.add_argument("--L", type=float, default=None)
    p.add_argument("--semi-infinite", action="store_true")
    p.add_argument("--mass", type=float, default=None)
    p.add_argument("--Tend", type=float, required=True)
    p.add_argument("--nsteps", type=int, required=True)
    p.add_argument("--n-outputs", type=int, default=200)
    p.add_argument("--spacing", choices=("geom", "uniform"), default="geom")
    p.add_argument("--t0-frac", type=float, default=1e-4)
    p.add_argument("--nelem", type=int, default=None)
    p.add_argument("--profile-T", type=str, default="")
    p.add_argument("--max-newton", type=int, default=None)
    p.add_argument("--gcn", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)

    p = sub.add_parser("abds")
    p.add_argument("--mode", choices=("inf", "fin"), required=True)
    p.add_argument("--L", type=float, default=None)
    p.add_argument("--Tend", type=float, required=True)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--NT", type=int, default=None)
    p.add_argument("--nframe", type=int, default=4000)
    p.add_argument("--dx", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--Lhat", type=float, default=None)
    p.add_argument("--profile-T", type=str, default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)

    sub.add_parser("check-units")

    p = sub.add_parser("fig5")
    p.add_argument("--series", action="append", required=True,
                   help="L:path/to/series.csv, repeatable")
    p.add_argument("--code", required=True)
    p.add_argument("--out", type=Path, required=True)

    a = ap.parse_args()
    if a.cmd == "check-units":
        sys.exit(check_units())
    if a.cmd == "fig5":
        pairs = []
        for s in a.series:
            k, v = s.split(":", 1)
            pairs.append((float(k), Path(v)))
        build_fig5(pairs, a.out, a.code)
        return

    prof = [float(q) for q in a.profile_T.split(",") if q.strip()]
    if a.cmd == "tcfem":
        # ONE CASE PER DIRECTORY: pyoomph chdir's and writes its scratch there.
        run_tcfem(a.oh, a.Tend, a.nsteps, a.n_outputs, a.spacing,
                  a.out, a.tag, L=a.L,
                  semi_infinite=a.semi_infinite, nelem=a.nelem, mass=a.mass,
                  profile_T=prof, t0_frac=a.t0_frac,
                  max_newton=a.max_newton, gcn=a.gcn)
    else:
        run_abds(a.mode, a.Tend, a.out, a.tag, L=a.L, N=a.N, NT=a.NT,
                 nframe=a.nframe, dx=a.dx, dt=a.dt, Lhat=a.Lhat,
                 profile_T=prof)


if __name__ == "__main__":
    main()
