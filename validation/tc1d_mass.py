#!/usr/bin/env python3
r"""TC1D-MASS: a third, independently formulated solver for the semi-infinite
Ahsan et al. retraction problem, written to try to break the measured
U_{-2} = -3/2.

WHY A THIRD CODE.  TC1D-FEM integrates the full momentum equation (2.16) with
inertia in a Lagrangian mass coordinate; ABDS-CN integrates the reduced heat
equation (3.8) on the front-fixed grid of the paper's appendix D.  They already
differ in model and in scheme, but both reach the free-edge speed through a
one-sided gradient at a moving boundary.  This solver removes the free boundary
altogether.

FORMULATION.  Let m be the mass coordinate measured from the free edge,
dm = H dX, so m is material (the edge is a material point).  The exact
invariant (3.1), V = -(1/H) dH/dX, becomes

    V = -dH/dm                                                          (a)

because d/dX = H d/dm.  Lagrangian continuity, d(1/H)/dT|_m = dV/dm, then gives

    H_T = H^2 H_mm         on 0 < m < M,                                (b)

a nonlinear diffusion equation on a FIXED domain.  The edge stress condition
(2.18a), H V_X = -1, is H^2 H_mm = 1 at m = 0, i.e. H_T(0,T) = 1, i.e.

    H(0,T) = 1 + T          -- exactly (3.10a), now an ordinary Dirichlet BC.

Semi-infinite far field: H(M,T) = 1 with M chosen so that the front, which sits
at m ~ T, is far from M.  H - 1 decays like exp(-(m - T + 1)) there, so
M = T_end + 40 leaves a far-field residual ~ 1e-17.  Initial data H(m,0) = 1.

The retraction speed is the edge value of (a),

    U(T) = dX_c/dT = -H_m(0,T),

so the quantity under test, 1 - U = 1 + H_m(0,T), is a one-sided slope of a
Dirichlet-controlled parabolic problem rather than a Stefan flux.  Nothing in
this code shares a discretisation, a coordinate, a time integrator or a
boundary treatment with either of the other two solvers.

DISCRETISATION.  Second-order central differences in m on a uniform grid,
BDF2 in T with a first-order start, Newton with an analytic tridiagonal
Jacobian solved by scipy.linalg.solve_banded.  The edge slope is taken with a
fourth-order one-sided stencil, and cross-checked against the exact curvature
H_mm(0,T) = 1/(1+T)^2 implied by the boundary condition.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


def edge_slope(H, dm):
    """Fourth-order one-sided first derivative at m = 0."""
    return (-25 * H[0] + 48 * H[1] - 36 * H[2] + 16 * H[3] - 3 * H[4]) / (12 * dm)


def edge_curvature(H, dm):
    """Fourth-order one-sided second derivative at m = 0, for the BC check."""
    return (45 * H[0] - 154 * H[1] + 214 * H[2] - 156 * H[3]
            + 61 * H[4] - 10 * H[5]) / (12 * dm ** 2)


def dH_dm(H, dm):
    """Fourth-order dH/dm on the whole uniform m grid.

    Uniformly fourth order so that V = -dH/dm carries the SAME truncation error
    at the free edge as everywhere else; node 0 reproduces ``edge_slope``
    exactly, so the profile's tip velocity is bit-for-bit the series value and a
    disagreement between the two would be a bug rather than a stencil change.
    """
    H = np.asarray(H, dtype=float)
    n = H.size
    if n < 5:
        raise ValueError(f"need at least 5 nodes for the 4th-order stencils, "
                         f"got n={n}")
    d = np.empty(n)
    d[2:-2] = (H[:-4] - 8 * H[1:-3] + 8 * H[3:-1] - H[4:]) / (12 * dm)
    d[0] = (-25 * H[0] + 48 * H[1] - 36 * H[2] + 16 * H[3] - 3 * H[4]) / (12 * dm)
    d[1] = (-3 * H[0] - 10 * H[1] + 18 * H[2] - 6 * H[3] + H[4]) / (12 * dm)
    d[-1] = (25 * H[-1] - 48 * H[-2] + 36 * H[-3] - 16 * H[-4]
             + 3 * H[-5]) / (12 * dm)
    d[-2] = (3 * H[-1] + 10 * H[-2] - 18 * H[-3] + 6 * H[-4]
             - H[-5]) / (12 * dm)
    return d


def profile_from_mass(H, dm):
    r"""Map the mass-coordinate state onto the paper's (Xhat, H, V) profile.

    ``dm = H dX`` with the edge at ``m = 0``, so the physical offset from the
    free edge is the quadrature

        X(m) - X_c = \int_0^m dm' / H(m'),

    and the exact invariant (3.1) in mass form is ``V = -dH/dm``.  Nothing else
    is needed: no free boundary is reconstructed and no interpolation onto a
    physical grid takes place, so the only errors are the quadrature's and the
    difference stencil's, both fourth order here.

    The integrand ``1/H`` is smooth and monotone, so composite Simpson on the
    uniform grid is used rather than the trapezoidal rule.  Both are returned by
    ``profile_quadrature_check`` when the caller wants the difference as a
    diagnostic.
    """
    H = np.asarray(H, dtype=float)
    return _cumsimpson(1.0 / H, dm), H.copy(), -dH_dm(H, dm)


def _cumsimpson(f, dx):
    """Cumulative integral of f on a uniform grid, fourth-order accurate.

    Simpson pairs the intervals, so the value at every ODD node needs a
    half-panel.  That half-panel is taken with the fourth-order Newton-Cotes
    formula ``(5 f_i + 8 f_{i+1} - f_{i+2}) dx / 12`` rather than a trapezoid,
    which would drop the odd nodes to second order and put a visible sawtooth
    into ``Xhat`` at the 1e-6 level.
    """
    f = np.asarray(f, dtype=float)
    n = f.size
    out = np.zeros(n)
    # full Simpson panels over [i-2, i] for even i
    panels = (f[0:n - 2:2] + 4.0 * f[1:n - 1:2] + f[2:n:2]) * dx / 3.0
    out[2::2] = np.cumsum(panels)
    # odd nodes: previous even node plus a fourth-order half panel forward
    prev = np.concatenate([[0.0], out[2::2]])[:len(range(1, n, 2))]
    i = np.arange(1, n, 2)
    half = np.empty(i.size)
    ok = i + 1 < n
    half[ok] = (5.0 * f[i[ok] - 1] + 8.0 * f[i[ok]] - f[i[ok] + 1]) * dx / 12.0
    if not ok.all():
        # The last node is odd (n even), so there is no node to its right.
        # Integrate [j-1, j] with the quadratic through j-2, j-1, j instead:
        # the SECOND half panel of Simpson, h(-f0 + 8 f1 + 5 f2)/12, which is
        # interpolatory rather than extrapolatory and still fourth order.
        j = i[~ok]
        half[~ok] = (-f[j - 2] + 8.0 * f[j - 1] + 5.0 * f[j]) * dx / 12.0
    out[1::2] = prev + half
    return out


def residual_and_jac(H, Hold, Hold2, dt, dt_prev, dm, T_new, first, H0_edge=1.0,
                     finite=False):
    """BDF2 (or BDF1 on the first step) residual of H_T = H^2 H_mm.

    ``finite`` selects the fixed-end condition of a sheet of finite length in
    place of the semi-infinite far field.  The reference problem imposes
    ``V = 0`` and ``dH/dX = 0`` there (their 2.19a,b); since ``V = -dH/dm`` and
    ``dH/dX = H dH/dm`` with ``H > 0``, both reduce to the single Neumann
    condition ``dH/dm = 0`` at ``m = M``.  It is applied through a mirrored
    ghost node, ``H[n] = H[n-2]``, which keeps the operator second order and the
    Jacobian tridiagonal; a one-sided first-derivative stencil would do neither.
    The mass domain is then the whole sheet, ``M = L``, because ``dm = H dX``
    and ``H = 1`` on ``0 < X < L`` at ``T = 0``.
    """
    n = H.size
    # edge_slope and edge_curvature use one-sided stencils reaching H[4]
    # and H[5]; a coarse dm or small M can produce fewer nodes than that
    # and the failure would surface as an obscure IndexError mid-run.
    if n < 6:
        raise ValueError(f"need at least 6 nodes for the edge stencils, "
                         f"got n={n}; reduce --dm or raise --M")
    Hmm = np.zeros(n)
    Hmm[1:-1] = (H[:-2] - 2 * H[1:-1] + H[2:]) / dm ** 2
    F = np.zeros(n)
    if first:
        a0, a1, a2 = 1.0 / dt, -1.0 / dt, 0.0
    else:
        # variable-step BDF2
        r = dt / dt_prev
        a0 = (1 + 2 * r) / (1 + r) / dt
        a1 = -(1 + r) / dt
        a2 = r ** 2 / (1 + r) / dt
    F[1:-1] = (a0 * H[1:-1] + a1 * Hold[1:-1] + a2 * Hold2[1:-1]
               - H[1:-1] ** 2 * Hmm[1:-1])
    F[0] = H[0] - (H0_edge + T_new)

    ab = np.zeros((3, n))                      # banded (l=1, u=1)
    ab[1, 1:-1] = a0 - 2 * H[1:-1] * Hmm[1:-1] + 2 * H[1:-1] ** 2 / dm ** 2
    ab[0, 2:] = -H[1:-1] ** 2 / dm ** 2        # superdiagonal, rows 1..n-2
    ab[2, :-2] = -H[1:-1] ** 2 / dm ** 2       # subdiagonal
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0

    if finite:
        # Ghost node H[n] = H[n-2] gives Hmm[-1] = 2 (H[-2] - H[-1]) / dm^2.
        Hmm_end = 2.0 * (H[-2] - H[-1]) / dm ** 2
        F[-1] = (a0 * H[-1] + a1 * Hold[-1] + a2 * Hold2[-1]
                 - H[-1] ** 2 * Hmm_end)
        ab[1, -1] = (a0 - 2 * H[-1] * Hmm_end + 2 * H[-1] ** 2 / dm ** 2)
        ab[2, -2] = -2.0 * H[-1] ** 2 / dm ** 2
    else:
        F[-1] = H[-1] - 1.0
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0
    return F, ab


def _write_profile(pdir, tag, T, H, dm, m):
    """One CSV per requested T: Xhat, H, V, m, with a quadrature self-check.

    Column order and naming follow the campaign's own profile files
    (``ahsan_campaign._write_profiles``) so that a comparison downstream needs
    no per-solver special case.  Unlike TC1D-FEM's staggered pair, H and V live
    on the SAME grid here -- both are functions of the single nodal H -- so one
    file per time is enough and nothing is interpolated.
    """
    Xhat, Hp, V = profile_from_mass(H, dm)
    # Diagnostic on the quadrature that defines the abscissa: the trapezoidal
    # rule is second order and Simpson fourth, so their difference is dominated
    # by the trapezoidal error and is a CONSISTENCY CHECK, not a bound on the
    # Simpson error. It is reported rather than assumed small; a genuine error
    # estimate would need a refined-grid comparison.
    trap = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / H[1:] + 1.0 / H[:-1])
                                            * dm)])
    quad_gap = float(np.max(np.abs(Xhat - trap)))
    path = Path(pdir) / f"{tag}_profile_T{T:.6g}.csv"
    np.savetxt(path, np.column_stack([Xhat, Hp, V, m]), delimiter=",",
               header="Xhat,H,V,m", comments="")
    return {"T": float(T), "file": path.name, "H_tip": float(Hp[0]),
            "V_tip": float(V[0]), "Xhat_max": float(Xhat[-1]),
            "simpson_minus_trapezoid": quad_gap}


def run(Tend, dm, M=None, dT0=1e-6, cfl=2e-3, dTmax=0.05, nout=2000,
        tol=1e-12, out=None, tag="mass", delta=0.0, lam=1.0, finite=False,
        uniform_output=False, profile_at=(), profile_out=None):
    """delta, lam: an alternative initial condition H(m,0) = 1 + delta e^{-m/lam},
    a sheet that already carries a thickened edge at the instant of rupture.

    It is admissible because H_T(0,T) = 1 follows from the edge stress condition
    for ANY initial thickness, so the exact edge condition becomes
    H(0,T) = 1 + delta + T.  The far field is still H -> 1, so mass is still
    finite.  The point of the test is that U_{-2} is claimed to depend on the
    whole history: if that is right, delta != 0 must move it, and if it does not
    move, 3/2 is a universal constant of the model rather than of this initial
    condition.

    OPTIONAL OUTPUT MODES.  Both are off by default and neither touches the
    numerics; they decide only WHEN the state is sampled and WHAT is written.

    ``uniform_output``  writes the edge series on ``linspace(Tend/nout, Tend,
        nout)`` instead of the default ``geomspace(1e-3, Tend, nout)``.  The
        geometric default is right for a log-log deficit plot but unusable for
        the fig. 5 estimator ``ahsan_campaign.tc_dtc``, which differentiates on
        the frame index and so requires a single dT.  Resampling a geometric
        record onto a uniform grid and then differencing it is NOT an
        alternative: the interpolation error varies along the record with the
        local spacing, so the finite difference of the interpolant is not the
        derivative of the solution.  The schedule is changed at source instead.

    ``profile_at``  a list of T at which to write the full (Xhat, H, V) profile
        through ``profile_from_mass``.  The stepper is clipped onto each of
        these times exactly, as it already is onto the output times, so the
        profile is the state at that T and not at a nearby one.
    """
    if finite:
        # For a finite sheet M is the physical length L, not a truncation
        # distance. The semi-infinite default below is chosen to keep the front
        # away from the far boundary and would silently model a sheet of length
        # Tend + 40 instead -- a wrong answer that looks entirely plausible.
        if M is None:
            raise ValueError("--finite requires an explicit --M: the mass "
                             "domain IS the sheet length L, so it cannot be "
                             "defaulted from Tend")
    M = M if M is not None else Tend + 40.0
    n = int(round(M / dm)) + 1
    m = np.linspace(0.0, M, n)
    dm = m[1] - m[0]
    H = 1.0 + delta * np.exp(-m / lam)
    Hold2 = H.copy()
    T = 0.0
    dt = dT0
    dt_prev = dT0
    first = True
    if uniform_output:
        # Uniform in T, first sample at dT_out and last exactly at Tend, so
        # dT_out = Tend/nout is the single spacing tc_dtc differentiates on.
        Tout = np.linspace(Tend / nout, Tend, nout)
    else:
        Tout = np.geomspace(1e-3, Tend, nout)
        # The geometric output schedule starts at 1e-3; if Tend is at or below
        # that, no output time is ever reached, rows stays empty, and the
        # summary indexes an empty array. Fail with the reason instead.
        if Tend <= 1e-3:
            raise ValueError(f"--Tend must exceed the first output time 1e-3, "
                             f"got {Tend}")
    Tprof = np.unique(np.asarray(sorted(profile_at), dtype=float))
    # Reject rather than silently drop: a mistyped time that falls outside the
    # run would otherwise produce a run with one fewer profile than asked for
    # and no indication of which one is missing.
    if Tprof.size and np.any(~np.isfinite(Tprof) | (Tprof <= 0.0)
                             | (Tprof > Tend * (1.0 + 1e-15))):
        raise ValueError(f"--profile-at values must lie in (0, {Tend}]; got "
                         f"{np.asarray(sorted(profile_at))}")
    if Tprof.size and profile_out is None and out is None:
        raise ValueError("--profile-at needs somewhere to write: pass --out "
                         "or --profile-out")
    pdir = Path(profile_out) if profile_out is not None else (
        Path(out) if out is not None else None)
    if Tprof.size:
        pdir.mkdir(parents=True, exist_ok=True)
    prof_meta = []
    ip = 0
    rows = []
    io = 0
    t0 = time.time()
    nstep = 0
    while T < Tend - 1e-14:
        # step size: fine at the start, then a fixed fraction of the elapsed
        # time, which keeps the relative BDF2 error uniform in log T.
        dt = min(max(dT0, cfl * (1.0 + T)), dTmax, Tend - T)
        if io < nout and T + dt > Tout[io]:
            dt = Tout[io] - T
        if ip < Tprof.size and T + dt > Tprof[ip]:
            dt = Tprof[ip] - T
        Hold = H.copy()
        Hn = H + (H - Hold2) * (dt / dt_prev) if not first else H.copy()
        T_new = T + dt
        for _ in range(30):
            F, ab = residual_and_jac(Hn, Hold, Hold2, dt, dt_prev, dm,
                                     T_new, first, 1.0 + delta, finite)
            d = solve_banded((1, 1), ab, F)
            Hn -= d
            if np.max(np.abs(d)) < tol * max(1.0, np.max(np.abs(Hn))):
                break
        else:
            raise RuntimeError(f"Newton failed at T={T_new}")
        Hold2, H = Hold, Hn
        dt_prev, T = dt, T_new
        first = False
        nstep += 1
        if io < nout and T >= Tout[io] - 1e-14:
            slope = edge_slope(H, dm)
            # Far-field probe INTERIOR to the domain. H[-1] is Dirichlet-forced
            # to 1, so H[-1] - 1 is identically zero and says nothing about
            # whether M is large enough; the honest diagnostic is the largest
            # departure from H = 1 in the outer tenth of the domain, which is
            # what an insufficient M would contaminate first.
            j = int(0.9 * (n - 1))
            rows.append([T, -slope, 1.0 + slope,
                         edge_curvature(H, dm) * (1.0 + delta + T) ** 2,
                         float(np.max(np.abs(H[j:] - 1.0)))])
            io += 1
        if ip < Tprof.size and T >= Tprof[ip] - 1e-14:
            prof_meta.append(_write_profile(pdir, tag, T, H, dm, m))
            ip += 1
    tab = np.array(rows)
    meta = dict(code="TC1D-MASS", tag=tag, Tend=Tend, dm=dm, M=M, n=n,
                finite=finite,
                cfl=cfl, dTmax=dTmax, delta=delta, lam=lam,
                nstep=nstep, wall_s=time.time() - t0,
                V_final=float(tab[-1, 1]),
                # tab[:20] is the start-up transient, where the edge curvature has not
        # settled. np.max over an empty slice raises, so a short diagnostic
        # run (nout <= 20) reports the check as unavailable rather than
        # crashing.
        bc_curvature_check=(float(np.max(np.abs(tab[20:, 3] - 1.0)))
                            if len(tab) > 20 else None),
                far_field_interior=float(np.max(np.abs(tab[:, 4]))))
    if uniform_output:
        meta["output_schedule"] = "uniform"
        meta["dT_out"] = float(Tend / nout)
    if Tprof.size:
        meta["profiles"] = prof_meta
        meta["profile_dir"] = str(pdir)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / f"{tag}_series.csv", tab, delimiter=",",
                   header="T,V,deficit,bc_curv_times_1pT2,far_field",
                   comments="")
        (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    for Tq in (10, 20, 50, 100, 150, 200, 300, 400):
        if Tq <= Tend:
            i = int(np.argmin(np.abs(tab[:, 0] - Tq)))
            print(f"  T={tab[i,0]:8.3f}  1-V={tab[i,2]:.6e}  "
                  f"theta^2(1-V)={(1+delta+tab[i,0])**2*tab[i,2]:.6f}", flush=True)
    return tab, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tend", type=float, default=200.0)
    ap.add_argument("--dm", type=float, default=0.01)
    ap.add_argument("--M", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=2e-3)
    ap.add_argument("--dTmax", type=float, default=0.05)
    ap.add_argument("--nout", type=int, default=2000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tag", type=str, default="mass")
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--finite", action="store_true",
                    help="fixed-end sheet of length L = M (Neumann dH/dm = 0) "
                         "instead of the semi-infinite far field")
    ap.add_argument("--uniform-output", action="store_true",
                    help="write the edge series on a UNIFORM T grid, "
                         "linspace(Tend/nout, Tend, nout), instead of the "
                         "geometric default; required by the fig. 5 estimator "
                         "ahsan_campaign.tc_dtc, which assumes a single dT")
    ap.add_argument("--profile-at", type=str, default="",
                    help="comma-separated T at which to write the full "
                         "(Xhat, H, V) profile; the stepper lands on each "
                         "exactly. Off by default")
    ap.add_argument("--profile-out", type=str, default=None,
                    help="directory for the profile CSVs (default: --out)")
    a = ap.parse_args()
    prof = [float(q) for q in a.profile_at.split(",") if q.strip()]
    run(a.Tend, a.dm, a.M, cfl=a.cfl, dTmax=a.dTmax, nout=a.nout,
        out=a.out, tag=a.tag, delta=a.delta, lam=a.lam, finite=a.finite,
        uniform_output=a.uniform_output, profile_at=prof,
        profile_out=a.profile_out)
