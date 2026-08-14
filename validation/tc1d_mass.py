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


def run(Tend, dm, M=None, dT0=1e-6, cfl=2e-3, dTmax=0.05, nout=2000,
        tol=1e-12, out=None, tag="mass", delta=0.0, lam=1.0, finite=False):
    """delta, lam: an alternative initial condition H(m,0) = 1 + delta e^{-m/lam},
    a sheet that already carries a thickened edge at the instant of rupture.

    It is admissible because H_T(0,T) = 1 follows from the edge stress condition
    for ANY initial thickness, so the exact edge condition becomes
    H(0,T) = 1 + delta + T.  The far field is still H -> 1, so mass is still
    finite.  The point of the test is that U_{-2} is claimed to depend on the
    whole history: if that is right, delta != 0 must move it, and if it does not
    move, 3/2 is a universal constant of the model rather than of this initial
    condition.
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
    Tout = np.geomspace(1e-3, Tend, nout)
    # The geometric output schedule starts at 1e-3; if Tend is at or below
    # that, no output time is ever reached, rows stays empty, and the
    # summary indexes an empty array. Fail with the reason instead.
    if Tend <= 1e-3:
        raise ValueError(f"--Tend must exceed the first output time 1e-3, "
                         f"got {Tend}")
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
    a = ap.parse_args()
    run(a.Tend, a.dm, a.M, cfl=a.cfl, dTmax=a.dTmax, nout=a.nout,
        out=a.out, tag=a.tag, delta=a.delta, lam=a.lam, finite=a.finite)
