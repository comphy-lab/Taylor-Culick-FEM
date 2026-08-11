#!/usr/bin/env python3
r"""Python/numpy port of the Ahsan-Brandao-Davidovitch-Stone thin-film solvers.

SOURCE. https://github.com/taosifahsan/Thin-Film-Simulations, the code released
with

    T. Ahsan, R. Brandao, B. Davidovitch, H. A. Stone,
    "Retraction of a viscous liquid sheet", arXiv:2507.04603 (2025),

local copy of the PDF in ``validation/literature/``. The MATLAB is written from
scratch with no toolboxes, so the port needs numpy and nothing else.

WHAT IS PORTED HERE. Three of their scripts, verbatim in scheme and in tip
treatment, each in its own function:

  ``simulate_full``          <- ``thin_film.m``
      Their FULL thin-film model for a finite sheet in the LAB frame on
      Lagrangian nodes: inertia + Trouton viscosity + the full nonlinear
      curvature. Two dimensionless groups, ``Oh`` and ``epsilon = l0/h0``.
      Explicit predictor, Crank-Nicolson corrector, Picard iteration, tridiagonal
      Thomas solve for the implicit viscous term, capillary term lagged onto the
      right-hand side. This is the only one of the three that CONTAINS A BULK
      CAPILLARY TERM.

  ``simulate_reduced_finite`` <- ``thin_film_Oh_large.m``
      Their REDUCED (Oh >> 1) problem, eqs (2.15)-(2.19) collapsed by the
      conserved quantity (3.1) into the heat equation (3.8), for a FINITE sheet
      of dimensionless length ``L = l0 / (4 h0 Oh)``. Solved in the static frame
      of Appendix D: variables (D1), PDE (D2), BCs (D3), IC (D4), boundary
      evolution (D5) with the second-order edge treatment (D6)-(D10) and the
      two-stage ``s_step``/``h_step`` iteration (D11), all at ``t = tn + dt/2``.
      Matches Appendix D of the paper verbatim.

  ``simulate_reduced_infinite`` <- ``thin_film_L=infty_Oh_large.m``
      The same reduced problem at ``L = infinity``, solved in the TIP frame
      (their Sec. VII). This is the semi-infinite limit, and it is the closest
      structural match to our pyoomph 1D solver, which is also semi-infinite.

DEVIATIONS FROM THE MATLAB, in full.

  1. Indexing. MATLAB is 1-based, numpy is 0-based. Every loop bound and every
     boundary index is shifted accordingly. Their variable names are kept in
     comments beside each translated line so the correspondence is checkable by
     eye; where a comment says ``h_f(N-1)`` it means the MATLAB expression, and
     the code beside it is the 0-based equivalent ``h_f[-2]``.
  2. Vectorisation. ``h_step`` and the Crank-Nicolson matrix assembly in
     ``thin_film.m`` are written as explicit MATLAB ``for`` loops. They are
     evaluated here as numpy slice expressions. The arithmetic is identical
     term for term and in the same order of operations per entry; only the loop
     is gone. The Thomas solve is left as a loop because it is sequential.
  3. Recorded time. Their save blocks write ``t = linspace(0,T,Nframe)`` and
     then fill frame ``round(it/frame_step)``, which is off by up to one frame
     from the actual ``it*dt`` of the saved state (and ``thin_film_Oh_large.m``
     additionally tags the saved thickness with ``1+tn+dt`` rather than
     ``1+tn``). The ports keep their arrays bit-for-bit but ALSO return a
     ``t_true`` array holding the exact time of each saved frame. Comparisons in
     this module use ``t_true``. This is the only place the port knowingly
     produces a different number from the MATLAB, and it is a bookkeeping fix,
     not a scheme change.
  4. Plotting is not ported. The MATLAB plotting functions are figure cosmetics.
     ``tip_velocity_full`` reproduces the diagnostic inside their
     ``plot_tip_velocity``, which is the only plotting routine that computes a
     physical quantity.

CONVENTIONS, AND THE CONVERSION TO OUR SOLVER'S UNITS.

Ahsan et al. use the HALF-thickness ``h0`` (their fig. 1: the sheet is
``2 h0`` thick), and

    uTC = sqrt(gamma / (rho h0)),   Oh_paper = mu / sqrt(rho h0 gamma),
    lI  = 4 mu sqrt(h0/(rho gamma)) = 4 h0 Oh_paper,   tI = 4 mu h0 / gamma,
    X = x/lI,  T = t/tI,  H = h/h0,  V = v/uTC,  L = l0/lI.

Our solver (``simulationCases/trouton1d_pyoomph.py``) uses the FULL thickness
``H_full`` with ``rho = gamma = H_full = 1``, ``mu = Oh_SB sqrt(2)``,
``tau_vis = mu H_full / (2 gamma) = mu/2``, ``t* = t/tau_vis``,
``V_TC = sqrt(2 gamma/(rho H_full)) = sqrt(2)``, ``u* = u/V_TC``. Hence
``h0 = H_full/2``.

    uTC      = sqrt(gamma/(rho H_full/2)) = sqrt(2 gamma/(rho H_full)) = V_TC
               -> their V IS our u*, with no factor.
    Oh_paper = mu / sqrt(rho gamma H_full/2) = mu sqrt(2) = 2 Oh_SB.
    tI       = 4 mu (H_full/2)/gamma = 2 mu H_full/gamma = 4 tau_vis
               -> T = t/tI = t*/4.
    H        = h_half/h0 = h_full/H_full -> their H IS our h, both 1 when flat.

Both conversions are confirmed independently by the physics rather than only by
algebra: their eq. (3.10a) is ``H(Xc,T) = 1 + T`` and our solver carries the
exact tip thickness ``h_tip = 1 + t*/4`` (Savva & Bush eq. 2.29); the two agree
only for ``T = t*/4``. And their tip condition (2.18a) ``H V_X = -1`` becomes,
in our units, ``2 mu h u_x = -1``, i.e. ``h^2 u_m = -1/(2 mu)``, which is
exactly our ``TipFlux``.

THE REDUCED MODEL IS OUR MODEL. Their (2.15)-(2.16) are

    H_T + (H V)_X = 0,      H DV/DT = (H V_X)_X,

with the free-end conditions (2.18). Pushed into our units these are

    h_t + (h u)_x = 0,      Du/Dt = (4 mu / h) (h u_x)_x,   h u_x|tip = -1/(2mu)

which is our solver's system with ``with_capillarity=False`` -- the SAME
equations, not a related model. Their capillary term is absent because they
prove it negligible at ``Oh >> 1`` (their eq. 2.11), not because they dropped it
for convenience.

A COROLLARY WORTH STATING: the reduced problem has NO parameter except ``L``.
So ``u*(t*)`` from our solver must be independent of ``Oh_SB`` when capillarity
is off, and at ``L = infinity`` it must equal one single universal curve. That
is a check both codes can be held to.

VERIFICATION ACHIEVED (2026-08-11).

  1. AGAINST THE ORIGINAL MATLAB. GNU Octave 10.3.0 was installed user-locally
     on reynolds (micromamba, conda-forge; it is not a system package there and
     there is no passwordless sudo). All three of their scripts were run
     unmodified apart from replacing the plotting calls with an ASCII dump, and
     diffed against this port at identical parameters:

       thin_film_L=infty_Oh_large.m  N=500,  NT=5000,  T=2.5   max |dU| 5.0e-10
       thin_film_Oh_large.m          N=1000, NT=5000,  T=1     max |dU| 5.0e-10
       thin_film.m  Oh=10, eps=10,   N=2000, NT=10000, T=2     max |dv| 5.0e-10

     including full end-of-run profiles of x, v and h/H. 5e-10 IS the write
     precision of Octave's ``save -ascii`` (about ten significant digits), so
     the ports agree with the MATLAB to every digit the comparison can see.
     The only column that differs is their frame-time array, by exactly one
     frame -- deviation 3 above, confirmed rather than assumed.

  2. AGAINST THE PAPER. ``verify`` checks eq (5.10) (2.6% at the top of the
     early window, converging as O(T) below it), eq (7.11) (4.7e-3 for T > 20),
     U -> 1, fig. 5's Tc = L (20.05 against 20), and eq (3.10a) H(Xc,T) = 1+T on
     the FULL solver, which does not know about it (4.4%).

  3. AGAINST OUR SOLVER. See the ``compare`` subcommand.

USAGE

    uv run python validation/ahsan_thinfilm.py verify
    uv run python validation/ahsan_thinfilm.py reduced-inf --T 100 --out u.csv
    uv run python validation/ahsan_thinfilm.py reduced-fin --L 400 --T 100
    uv run python validation/ahsan_thinfilm.py full --oh 10 --epsilon 40
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversions between the two papers' conventions. See the module
# docstring for the derivation; these are the only place the mapping is
# written down, so a comparison script cannot quietly use a different one.
# ---------------------------------------------------------------------------

def oh_paper_from_oh_sb(oh_sb: float) -> float:
    """Oh_paper = mu/sqrt(rho h0 gamma) with h0 = H_full/2, i.e. 2 Oh_SB."""
    return 2.0 * oh_sb


def T_from_tstar(tstar):
    """T = t/tI, tI = 4 mu h0/gamma = 4 tau_vis, so T = t*/4."""
    return np.asarray(tstar) / 4.0


def tstar_from_T(T):
    return np.asarray(T) * 4.0


# ---------------------------------------------------------------------------
# Finite-difference helpers, ported one-for-one.
#
# NOTE there are TWO different D/D2 pairs in their repository. thin_film.m uses
# a NON-UNIFORM-x version whose end values are extrapolated from a one-sided
# difference; the Oh>>1 scripts use a uniform-dx version whose end values are
# extrapolated from the interior stencil alone. They are NOT interchangeable and
# are kept separate here on purpose.
# ---------------------------------------------------------------------------

def _D_nonuniform(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """thin_film.m ``function y = D(f,x)``: df/dx on Lagrangian nodes."""
    N = f.size
    df = np.zeros(N)
    dx = np.ones(N)
    df[1:N - 1] = (f[2:N] - f[0:N - 2]) / 2.0
    dx[1:N - 1] = (x[2:N] - x[0:N - 2]) / 2.0
    y = df / dx
    y0 = (f[1] - f[0]) / (x[1] - x[0])            # y0
    yN = (f[N - 1] - f[N - 2]) / (x[N - 1] - x[N - 2])   # yN
    y[0] = 2.0 * y0 - y[1]                        # y(1)=2*y0-y(2)
    y[N - 1] = 2.0 * yN - y[N - 2]                # y(N)=2*yN-y(N-1)
    return y


def _D2_nonuniform(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """thin_film.m ``function y = D2(f,x)``."""
    N = f.size
    y = np.zeros(N)
    dx = (x[2:N] - x[0:N - 2]) / 2.0
    y[1:N - 1] = (f[2:N] + f[0:N - 2] - 2.0 * f[1:N - 1]) / dx ** 2
    y[0] = 2.0 * y[1] - y[2]                      # y(1)=2*y(2)-y(3)
    y[N - 1] = 2.0 * y[N - 2] - y[N - 3]          # y(N)=2*y(N-1)-y(N-2)
    return y


def _D_uniform(f: np.ndarray, dx: float, endstyle: str) -> np.ndarray:
    """``function y = D(f,dx)``.

    ``endstyle="onesided"`` is thin_film_Oh_large.m
    (``y(1) = 2*(f(2)-f(1))/dx - y(2)``); ``endstyle="interior"`` is
    thin_film_L=infty_Oh_large.m (``y(1) = 2*y(2)-y(3)``). The two files really
    do differ here.
    """
    N = f.size
    df = np.zeros(N)
    df[1:N - 1] = (f[2:N] - f[0:N - 2]) / 2.0
    y = df / dx
    if endstyle == "onesided":
        y[0] = 2.0 * (f[1] - f[0]) / dx - y[1]
        y[N - 1] = 2.0 * (f[N - 1] - f[N - 2]) / dx - y[N - 2]
    else:
        y[0] = 2.0 * y[1] - y[2]
        y[N - 1] = 2.0 * y[N - 2] - y[N - 3]
    return y


def _D2_uniform(f: np.ndarray, dx: float) -> np.ndarray:
    """``function y = D2(f,dx)``; identical in both Oh>>1 scripts."""
    N = f.size
    y = np.zeros(N)
    y[1:N - 1] = (f[2:N] + f[0:N - 2] - 2.0 * f[1:N - 1]) / dx ** 2
    y[0] = 2.0 * y[1] - y[2]
    y[N - 1] = 2.0 * y[N - 2] - y[N - 3]
    return y


# DEVIATION 5, OPT-IN AND VALIDATED. The ported Thomas sweep is a sequential
# Python loop, which costs ~1 us per node per solve; at their own resolutions
# (NT = 1e6, N = 1000-3000) that is many hours per case. ``use_fast_solver(True)``
# swaps in LAPACK's banded solve, which is the same factorisation with partial
# pivoting. It is NOT the default, and ``verify`` measures the difference
# between the two backends rather than assuming it is negligible: on these
# strictly diagonally dominant matrices they agree to round-off.
_FAST = False


def use_fast_solver(flag: bool) -> None:
    global _FAST
    if flag:
        import scipy.linalg  # noqa: F401  (import here so numpy-only use works)
    _FAST = flag


def solve_tridiagonal(A_0: np.ndarray, A_minus: np.ndarray,
                      A_plus: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``function x = solve_tridiagnoal(A_0,A_minus,A_plus,d)`` (their spelling).

    Thomas algorithm, no pivoting, exactly as in their appendix B. Kept as a
    Python loop: the recurrence is sequential and vectorising it would change
    the arithmetic. A_0 and d are copied because the MATLAB overwrites them
    locally (pass-by-value) and the callers rely on that.
    """
    n = d.size
    if _FAST:
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n))
        ab[0, 1:] = A_plus[:n - 1]      # superdiagonal: row i has A_plus(i)
        ab[1, :] = A_0
        ab[2, :n - 1] = A_minus[1:]     # subdiagonal:   row i has A_minus(i)
        return solve_banded((1, 1), ab, d)
    a0 = np.array(A_0, dtype=float)
    dd = np.array(d, dtype=float)
    x = np.zeros(n)
    for i in range(1, n):
        w = A_minus[i] / a0[i - 1]
        a0[i] = a0[i] - w * A_plus[i - 1]
        dd[i] = dd[i] - w * dd[i - 1]
    x[n - 1] = dd[n - 1] / a0[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (dd[i] - A_plus[i] * x[i + 1]) / a0[i]
    return x


# ===========================================================================
# 1. thin_film.m -- FULL model, finite sheet, lab frame, Lagrangian nodes
# ===========================================================================

def initial_condition_full(epsilon: float, N: int, h0: float = 1.0,
                           v0: float = 0.0):
    """``function [x,v,h] = initial_condition(epsilon,N,h0,v0)``.

    The profile is their smoothed sqrt tip: with xp = L0 - x and a = 1/20,
    f = 1/2 - a - xp + (1/2) sqrt((1+2a)^2 + 4 xp (xp + 2a - 1)) and
    h = h0 sqrt(1 - f^2). Note x = 0 is the FREE EDGE and x = L0 the clamped
    end, opposite to the paper's fig. 1 -- their code retracts to the right.
    """
    x = np.linspace(0.0, epsilon * h0, N)
    v = v0 * np.zeros(N)
    L0 = h0 * epsilon
    xp = L0 - x
    a = 1.0 / 20.0
    f = 0.5 - a - xp + 0.5 * np.sqrt((1 + 2 * a) ** 2 + 4 * xp * (xp + 2 * a - 1))
    h = h0 * np.sqrt(1.0 - f ** 2)
    return x, v, h


def _kappa_x(h: np.ndarray, hx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``function kx = kappa_x(h,hx,x)``: full nonlinear curvature gradient.

    kappa = h_xx/(1+h_x^2)^{3/2} with h the HALF-thickness. Identical to our
    solver's ``0.5 h_xx/(1+0.25 h_x^2)^{3/2}`` written for the full thickness.
    """
    hxx = _D2_nonuniform(h, x)
    k = hxx / (1.0 + hx ** 2) ** 1.5
    return _D_nonuniform(k, x)


def _v_step_explicit(h_n, v_n, x_n, dt, Oh):
    """``function v_f = v_step_explicit(h_n,v_n,x_n,dt,Oh)``."""
    hx_n = _D_nonuniform(h_n, x_n)
    kx_n = _kappa_x(h_n, hx_n, x_n)
    vx_n = _D_nonuniform(v_n, x_n)
    vxx_n = _D2_nonuniform(v_n, x_n)
    # h_n(N) = 0 by construction, so hx_n*vx_n/h_n is Inf/NaN at the last node
    # -- in MATLAB too. It is overwritten two lines down by v_f(N)=v_f(N-1),
    # so it never propagates; the errstate only silences the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        dv_dt = 4.0 * Oh * (hx_n * vx_n / h_n + vxx_n) + kx_n
    v_f = v_n + dt * dv_dt
    v_f[0] = 0.0                # v_f(1)=0        -> v(0)=0
    v_f[-1] = v_f[-2]           # v_f(N)=v_f(N-1) -> v_x(L)=0
    return v_f


def _v_step_crank_nicolson(h_n, v_n, x_n, h_f, x_f, dt, Oh):
    """``function v_f = v_step_crank_nicolson(h_n,v_n,x_n,h_f,x_f,dt,Oh)``.

    The viscous term is implicit (tridiagonal); the capillary term kx_f is
    LAGGED -- it is evaluated on the current Picard iterate of (h_f, x_f) and
    enters the right-hand side, so the matrix stays linear.
    """
    N = x_n.size
    hx_n = _D_nonuniform(h_n, x_n)
    kx_n = _kappa_x(h_n, hx_n, x_n)
    vx_n = _D_nonuniform(v_n, x_n)
    vxx_n = _D2_nonuniform(v_n, x_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        vt_exp = 4.0 * Oh * (hx_n * vx_n / h_n + vxx_n) + kx_n  # NaN at N, see above

    hx_f = _D_nonuniform(h_f, x_f)

    A_0 = np.zeros(N)
    A_plus = np.zeros(N)
    A_minus = np.zeros(N)
    # for ix = 2:N-1  ->  0-based 1..N-2
    dx_f = (x_f[2:N] - x_f[0:N - 2]) / 2.0
    fact_A = 4.0 * Oh * dt / dx_f ** 2
    fact_B = hx_f[1:N - 1] / h_f[1:N - 1] * dx_f / 2.0
    A_0[1:N - 1] = 1.0 + fact_A
    A_plus[1:N - 1] = -(1.0 + fact_B) * fact_A / 2.0
    A_minus[1:N - 1] = -(1.0 - fact_B) * fact_A / 2.0

    A_0[0] = 1.0                # A_0(1)=1        -> v(0)=0
    A_minus[-1] = -1.0          # A_minus(N)=-1   -> v_x(L)=0
    A_0[-1] = 1.0               # A_0(N)=1

    kx_f = _kappa_x(h_f, hx_f, x_f)
    d = v_n + dt * kx_f / 2.0 + dt * vt_exp / 2.0
    d[0] = 0.0                  # d(1)=0
    d[-1] = 0.0                 # d(N)=0
    return solve_tridiagonal(A_0, A_minus, A_plus, d)


def _h_step(h_n, v_n, x_n, v_f, x_f, dt):
    """``function h_f = h_step(h_n,v_n,x_n,v_f,x_f,dt)``.

    Lagrangian mass conservation h_f = h_n (1 - dv_i/dx_i dt/2)/(1 + dv_f/dx_f
    dt/2), i.e. Crank-Nicolson on ``Dh/Dt = -h v_x``. Then their TIP TREATMENT,
    which is a hard-imposed shape rather than a boundary condition on a flux:

        h_f(N)   = 0
        ratio    = sqrt((x_f(N)-x_f(N-1))/(x_f(N)-x_f(N-2)))
        h_f(N-1) = h_f(N-2) * ratio            [ h ~ (L-x)^{1/2} imposed ]
        h_f(1)   = h_f(2)                      [ h_x(0)=0 at the clamped end ]
    """
    N = x_n.size
    h_f = h_n.copy()
    # for ix = 2:N-2  ->  0-based 1..N-3
    s = slice(1, N - 2)
    sp = slice(2, N - 1)
    sm = slice(0, N - 3)
    dvi = (v_n[sp] - v_n[sm]) / 2.0
    dxi = (x_n[sp] - x_n[sm]) / 2.0
    dxf = (x_f[sp] - x_f[sm]) / 2.0
    dvf = (v_f[sp] - v_f[sm]) / 2.0
    h_f[s] = h_n[s] / (1.0 + dvf / dxf * dt / 2.0) * (1.0 - dvi / dxi * dt / 2.0)

    h_f[-1] = 0.0                                             # h_f(N)=0
    ratio = math.sqrt((x_f[-1] - x_f[-2]) / (x_f[-1] - x_f[-3]))
    h_f[-2] = h_f[-3] * ratio                                 # h_f(N-1)
    h_f[0] = h_f[1]                                           # h_f(1)=h_f(2)
    return h_f


def simulate_full(x_0, v_0, h_0, Oh, T, NT, Nframe, verbose=False):
    """``function [t,x,v,h] = simulate(x_0,v_0,h_0,Oh,T,NT,Nframe)``.

    Integrates ``v_t + v v_x = 4 Oh (h v_x)_x / h + kappa_x`` on Lagrangian
    nodes, then rescales to the paper's variables by ``T = t/(4 Oh)``,
    ``X = x/(4 Oh)`` on the way out.

    Returns ``(t, x, v, h, t_true)``; ``t`` is their (off-by-a-frame) array and
    ``t_true`` the exact time of each saved frame, both already divided by 4 Oh.
    Arrays are shaped ``(N, Nframe)`` as in MATLAB.
    """
    x_n = np.array(x_0, dtype=float)
    v_n = np.array(v_0, dtype=float)
    h_n = np.array(h_0, dtype=float)
    N = x_n.size

    dt = T / NT
    frame_step = int(round(NT / Nframe))
    h = np.zeros((N, Nframe))
    x = np.zeros((N, Nframe))
    v = np.zeros((N, Nframe))
    t = np.linspace(0.0, 0.0, Nframe)
    t_true = np.zeros(Nframe)
    nsaved = 0

    t0 = time.time()
    for it in range(1, NT + 1):
        # explicit predictor
        v_f = _v_step_explicit(h_n, v_n, x_n, dt, Oh)
        x_f = x_n + (v_n + v_f) / 2.0 * dt
        h_f = _h_step(h_n, v_n, x_n, v_f, x_f, dt)

        # Crank-Nicolson corrector, Picard iteration on (v_f, x_f, h_f)
        MSE = 1e12
        convergence_cutoff = 1e-9
        Num_repeats = 0
        Num_Failure = 20
        while MSE > convergence_cutoff and Num_repeats < Num_Failure:
            v_temp = v_f
            v_f = _v_step_crank_nicolson(h_n, v_n, x_n, h_f, x_f, dt, Oh)
            MSE = float(np.sum((v_temp - v_f) ** 2))
            x_f = x_n + (v_n + v_f) / 2.0 * dt
            h_f = _h_step(h_n, v_n, x_n, v_f, x_f, dt)
            Num_repeats += 1

        if Num_repeats == Num_Failure:
            print(f"CONVERGENCE FAILED, MSE = {MSE:5.4e}", file=sys.stderr)
            break

        if it % frame_step == 0:
            j = int(round(it / frame_step)) - 1     # MATLAB round(it/frame_step)
            if 0 <= j < Nframe:
                # NOTE they store the PRESENT state h_n at time index it, so the
                # frame holds the state at (it-1)*dt. t_true records that.
                t[j] = it / NT * T
                h[:, j] = h_n
                x[:, j] = x_n
                v[:, j] = v_n
                t_true[j] = (it - 1) * dt
                nsaved = j + 1

        x_n, v_n, h_n = x_f, v_f, h_f

        if verbose and it % max(1, NT // 20) == 0:
            print(f"  it={it}/{NT} T={it/NT*T/(4*Oh):.4f} L={x_n[-1]:.4f} "
                  f"v(L)={-v_n[-1]:.4f} h(0)={h_n[0]:.4f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)

    # THEIR LOOP CAN BREAK EARLY. `Num_repeats == Num_Failure` aborts the run
    # even when the last Picard sweep did converge, so a case that is merely
    # slow to converge stops mid-way and the unfilled frames stay zero. Return
    # only the frames actually written, or every downstream diagnostic divides
    # by a zero tip position.
    n = nsaved
    return (t[:n] / (4.0 * Oh), x[:, :n] / (4.0 * Oh), v[:, :n], h[:, :n],
            t_true[:n] / (4.0 * Oh))


def tip_velocity_full(t, v, x, h, Oh):
    """The diagnostic inside their ``plot_tip_velocity``.

    U(T) is NOT v at the last node: they take the mass-weighted mean velocity
    over the tip region, whose extent is one rim thickness hmax = h(1,it) in the
    unscaled x, i.e. hmax/(4 Oh) in X. The index is found by assuming the nodes
    are evenly spread over [0, x(N)], which is only approximately true for
    Lagrangian nodes -- their approximation, kept.
    """
    N = v.shape[0]
    vmax = np.zeros(t.size)
    for it in range(t.size):
        hmax = h[0, it]
        xt = x[N - 1, it]
        if not np.isfinite(xt) or xt <= 0.0:
            vmax[it] = np.nan
            continue
        ix = int(round(N * (xt - hmax / (4.0 * Oh)) / xt))
        ix = min(max(ix, 1), N) - 1     # MATLAB ix:1:N -> 0-based ix-1..N-1
        w = h[ix:N, it]
        vmax[it] = float(np.sum(v[ix:N, it] * w) / np.sum(w))
    return -vmax


# ===========================================================================
# 2. thin_film_Oh_large.m -- REDUCED model, FINITE L, Appendix D static frame
# ===========================================================================

def _s_step(dt, dx, sn, tn, hn, hf):
    """``function sf = s_step(dt,dx,sn,tn,hn,hf)`` -- eqs (D5),(D9),(D10).

    h := dH'/dX' at X' = 1 - dX'/2 evaluated Crank-Nicolson in time, using
    H'(1,T) = 1 exactly (D3b):
        h  = (1 - H'(N-1,T)/2 - H'(N-1,T+dT)/2)/dX'          <- (D10)
        dS/dT|_{T+dT/2} = -2h - (h^2 + S/(1+T)) dX'          <- (D9)
    with T evaluated at tn + dt/2, the half step.
    """
    N = hn.size
    hx = (1.0 - hn[N - 2] / 2.0 - hf[N - 2] / 2.0) / dx      # hn(N-1),hf(N-1)
    t = tn + dt / 2.0
    st = -2.0 * hx - (hx ** 2 + sn / (1.0 + t)) * dx
    return sn + st * dt


def _h_step_reduced(dt, x, sn, sf, tn, hn):
    """``function hf = h_step(dt,x,sn,sf,tn,hn)`` -- Crank-Nicolson on (D2).

        dH'/dT = (1/S) H'_X'X'  +  (X'/(2S)) (dS/dT) H'_X'  -  H'/(1+T)

    with S = (Sn+Sf)/2, dS/dT = (Sf-Sn)/dt and T = tn + dt/2 (the half step),
    and the BCs (D3): H'(0) = H'(1) [zero slope] and H'(N) = 1.
    """
    dx = x[1] - x[0]
    N = x.size
    t = tn + dt / 2.0
    st = (sf - sn) / dt
    s = (sf + sn) / 2.0

    R = dt / dx ** 2 / s
    H = dt / (1.0 + t) / 2.0
    G = st * dt * x / (8.0 * s * dx)

    A_0 = np.empty(N)
    A_0[0] = 1.0
    A_0[1:N - 1] = 1.0 + R + H
    A_0[N - 1] = 1.0
    A_plus = np.zeros(N)
    A_plus[0] = -1.0                                  # H'(1)=H'(2): zero slope
    A_plus[1:N - 1] = -(R / 2.0 + G[1:N - 1])
    A_minus = np.zeros(N)
    A_minus[1:N - 1] = -(R / 2.0 - G[1:N - 1])

    ht = (_D2_uniform(hn, dx) / s
          + st / (2.0 * s) * x * _D_uniform(hn, dx, "onesided")
          - hn / (1.0 + t))

    d = np.empty(N)
    d[0] = 0.0
    d[1:N - 1] = hn[1:N - 1] + dt / 2.0 * ht[1:N - 1]
    d[N - 1] = 1.0                                    # H'(1,T)=1
    return solve_tridiagonal(A_0, A_minus, A_plus, d)


def _step_reduced(dt, x, sn, tn, hn):
    """``function [hf,sf] = step(dt,x,sn,tn,hn)`` -- the (D11) iteration.

    1. Sn -> Sf = S_step(Sn, Hn, Hn)      [rough, with Hf = Hn]
    2. Hn -> Hf = H_step(Hn, Sn, Sf)
    3. Sf     -> S_step(Sn, Hn, Hf)       [accurate]
    Their step 4, a second H_step, IS COMMENTED OUT in the MATLAB. Left out
    here too; uncommenting it would be a deviation, not a fix.
    """
    dx = x[1] - x[0]
    sf = _s_step(dt, dx, sn, tn, hn, hn)
    hf = _h_step_reduced(dt, x, sn, sf, tn, hn)
    sf = _s_step(dt, dx, sn, tn, hn, hf)
    return hf, sf


def simulate_reduced_finite(L, N, T, NT, Nframe=None, verbose=False):
    """``function [t,x,h,v] = simulate(L,N,T,NT,Nframe)`` of thin_film_Oh_large.m.

    Reduced (Oh >> 1) problem for a finite sheet, L = l0/(4 h0 Oh).

    Returns a dict with
      ``T``   exact frame times (the deviation-3 fix; their ``t`` is uniform),
      ``U``   retraction speed dXc/dT = V at the free edge,
      ``Xc``  L - sqrt(S), the free-edge position,
      ``H``   thickness profiles, shape (N, Nframe), and
      ``xw``  L - X = X' sqrt(S), distance from the clamped wall.

    Sign/index note: X' = (L-X)/(L-Xc), so index 0 is the CLAMPED WALL (X=L)
    and index N-1 is the FREE EDGE (X=Xc). Their ``v(N,:)`` is therefore the
    tip velocity.
    """
    if Nframe is None:
        Nframe = min(NT, 10 ** 5)
    frame_step = int(round(NT / Nframe))

    Hs = np.zeros((N, Nframe))
    xs = np.zeros((N, Nframe))
    vs = np.zeros((N, Nframe))
    Ss = np.zeros(Nframe)
    t_true = np.zeros(Nframe)

    sn = L ** 2                       # S(0) = L^2               (D5)
    hn = np.ones(N)                   # H'(X',0) = 1             (D4)
    x0 = np.linspace(0.0, 1.0, N)     # X' in [0,1]
    tn = 0.0
    dt = T / NT
    iframe = 0
    t0 = time.time()

    for it in range(1, NT + 1):
        if it % frame_step == 0 and iframe < Nframe:
            # verbatim: they tag the frame with (1+tn+dt) and X'*sqrt(sn)
            Hs[:, iframe] = hn * (1.0 + tn + dt)
            xs[:, iframe] = x0 * math.sqrt(sn)
            vs[:, iframe] = _D_uniform(np.log(Hs[:, iframe]),
                                       xs[1, iframe] - xs[0, iframe],
                                       "onesided")
            Ss[iframe] = sn
            t_true[iframe] = tn          # exact time of the state stored
            iframe += 1
        hn, sn = _step_reduced(dt, x0, sn, tn, hn)
        tn = tn + dt
        if verbose and it % max(1, NT // 20) == 0:
            print(f"  it={it}/{NT} T={tn:.4f} Xc={L-math.sqrt(sn):.5f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)

    n = iframe
    return {"T": t_true[:n], "U": vs[N - 1, :n], "S": Ss[:n],
            "Xc": L - np.sqrt(Ss[:n]), "H": Hs[:, :n], "xw": xs[:, :n],
            "L": L, "N": N, "NT": NT, "dt": dt}


# ===========================================================================
# 3. thin_film_L=infty_Oh_large.m -- REDUCED model, L = infinity, tip frame
# ===========================================================================
#
# Variables: Xhat = X - Xc(T) measured from the free edge, and the unknown is
# hh = H - 1 so that hh(0,T) = T [from H(Xc,T) = 1+T, eq (3.10a)] and
# hh -> 0 in the far field. In the tip frame the heat equation (3.8) becomes
#
#     hh_T = U hh_Xhat + hh_XhatXhat,     U = dXc/dT,
#
# and (3.11) closes U. Their ``U`` implements that closure with a quadratic
# correction rather than a plain one-sided difference; see below.

def _U_inf(h, t, dx):
    """``function y = U(h,t,dx)``.

    Eq (3.11) is U = -H_X(Xc)/(1+T). A plain one-sided difference of hh at the
    edge is only first order, so they instead solve the quadratic that comes
    from carrying the second-order Taylor term (the commented-out line in the
    MATLAB is the three-point one-sided alternative they did not use):

        hx = (h(2)-t)/dx,   c = (dx - 2 hx)/(1+t),   U = (1-sqrt(1-c dx))/dx.

    Note U -> c/2 as dx -> 0, and c/2 -> -hx/(1+t), which is (3.11).
    """
    hx = (h[1] - t) / dx
    c = (dx - 2.0 * hx) / (1.0 + t)
    return (1.0 - math.sqrt(1.0 - c * dx)) / dx


def _h_t_inf(h, t, dx):
    """``function y = h_t(h,t,dx)``: hh_T = U hh_X + hh_XX."""
    return _U_inf(h, t, dx) * _D_uniform(h, dx, "interior") + _D2_uniform(h, dx)


def _h_exp_inf(h, ht, t, dt):
    """``function y = h_exp(h,ht,t,dt)``: explicit step + BCs hh(0)=T, hh(L)=0."""
    y = h + dt * ht
    y[0] = t
    y[-1] = 0.0
    return y


def _matrix_inf(dx, dt, u, N):
    """``function [A_0,A_plus,A_minus] = Matrix(dx,dt,u,N)`` -- implicit half."""
    A_0 = np.full(N, 1.0 + dt / dx ** 2)
    A_plus = np.full(N, -dt / dx * (1.0 / dx + u / 2.0) / 2.0)
    A_minus = np.full(N, -dt / dx * (1.0 / dx - u / 2.0) / 2.0)
    A_0[0] = 1.0
    A_0[N - 1] = 1.0
    A_plus[0] = 0.0
    A_plus[N - 1] = 0.0
    A_minus[0] = 0.0
    A_minus[N - 1] = 0.0
    return A_0, A_plus, A_minus


def _h_cnm_1(h, t, dt, dx):
    """``function [y,d] = h_cnm_1(h,t,dt,dx)``: first CN pass."""
    N = h.size
    ht = _h_t_inf(h, t, dx)
    hf = _h_exp_inf(h, ht, t, dt)
    d = _h_exp_inf(h, ht, t, dt / 2.0)
    d[0] = t + dt
    d[N - 1] = 0.0
    u = _U_inf(hf, t + dt, dx)
    A_0, A_plus, A_minus = _matrix_inf(dx, dt, u, N)
    return solve_tridiagonal(A_0, A_minus, A_plus, d), d


def _h_cnm_2(h, t, dt, dx):
    """``function [y,u] = h_cnm_2(h,t,dt,dx)``: second CN pass, re-lagging U."""
    N = h.size
    hf, d = _h_cnm_1(h, t, dt, dx)
    u = _U_inf(hf, t + dt, dx)
    A_0, A_plus, A_minus = _matrix_inf(dx, dt, u, N)
    return solve_tridiagonal(A_0, A_minus, A_plus, d), u


def simulate_reduced_infinite(T, dt, dx, L, Nframe=None, verbose=False):
    """``function [t_save,h_save,u_save] = simulate(T,dt,dx,L,Nframe)``.

    L here is the truncation length of the Xhat axis, NOT the sheet length --
    the sheet is semi-infinite. Returns a dict with ``T`` (exact frame times),
    ``U`` = dXc/dT, ``H`` = 1 + hh profiles and ``Xhat``.
    """
    N = int(round(L / dx))
    N_time = int(round(T / dt))
    if Nframe is None:
        Nframe = min(N_time, 10 ** 5)
    frame_step = int(round(N_time / Nframe))

    h_save = np.zeros((N, Nframe))
    u_save = np.zeros(Nframe)
    t_true = np.zeros(Nframe)

    h = np.zeros(N)
    t = 0.0
    u = 0.0
    t0 = time.time()
    nsaved = 0
    for it in range(0, N_time):
        if it % frame_step == 0:
            j = it // frame_step
            if j < Nframe:
                h_save[:, j] = h
                u_save[j] = u
                t_true[j] = t
                nsaved = j + 1
        h, u = _h_cnm_2(h, t, dt, dx)
        t = t + dt
        if verbose and it % max(1, N_time // 20) == 0:
            print(f"  it={it}/{N_time} T={t:.4f} U={u:.6f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)

    return {"T": t_true[:nsaved], "U": u_save[:nsaved],
            "H": 1.0 + h_save[:, :nsaved],
            "Xhat": np.linspace(0.0, L, N), "L": L, "dx": dx, "dt": dt}


# ===========================================================================
# Asymptotics the paper states, for verification
# ===========================================================================

def U_early(T):
    """Eq (5.10): dXc/dT ~ 2 sqrt(T/pi) as T -> 0. Same as Savva & Bush."""
    return 2.0 * np.sqrt(np.asarray(T) / math.pi)


def U_late_infinite(T):
    """Eq (7.11): dXc/dT ~ 1 - 1/(1+T)^2 for L = infinity, T >> 1."""
    return 1.0 - 1.0 / (1.0 + np.asarray(T)) ** 2


# ===========================================================================
# Verification driver
# ===========================================================================

def verify(quick: bool = False) -> int:
    """Check the ports against everything the paper states.

    GNU Octave is not installed on any host available to this project, so a
    direct numerical comparison against the original MATLAB is NOT possible.
    What is checked instead:
      (a) early-time law, eq (5.10), on both reduced solvers;
      (b) late-time law, eq (7.11), on the L = infinity solver;
      (c) tip thickness H(Xc,T) = 1+T, eq (3.10a) -- built into the reduced
          formulation, but a live check on the FULL solver, which does not know
          about it;
      (d) L >> T agreement between the finite-L and L = infinity solvers,
          which is a cross-check of two independently written schemes on the
          same continuous problem;
      (e) fig. 5 behaviour: the crossover time Tc at which U drops sharply is
          close to L for large L.
    """
    fails = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    print("(0) LAPACK banded backend vs the ported Thomas sweep")
    use_fast_solver(False)
    a = simulate_reduced_infinite(T=1.0, dt=1e-4, dx=0.02, L=10.0, Nframe=200)
    use_fast_solver(True)
    b = simulate_reduced_infinite(T=1.0, dt=1e-4, dx=0.02, L=10.0, Nframe=200)
    dsolv = float(np.max(np.abs(a["U"] - b["U"])))
    check("solver backends agree", dsolv < 1e-11, f"max |dU| = {dsolv:.2e}")
    use_fast_solver(True)

    print("(a) L = infinity reduced solver vs eq (5.10), short dedicated run")
    # Their save block is UNIFORM in time, so the early-time window has to be
    # reached by running a short case rather than by sampling a long one.
    NTe = 100000 if not quick else 20000
    re_ = simulate_reduced_infinite(T=0.1, dt=0.1 / NTe, dx=2.0 / 2000.0,
                                    L=2.0, Nframe=1000)
    m = (re_["T"] > 2e-3) & (re_["T"] < 0.05)
    rel_early = np.max(np.abs(re_["U"][m] / U_early(re_["T"][m]) - 1.0))
    check("early-time 2 sqrt(T/pi), 2e-3 < T < 0.05", rel_early < 0.05,
          f"max rel dev {rel_early:.3%}")

    print("(b) L = infinity reduced solver vs eq (7.11)")
    NT = 200000 if not quick else 40000
    Tend = 100.0
    r = simulate_reduced_infinite(T=Tend, dt=Tend / NT, dx=10.0 / 500.0,
                                  L=10.0, Nframe=2000)
    T, U = r["T"], r["U"]
    m = T > 20.0
    rel_late = np.max(np.abs(U[m] - U_late_infinite(T[m])))
    check("late-time 1 - 1/(1+T)^2, T > 20", rel_late < 5e-3,
          f"max abs dev {rel_late:.2e}, U(100) = {U[-1]:.6f}")
    # U(100) sits slightly ABOVE 1 at their own default Lhat = 10: the far-field
    # condition hh(Lhat)=0 is imposed where hh ~ T exp(-Lhat) ~ 4.5e-3, which
    # steepens the edge gradient. Enlarging Lhat removes it; `compare` uses 30.
    check("U -> 1 (Taylor-Culick)", abs(U[-1] - 1.0) < 0.02,
          f"U(T=100) = {U[-1]:.6f}, 1-1/(1+T)^2 = {U_late_infinite(100.0):.6f}"
          f" (their default Lhat=10 truncation overshoots by ~5e-3)")

    print("(d) finite-L solver at L >> T vs the L = infinity solver")
    # THE STATIC-FRAME SCHEME HAS A RESOLUTION CONSTRAINT THAT SCALES WITH L.
    # Its grid is uniform in X' = (L-X)/(L-Xc), so the physical spacing is
    # dX = (L-Xc)/N ~ L/N, while the diffusive layer at the free edge is
    # O(sqrt(T)) wide. Resolving it needs L/N << sqrt(T). At L = 400, N = 2000
    # that is dX = 0.2 against a layer of 0.07 at T = 0.005: the early-time law
    # is then missed by 70%, which is a property of THEIR scheme, not of the
    # port. The finite-L solver is therefore checked where it is resolved.
    NTf = 200000 if not quick else 40000
    rf = simulate_reduced_finite(L=60.0, N=6000, T=40.0, NT=NTf, Nframe=2000)
    Ui = np.interp(rf["T"], T, U)
    m = (rf["T"] > 2.0) & (rf["T"] < 30.0)     # T << L = 60: semi-infinite
    dev = np.max(np.abs(rf["U"][m] - Ui[m]))
    check("|U_finite(L=60,N=6000) - U_inf| for 2 < T < 30 (dX=0.01)",
          dev < 5e-3, f"max abs dev {dev:.2e}")
    rfe = simulate_reduced_finite(L=2.0, N=2000, T=0.1,
                                  NT=100000 if not quick else 20000,
                                  Nframe=1000)
    m = (rfe["T"] > 2e-3) & (rfe["T"] < 0.05)
    rel_early_f = np.max(np.abs(rfe["U"][m] / U_early(rfe["T"][m]) - 1.0))
    check("finite-L early-time 2 sqrt(T/pi) at L=2, dX=1e-3", rel_early_f < 0.06,
          f"max rel dev {rel_early_f:.3%}")

    print("(e) fig. 5: crossover time Tc ~ L")
    NTc = 400000 if not quick else 80000
    rc = simulate_reduced_finite(L=20.0, N=1000, T=40.0, NT=NTc, Nframe=4000)
    dU = np.gradient(rc["U"], rc["T"])
    Tc = rc["T"][int(np.argmin(dU))]
    check("Tc close to L for L = 20", abs(Tc - 20.0) / 20.0 < 0.25,
          f"Tc = {Tc:.2f} vs L = 20 (their fig. 5a dashed line Tc = L)")
    Umax = np.max(rc["U"])
    check("U reaches ~1 before the finite-sheet crossover (fig. 4)",
          0.9 < Umax < 1.05, f"max U = {Umax:.4f}")

    print("(c) FULL solver: tip thickness H(Xc,T) = 1 + T, eq (3.10a)")
    Oh = 10.0
    eps = 40.0
    NTfull = 60000 if quick else 200000
    tt, xx, vv, hh, t_true = simulate_full(
        *initial_condition_full(eps, 800), Oh=Oh, T=0.5 * (4 * Oh),
        NT=NTfull, Nframe=400)
    # Their h(0,.) is the CLAMPED end; the free edge is the last node. The
    # rim thickness the reduced model calls H(Xc,T) is the thickness just
    # behind the tip, which for a long sheet is h at the clamped end only once
    # the film has thickened uniformly -- so compare the MAXIMUM thickness.
    Hmax = hh.max(axis=0)
    m = t_true > 0.05
    dev_h = np.max(np.abs(Hmax[m] / (1.0 + t_true[m]) - 1.0))
    check("max H vs 1+T for T > 0.05 (Oh=10, L=eps/(4Oh)=1)", dev_h < 0.10,
          f"max rel dev {dev_h:.2%}")

    print()
    if fails:
        print(f"FAILED: {', '.join(fails)}")
        return 1
    print("all checks passed")
    return 0


# ===========================================================================
# Comparison against our pyoomph 1D solver
# ===========================================================================

def load_control1d(path: Path):
    """Read a ``control1d.txt`` written by ``validation/trouton1d_sweep.py``.

    Columns are positional: t, tstar, u_tip, ustar, h_tip, drift_h, drift_u
    and (since 2026-08-11) h_tip_exact. Returns (tstar, ustar, h_tip).
    """
    a = np.loadtxt(path, comments="#")
    return a[:, 1], a[:, 3], a[:, 4]


def _ahsan_inf_reference(Tmax: float, Tsplit: float, dx: float, dt: float,
                         Lhat: float, refine: int = 1):
    """U(T) from the L = infinity port, spliced from two resolutions.

    The tip-frame diffusive layer is O(sqrt(T)) wide, so a single dx that
    resolves T = 50 does not resolve T = 1e-3. Two runs: a fine short one over
    [0, Tsplit] and the production one over [0, Tmax], joined at Tsplit.
    ``refine`` halves dx and dt ``refine`` times, so calling with 1 and 2 gives
    the reference's own discretisation error.
    """
    f = 2.0 ** (refine - 1)
    dxe, dte = dx / (5.0 * f), dt / (10.0 * f)
    re_ = simulate_reduced_infinite(T=Tsplit, dt=dte, dx=dxe,
                                    L=max(4.0, 8.0 * math.sqrt(Tsplit)),
                                    Nframe=4000, verbose=True)
    rl = simulate_reduced_infinite(T=Tmax, dt=dt / f, dx=dx / f, L=Lhat,
                                   Nframe=8000, verbose=True)
    T = np.concatenate([re_["T"][re_["T"] < Tsplit], rl["T"][rl["T"] >= Tsplit]])
    U = np.concatenate([re_["U"][re_["T"] < Tsplit], rl["U"][rl["T"] >= Tsplit]])
    return T, U


def compare(control: Path, Tmax: float, NT: int, dx: float, Lhat: float,
            oh_full, csv_out: Path | None, Tsplit: float = 0.5,
            full_L: float = 20.0, full_N: int = 8000,
            full_Tmax: float = 5.0, full_NT: int = 200000) -> int:
    """Compare u*(t*) from our pyoomph solver against the Ahsan port.

    The primary reference is ``simulate_reduced_infinite``: our solver is
    semi-infinite and their ``L = infinity`` script is the same limit, so this
    is a like-for-like comparison of two independent discretisations of ONE
    continuous problem. Their finite-L script is run as well, at L >> Tmax, as
    a third independent curve, and optionally ``thin_film.m``, which is a
    DIFFERENT continuous problem (it retains the bulk capillary term).
    """
    tstar, ustar, _ = load_control1d(control)
    Tp = T_from_tstar(tstar)
    keep = Tp <= Tmax
    Tp, ustar = Tp[keep], ustar[keep]
    dt = Tmax / NT

    print(f"reference: Ahsan L=infinity, T<={Tmax}, dt={dt:g}, dx={dx:g}, "
          f"Lhat={Lhat:g}, Tsplit={Tsplit:g}")
    # The reference costs ~20 min and does not depend on which pyoomph run it
    # is being compared against, so cache it beside the output.
    tag = f"ahsan_ref_T{Tmax:g}_dt{dt:g}_dx{dx:g}_L{Lhat:g}_s{Tsplit:g}.npz"
    cache = (csv_out.parent if csv_out else Path(".")) / tag
    if cache.exists():
        z = np.load(cache)
        Tr, Ur, Tr2, Ur2 = z["Tr"], z["Ur"], z["Tr2"], z["Ur2"]
        print(f"  loaded cached reference {cache}")
    else:
        Tr, Ur = _ahsan_inf_reference(Tmax, Tsplit, dx, dt, Lhat, refine=1)
        print("reference resolution check: same at half dx and half dt")
        Tr2, Ur2 = _ahsan_inf_reference(Tmax, Tsplit, dx, dt, Lhat, refine=2)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Tr=Tr, Ur=Ur, Tr2=Tr2, Ur2=Ur2)
    Ui = np.interp(Tp, Tr, Ur)
    Ui2 = np.interp(Tp, Tr2, Ur2)
    m0 = Tp > 1e-3
    ref_err = float(np.max(np.abs(Ui[m0] - Ui2[m0])))
    print(f"  reference self-difference (1x vs 2x resolution): "
          f"max |dU| = {ref_err:.3e}   <-- the floor any verdict sits on")

    # Their static-frame grid is uniform in X' = (L-X)/(L-Xc), so the physical
    # spacing is ~L/N and the free-edge diffusive layer is O(sqrt(T)) wide:
    # this curve is only resolved for T >> (L/N)^2. At L = 5 Tmax, N = 8000 and
    # Tmax = 50 that is T >> 1e-3. L = 5 Tmax keeps the clamped end irrelevant
    # (their fig. 3c shows the finite sheet tracks the L = infinity curve until
    # T ~ L).
    L_fin = 5.0 * Tmax
    print(f"cross-check: Ahsan finite L={L_fin:g} (their OTHER scheme)")
    rf = simulate_reduced_finite(L=L_fin, N=8000, T=Tmax, NT=NT,
                                 Nframe=8000, verbose=True)
    Uf = np.interp(Tp, rf["T"], rf["U"])

    cols = [Tp, tstar[keep], ustar, Ui, Ui2, Uf, U_early(Tp),
            U_late_infinite(Tp)]
    hdr = ["T", "tstar", "u_star_pyoomph", "U_ahsan_Linf", "U_ahsan_Linf_2x",
           "U_ahsan_Lfin", "U_eq5p10", "U_eq7p11"]

    # thin_film.m: their FULL model -- the same reduced regime PLUS the bulk
    # capillary term and a finite-Oh tip region, with a hard-imposed sqrt tip
    # shape instead of a force closure. It is a DIFFERENT continuous problem,
    # and it is the direct test of whether our with_capillarity=False costs
    # anything.
    #
    # IT CANNOT REACH T = Tmax. Its grid must resolve the tip on the scale h0
    # while spanning eps = 4 Oh L nodes, so semi-infinite behaviour to T = 50
    # would need L >= 500 and eps >= 4000 at a spacing of order 0.01: 4e5 nodes.
    # It is run instead over its own, shorter window with L >> T_full.
    fullcols = {}
    for ohf in (oh_full or []):
        eps = 4.0 * ohf * full_L
        print(f"full model: thin_film.m Oh_paper={ohf:g}, epsilon={eps:g} "
              f"(L={full_L:g}), N={full_N}, dx={eps/full_N:.4g}, "
              f"T<={full_Tmax:g}")
        t, x, v, h, t_true = simulate_full(
            *initial_condition_full(eps, full_N), Oh=ohf,
            T=full_Tmax * 4.0 * ohf, NT=full_NT, Nframe=2000, verbose=True)
        fullcols[ohf] = (t_true, tip_velocity_full(t_true, v, x, h, ohf))
    for ohf, (tf, uf) in fullcols.items():
        col = np.where(Tp <= tf[-1], np.interp(Tp, tf, uf), np.nan)
        cols.append(col)
        hdr.append(f"U_full_Oh{ohf:g}")

    tab = np.column_stack(cols)
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(csv_out, tab, delimiter=",", header=",".join(hdr),
                   comments="")
        print(f"wrote {csv_out}")

    print()
    print(f"{'T':>10} {'t*':>9} " + " ".join(f"{c:>15}" for c in hdr[2:]))
    for row in tab[:: max(1, tab.shape[0] // 25)]:
        print(f"{row[0]:10.4g} {row[1]:9.4g} "
              + " ".join(f"{q:15.6f}" for q in row[2:]))

    def stats(label, ref):
        d = np.abs(ustar[m0] - ref[m0])
        print(f"{label}")
        print(f"   max |du*| = {d.max():.4e} at T = {Tp[m0][d.argmax()]:.4g}")
        print(f"   RMS |du*| = {math.sqrt(float(np.mean(d ** 2))):.4e}")
        print(f"   max rel   = {np.max(d / np.maximum(np.abs(ref[m0]), 1e-12)):.4e}")

    print()
    print(f"WINDOW: T in (1e-3, {Tmax:g}], {int(m0.sum())} points")
    stats("pyoomph vs Ahsan L=infinity (SAME equations):", Ui)
    stats("pyoomph vs Ahsan finite L=%g (same equations, clamped far end):"
          % L_fin, Uf)
    for j, ohf in enumerate(fullcols):
        col = tab[:, 8 + j]
        mf = m0 & np.isfinite(col) & (Tp > 1e-2)
        d = np.abs(ustar[mf] - col[mf])
        print(f"pyoomph vs Ahsan FULL model Oh_paper={ohf:g} (DIFFERENT "
              f"equations: bulk capillarity + finite-Oh tip), "
              f"T in (1e-2, {full_Tmax:g}]:")
        print(f"   max |du*| = {d.max():.4e} at T = {Tp[mf][d.argmax()]:.4g}")
        print(f"   RMS |du*| = {math.sqrt(float(np.mean(d ** 2))):.4e}")
    d_f = np.abs(Ui[m0] - Uf[m0])
    print(f"Ahsan L=inf vs Ahsan finite: max |dU| = {d_f.max():.4e}  "
          "(their own two schemes against each other)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fast", action="store_true",
                        help="LAPACK banded solve instead of the ported Thomas "
                             "loop; validated by `verify` check (0)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", parents=[common])
    p.add_argument("--quick", action="store_true")

    p = sub.add_parser("reduced-inf", help="thin_film_L=infty_Oh_large.m", parents=[common])
    p.add_argument("--T", type=float, default=100.0)
    p.add_argument("--NT", type=int, default=1000000)
    p.add_argument("--L", type=float, default=10.0)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--nframe", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("reduced-fin", help="thin_film_Oh_large.m", parents=[common])
    p.add_argument("--L", type=float, default=20.0)
    p.add_argument("--T", type=float, default=40.0)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--NT", type=int, default=200000)
    p.add_argument("--nframe", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("full", help="thin_film.m", parents=[common])
    p.add_argument("--oh", type=float, default=10.0, help="Oh_paper = 2*Oh_SB")
    p.add_argument("--epsilon", type=float, default=10.0, help="l0/h0")
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--T", type=float, default=None, help="default 0.5*(4*Oh)")
    p.add_argument("--NT", type=int, default=100000)
    p.add_argument("--nframe", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("compare", help="pyoomph control1d.txt vs this port", parents=[common])
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--Tmax", type=float, default=50.0)
    p.add_argument("--NT", type=int, default=1000000)
    p.add_argument("--dx", type=float, default=0.01)
    p.add_argument("--Tsplit", type=float, default=0.5)
    p.add_argument("--Lhat", type=float, default=40.0,
                   help="Xhat truncation for the L=infinity solver")
    p.add_argument("--oh-full", type=str, default=None,
                   help="comma list of Oh_paper (= 2 Oh_SB) for thin_film.m")
    p.add_argument("--full-L", type=float, default=20.0)
    p.add_argument("--full-N", type=int, default=8000)
    p.add_argument("--full-Tmax", type=float, default=5.0)
    p.add_argument("--full-NT", type=int, default=200000)
    p.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()
    if getattr(args, "fast", False):
        use_fast_solver(True)

    if args.cmd == "compare":
        ohs = ([float(q) for q in args.oh_full.split(",")]
               if args.oh_full else [])
        sys.exit(compare(args.control, args.Tmax, args.NT, args.dx,
                         args.Lhat, ohs, args.out, args.Tsplit,
                         args.full_L, args.full_N, args.full_Tmax,
                         args.full_NT))

    if args.cmd == "verify":
        sys.exit(verify(args.quick))

    if args.cmd == "reduced-inf":
        r = simulate_reduced_infinite(T=args.T, dt=args.T / args.NT,
                                      dx=args.L / args.N, L=args.L,
                                      Nframe=args.nframe, verbose=True)
        tab = np.column_stack([r["T"], tstar_from_T(r["T"]), r["U"],
                               U_early(r["T"]), U_late_infinite(r["T"])])
        hdr = "T,tstar,U,U_early_5p10,U_late_7p11"
    elif args.cmd == "reduced-fin":
        r = simulate_reduced_finite(L=args.L, N=args.N, T=args.T, NT=args.NT,
                                    Nframe=args.nframe, verbose=True)
        tab = np.column_stack([r["T"], tstar_from_T(r["T"]), r["U"], r["Xc"],
                               U_early(r["T"])])
        hdr = "T,tstar,U,Xc,U_early_5p10"
    else:
        T = args.T if args.T is not None else 0.5 * (4 * args.oh)
        x0, v0, h0 = initial_condition_full(args.epsilon, args.N)
        t, x, v, h, t_true = simulate_full(x0, v0, h0, args.oh, T, args.NT,
                                           args.nframe, verbose=True)
        U = tip_velocity_full(t_true, v, x, h, args.oh)
        tab = np.column_stack([t_true, tstar_from_T(t_true), U,
                               -v[-1, :], h.max(axis=0),
                               U_early(t_true)])
        hdr = "T,tstar,U_rim,U_lastnode,Hmax,U_early_5p10"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.out, tab, delimiter=",", header=hdr, comments="")
        print(f"wrote {args.out} ({tab.shape[0]} rows)")
    else:
        print(hdr)
        for row in tab[:: max(1, tab.shape[0] // 40)]:
            print(",".join(f"{q:.8g}" for q in row))


if __name__ == "__main__":
    main()
