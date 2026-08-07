"""Independent 1D Trouton (slender-sheet) solve of planar retraction.

WHY THIS EXISTS. Our 2D free-surface FEM differs from the digitised Savva & Bush
curve by 1.24% at ``Oh_SB = 1``. Three explanations are possible and a direct
FEM-vs-figure comparison cannot separate them:

  1. the digitisation is wrong;
  2. their lubrication model genuinely differs from full Navier-Stokes;
  3. our FEM is wrong.

Solving *their* equations independently splits the question:

  * this solver vs the digitised curves -> tests (1) and our reading of their
    model;
  * our 2D FEM vs this solver           -> tests (2) against (3).

WHAT THIS IS NOT. This is *our* implementation of *their* model, so it is not an
independent comparator in the validation sense -- a shared misreading of the
physics would satisfy both. It discriminates between the three explanations; it
is not extra validation evidence.

THE MODEL (Savva & Bush 2009 eqs. 2.1-2.2, after Brenner & Gueyffier 1999), for
a planar sheet of FULL thickness ``h(x,t)`` retracting along x at speed
``u(x,t)``:

    u_t + u u_x = (4 mu / (rho h)) (h u_x)_x + (gamma / rho) kappa_x
    h_t + (h u)_x = 0
    kappa = (1/2) h_xx / [1 + (1/4) h_x^2]^{3/2}

The 1/2 and 1/4 arise because the interface is at ``z = +/- h/2`` while ``h`` is
the full thickness; kappa is the curvature of the *surface*, not of h. Getting
that wrong changes the capillary driving by a factor of two.

FORMULATION -- why a fixed grid in x fails. The tip is a *moving free boundary*
where ``h -> 0``, which makes the viscous coefficient ``4 mu / (rho h)``
degenerate: Ahsan et al. (arXiv:2507.04603) identify the reduced problem as a
Stefan problem. A naive fixed-grid solve with a resolved semicircular cap does
not retract at all, because the cap has a vertical tangent and the one-sided
curvature closure at the end node is wrong by a factor of four.

Instead we use the LAGRANGIAN MASS COORDINATE

    m(x,t) = int_{x0(t)}^{x} h dx'

which is a material label, so the domain ``m in [0, M]`` is FIXED and the moving
tip sits at ``m = 0``. With ``d/dx = h d/dm``:

    h_t|_m = -h^2 u_m
    u_t|_m = 4 mu d/dm (h^2 u_m) + h kappa_m
    x_t|_m = u

Savva & Bush close the tip not by resolving the cap but with the jump condition
(their eq. 2.28) ``[h u_x] = -1/4`` in their scaling. Dimensionally that is a
prescribed axial force at the free end -- the cap's capillary pull, ``2 gamma``
against the Trouton stress resultant ``4 mu h u_x``:

    h u_x |_tip  =  -gamma / (2 mu)        i.e.   h^2 u_m|_{m=0} = -gamma/(2 mu)

which is exactly the natural (flux) boundary condition for the conservative
viscous operator above. Verified consistent with their non-dimensionalisation:
substituting ``x = Oh H x*``, ``h = H h*``, ``u = u_c u*`` recovers ``-1/4``.

The far end ``m = M`` is the undisturbed sheet: ``u = 0``, ``h = H``. That is
valid only while the disturbance has not reached it, which the driver checks.

INITIAL CONDITION. A flat sheet at rest, ruptured at t = 0. In mass coordinates
that is simply ``h = 1``, ``u = 0``, ``x = m``. The early-time response is then
the classical impulsively-started diffusion problem, for which Savva & Bush give
``u* = sqrt(t*/pi)`` -- so reproducing that curve is a built-in correctness
check on this solver, not merely a comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

V_TC = math.sqrt(2.0)


def _grad(f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """d/dm on a possibly non-uniform grid (numpy handles the spacing)."""
    return np.gradient(f, m, edge_order=2)


def _rhs(_t: float, y: np.ndarray, n: int, m: np.ndarray, mu: float, tip_flux: float):
    h = np.maximum(y[:n], 1e-9)
    u = y[n : 2 * n]

    # --- surface curvature, via d/dx = h d/dm -------------------------------
    h_m = _grad(h, m)
    h_mm = _grad(h_m, m)
    h_x = h * h_m
    h_xx = h * (h_m**2 + h * h_mm)
    kappa = 0.5 * h_xx / (1.0 + 0.25 * h_x**2) ** 1.5
    kappa_m = _grad(kappa, m)

    # --- viscous term, finite volume on the non-uniform grid ----------------
    # F_{i+1/2} = h^2 u_m on cell faces; the tip face carries the prescribed cap
    # pull instead of a computed gradient. Dividing by the CELL WIDTH (not a
    # uniform dm) keeps the operator conservative when the mesh is graded.
    h_mid = 0.5 * (h[1:] + h[:-1])
    F = h_mid**2 * (u[1:] - u[:-1]) / (m[1:] - m[:-1])
    F_full = np.empty(n + 1)
    F_full[1:-1] = F
    F_full[0] = tip_flux  # <- Savva & Bush eq. 2.28, the massless-cap closure
    F_full[-1] = F[-1]

    faces = np.empty(n + 1)
    faces[1:-1] = 0.5 * (m[1:] + m[:-1])
    faces[0], faces[-1] = m[0], m[-1]
    cell = np.diff(faces)
    cell[0] = faces[1] - faces[0]

    du = 4.0 * mu * (F_full[1:] - F_full[:-1]) / cell + h * kappa_m
    dh = -(h**2) * _grad(u, m)

    # Undisturbed far field.
    du[-1] = 0.0
    dh[-1] = 0.0
    return np.concatenate([dh, du, u])


def run(oh_sb: float, mass: float, n: int, tstar_end: float, n_out: int = 600,
        rtol: float = 1e-9, atol: float = 1e-11, grade: float = 2.0):
    """Integrate to ``t* = tstar_end``; return (t, t*, u*_tip, far-field drift)."""
    mu = oh_sb * math.sqrt(2.0)
    tau_vis = mu / 2.0
    t_end = tstar_end * tau_vis

    # GRADED mesh clustered at the tip. The tip carries a boundary layer whose
    # mass-coordinate extent shrinks as the run starts, so a uniform grid wastes
    # points in the undisturbed far field and under-resolves exactly where the
    # closure acts. grade = 1 recovers the uniform mesh.
    xi = np.linspace(0.0, 1.0, n)
    m = mass * xi**grade
    tip_flux = -1.0 / (2.0 * mu)  # gamma = 1

    # Jacobian sparsity. Without this, the solver forms DENSE finite-difference
    # Jacobians on 3n unknowns -- 3n RHS evaluations each -- which is what made
    # a graded mesh appear to hang: grading shrinks the tip cell by ~10^3, the
    # viscous operator's stiffness rises as 1/dm^2, and the implicit solver then
    # needs Jacobians constantly. The coupling is banded (curvature reaches two
    # neighbours, so a half-bandwidth of 3 is safe), so this is cheap to supply.
    band = 3
    S = lil_matrix((3 * n, 3 * n), dtype=int)
    for blk_r in range(3):
        for blk_c in range(3):
            for i in range(n):
                lo, hi = max(0, i - band), min(n, i + band + 1)
                S[blk_r * n + i, blk_c * n + lo : blk_c * n + hi] = 1
    S = S.tocsr()

    y0 = np.concatenate([np.ones(n), np.zeros(n), m.copy()])
    sol = solve_ivp(_rhs, (0.0, t_end), y0, t_eval=np.linspace(0.0, t_end, n_out),
                    method="BDF", args=(n, m, mu, tip_flux), rtol=rtol, atol=atol,
                    jac_sparsity=S)
    if not sol.success:
        raise RuntimeError(sol.message)

    u_tip = sol.y[n, :]  # tip speed; positive for retraction with the eq-2.28 flux sign
    drift = float(np.max(np.abs(sol.y[n - 1, :] - 1.0)))
    return sol.t, sol.t / tau_vis, u_tip / V_TC, drift, mu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oh", type=float, nargs="*", default=[0.5, 1.0, 5.0, 10.0])
    ap.add_argument("--mass", type=float, default=None,
                    help="sheet mass per unit width; default max(30, 30*mu)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--grade", type=float, default=2.0,
                    help="tip mesh clustering exponent; 1 = uniform")
    ap.add_argument("--tstar-end", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for oh in args.oh:
        mu = oh * math.sqrt(2.0)
        mass = args.mass if args.mass else max(30.0, 30.0 * mu)
        t, tstar, ustar, drift, mu = run(oh, mass, args.n, args.tstar_end, grade=args.grade)

        # Built-in check: the early-time response must follow sqrt(t*/pi).
        # Compare only where the asymptote is actually valid. Savva & Bush state
        # it holds to t* ~ 0.4, and the true solution falls progressively below
        # it after that -- measuring out to 0.4 reports THEIR breakdown as OUR
        # error (0.0093 vs 0.00056 on [0.005, 0.05]).
        e = (tstar > 0.005) & (tstar < 0.05)
        rms = float(np.sqrt(np.mean((ustar[e] - np.sqrt(tstar[e] / np.pi)) ** 2))) \
            if e.sum() > 5 else float("nan")
        flag = "   FAR FIELD DISTURBED, increase --mass" if drift > 1e-3 else ""
        print(f"Oh_SB={oh:<5g} mu={mu:<7.4f} M={mass:<6.1f} n={args.n}  "
              f"u*(t*={args.tstar_end:g})={ustar[-1]:.5f}  "
              f"early-time RMS vs sqrt(t*/pi)={rms:.4f}  drift={drift:.1e}{flag}",
              flush=True)

        f = args.out / f"trouton1d_Oh{oh:g}.csv"
        np.savetxt(f, np.column_stack([tstar, ustar]), delimiter=",",
                   header=("Independent 1D Trouton solve of Savva & Bush eqs 2.1-2.2,\n"
                           "Lagrangian mass coordinate, tip closed by their eq. 2.28.\n"
                           f"Oh_SB={oh:g} mu={mu:.6f} mass={mass:g} n={args.n}\n"
                           f"early-time RMS vs sqrt(t*/pi) = {rms:.5f}; far-field drift = {drift:.2e}\n"
                           "OUR implementation of THEIR model: a discriminator between\n"
                           "digitisation error and model difference, NOT independent\n"
                           "validation evidence.\n"
                           "tstar,ustar"), comments="# ")
    print(f"\nwrote trouton1d_Oh*.csv to {args.out}")


if __name__ == "__main__":
    main()
