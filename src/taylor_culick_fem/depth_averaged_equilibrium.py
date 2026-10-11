"""Static depth-averaged axisymmetric theory of an arrested hole in an elastic sheet.

Each material column at reference radius ``rho`` keeps its own thickness and is stretched
radially by ``lambda_r = dr/drho``, azimuthally by ``lambda_theta = r/rho`` and through the
thickness by ``lambda_z = 1 / (lambda_r lambda_theta)`` (incompressibility). With an
incompressible neo-Hookean sheet of modulus ``G`` in plane stress, the membrane tensions per
unit deformed length are

    N_rr = G h0 lambda_z (lambda_r^2 - lambda_z^2),
    N_tt = G h0 lambda_z (lambda_theta^2 - lambda_z^2),

radial equilibrium is ``d(r N_rr)/dr = N_tt``, and the free edge at ``rho = r0`` carries
the surface tension of both faces, ``N_rr(r0) = -2 sigma``. Far away the sheet relaxes to the
linear Lame field ``r = rho + A/rho``. Lengths are on ``h0``, tensions on ``sigma``, so
``G = Ec``.

The equations are homogeneous in ``G``: for a far-field amplitude ``A`` the deformation is
fixed and ``Ec = -2 / (N_rr(r0) / G)``. They also have no length scale besides ``r0``, so the
hole radius scales exactly with the edge radius.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def radial_stretch(c: float, tension: float) -> float:
    """``lambda_r`` such that ``c (lambda_r - c^2 / lambda_r^3) = tension`` (unit modulus).

    ``c = rho / r = 1 / lambda_theta``; the left side increases monotonically in
    ``lambda_r``, so the root is unique.
    """
    f = lambda a: c * (a - c**2 / a**3) - tension
    lo, hi = 1e-8, 1.0
    while f(hi) < 0:
        hi *= 2
    while f(lo) > 0:
        lo /= 2
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-13)


def _rhs(rho, y):
    r, s = y
    c = rho / r
    a = radial_stretch(c, s / r)
    lz = c / a
    return [a, lz * ((r / rho) ** 2 - lz**2) * a]


def shoot(amplitude: float, r0: float = 1.0, dense: bool = False):
    """Integrate inward from the Lame far field of amplitude ``A`` to the edge ``r0``.

    The state is ``(r, r N_rr / G)``. Returns the ``solve_ivp`` solution.
    """
    rho_max = max(1e4 * r0, 100 * np.sqrt(amplitude))
    r = rho_max + amplitude / rho_max
    a = 1 - amplitude / rho_max**2
    c = rho_max / r
    lz = c / a
    s = r * lz * (a**2 - lz**2)
    return solve_ivp(_rhs, [rho_max, r0], [r, s], method="LSODA", rtol=1e-11, atol=1e-13,
                     dense_output=dense)


def elastocapillary_number(amplitude: float, r0: float = 1.0) -> float:
    """``Ec`` of the equilibrium with far-field amplitude ``A``."""
    r1, s1 = shoot(amplitude, r0).y[:, -1]
    tension = s1 / r1
    if tension >= 0:
        raise ValueError("the edge is not in compression for this amplitude")
    return -2.0 / tension


def profile(ec: float, r0: float = 1.0, samples: int = 4000) -> dict[str, np.ndarray | float]:
    """The equilibrium at ``Ec``: reference radius, deformed radius and stretches.

    Returns ``rho``, ``r``, ``lambda_r``, ``lambda_z`` (thickness ratio) and the hole radius
    ``R`` and amplitude ``A``.
    """
    f = lambda la: np.log(elastocapillary_number(np.exp(la), r0)) - np.log(ec)
    log_a = brentq(f, np.log(1e-6), np.log(1e18), xtol=1e-12)
    sol = shoot(np.exp(log_a), r0, dense=True)
    rho = np.unique(np.concatenate([np.geomspace(r0, sol.t[0], samples), [r0]]))
    r, s = sol.sol(rho)
    c = rho / r
    a = np.array([radial_stretch(ci, si / ri) for ci, si, ri in zip(c, s, r, strict=True)])
    return {"rho": rho, "r": r, "lambda_r": a, "lambda_z": c / a, "R": float(r[0]),
            "A": float(np.exp(log_a))}


def thickness_at_distance(prof: dict, distance: float) -> float:
    """Thickness ratio ``lambda_z`` at a deformed distance from the edge."""
    return float(np.interp(prof["R"] + distance, prof["r"], prof["lambda_z"]))


def branch(amplitudes, r0: float = 1.0) -> np.ndarray:
    """Rows ``(Ec, R, h_edge, lambda_r_edge, lambda_theta_edge, A)`` for the given amplitudes."""
    rows = []
    for amplitude in amplitudes:
        r1, s1 = shoot(amplitude, r0).y[:, -1]
        tension = s1 / r1
        if tension >= 0:
            continue
        c1 = r0 / r1
        a1 = radial_stretch(c1, tension)
        rows.append((-2.0 / tension, r1, c1 / a1, a1, r1 / r0, amplitude))
    return np.asarray(rows)
