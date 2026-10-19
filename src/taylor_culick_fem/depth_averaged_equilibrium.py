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

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _checked_solution(sol, endpoint: float):
    if not sol.success or not np.all(np.isfinite(sol.y)) or sol.t[-1] != endpoint:
        raise RuntimeError(f"equilibrium integration did not reach its endpoint: {sol.message}")
    return sol


def _log_column_stretch(y: float, tension: float) -> float:
    """Return log(b), b=exp(2y) k=1/lambda_z, on the compressive branch.

    The monotone equation exp(-2y)b**4 + (-T)b**3 = 1 is evaluated
    with logaddexp, so neither very small k nor large stretches overflow.
    """
    if not math.isfinite(y) or not math.isfinite(tension) or y <= 0 or tension >= 0:
        raise ValueError("the logarithmic compressive branch requires y>0 and T<0")
    log_t = math.log(-tension)
    # At the analytic upper bound one term equals 1; the other can underflow,
    # leaving a slightly negative rounded residual. Move outward so the
    # controlling term is at least 2, preserving a strict positive bracket.
    upper = min(y / 2 + math.log(2) / 4, (-log_t + math.log(2)) / 3)
    residual = lambda z: float(np.logaddexp(4 * z - 2 * y, log_t + 3 * z))
    lower = upper - 1
    while residual(lower) >= 0:
        lower -= max(1.0, upper - lower)
    return brentq(residual, lower, upper, xtol=1e-14, rtol=1e-14)


@dataclass(frozen=True)
class LogEquilibriumProfile:
    """Solved sharp-edge membrane equilibrium; no FEM rim-height calibration.

    ``log_radius_ratio`` is ln(R/r0), and ``edge_thickness`` is lambda_z.
    Use ``thickness_at_distances`` for finite deformed distances from the edge.
    No evaluation requires constructing R, which may exceed floating-point range.
    """

    ec: float
    r0: float
    log_radius_ratio: float
    edge_thickness: float
    diagnostics: dict
    _solution: object = field(repr=False, compare=False)
    _seed: float = field(repr=False, compare=False)
    _rtol: float = field(repr=False, compare=False)
    _atol: float = field(repr=False, compare=False)

    def thickness_at_distances(self, distances) -> np.ndarray:
        """Evaluate membrane thickness at nonnegative finite distances in h0 units.

        With q=y_edge-y and x=ln(rho), integrate x_q=1/(1-k) and
        distance_q=exp(x-y)b/(1-k). This integrates distance directly, avoiding
        subtraction of two radii that can be exponentially larger than the offset.
        """
        ds = np.asarray(distances, dtype=float)
        if ds.ndim != 1 or not np.all(np.isfinite(ds)) or np.any(ds < 0):
            raise ValueError("distances must be a finite nonnegative one-dimensional sequence")
        result = np.full(ds.shape, self.edge_thickness)
        targets = sorted(set(ds[ds > 0].tolist()))
        if not targets:
            return result

        def rhs(q, state):
            y = self.log_radius_ratio - q
            tension = float(self._solution.sol(y)[0])
            log_b = _log_column_stretch(y, tension)
            k = math.exp(log_b - 2 * y)
            if not 0 <= k < 1:
                raise RuntimeError("membrane trajectory left the outward compressive branch")
            exponent = state[0] - y + log_b
            if exponent > math.log(np.finfo(float).max):
                raise RuntimeError("finite-distance integration overflowed before reaching its target")
            return [1 / (1 - k), math.exp(exponent) / (1 - k)]

        events = []
        for target in targets:
            def event(q, state, target=target):
                return state[1] - target
            event.direction = 1
            event.terminal = target == targets[-1]
            events.append(event)
        stop = self.log_radius_ratio - self._seed
        sol = solve_ivp(rhs, (0, stop), [math.log(self.r0), 0.0], method="DOP853",
                        rtol=self._rtol, atol=self._atol, events=events,
                        max_step=min(0.5, stop))
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise RuntimeError(f"finite-distance integration failed: {sol.message}")
        for target, times in zip(targets, sol.t_events, strict=True):
            if len(times) != 1:
                raise RuntimeError("finite-distance integration ended before reaching a requested distance")
            y = self.log_radius_ratio - float(times[0])
            log_b = _log_column_stretch(y, float(self._solution.sol(y)[0]))
            result[ds == target] = math.exp(-log_b)
        return result


def log_profile(ec: float, r0: float = 1.0, *, rtol: float = 1e-10,
                atol: float = 1e-12, far_seed: float = 1e-5) -> LogEquilibriumProfile:
    """Solve the universal logarithmic membrane trajectory at positive finite Ec.

    Set x=ln(rho), y=ln(r/rho), k=(dr/drho)/(r/rho), T=N_rr/G.
    Incompressibility and equilibrium give k-exp(-6y)/k**3=T,
    y_x=k-1 and T_x=1-k**2. Hence dT/dy=-(1+k), independent
    of x and of the far-field amplitude. Translation in x places the edge at
    rho=r0; its y is exactly ln(R/r0). The Lamé branch starts at y=0,T=0.
    Its local expansion is T=-2y+y**2-7y**3/12+O(y**4).

    ``far_seed`` truncates that expansion; vary it separately from ODE tolerances
    when checking accuracy. The free-edge event is T=-2/Ec. Since 0<k<1,
    dT/dy lies between -2 and -1, supplying a finite integration bound.
    """
    ec = _positive_finite("Ec", ec)
    r0 = _positive_finite("r0", r0)
    rtol = _positive_finite("rtol", rtol)
    atol = _positive_finite("atol", atol)
    seed = _positive_finite("far_seed", far_seed)
    if seed > 1e-3:
        raise ValueError("far_seed must be <=1e-3 for the far-field expansion")
    target = -2 / ec
    initial = -2 * seed + seed**2 - 7 * seed**3 / 12
    if not math.isfinite(target) or target >= initial:
        raise ValueError("Ec puts the free edge outside the resolved far-field interval")

    def rhs(y, state):
        log_b = _log_column_stretch(y, float(state[0]))
        k = math.exp(log_b - 2 * y)
        if not 0 <= k < 1:
            raise RuntimeError("membrane trajectory left the compressive branch")
        return [-1 - k]

    def edge(y, state):
        return state[0] - target
    edge.direction = -1
    edge.terminal = True
    bound = seed + 2 / ec
    sol = solve_ivp(rhs, (seed, bound), [initial], method="DOP853", rtol=rtol,
                    atol=atol, events=edge, dense_output=True, max_step=0.5)
    if not sol.success or not np.all(np.isfinite(sol.y)) or len(sol.t_events[0]) != 1:
        raise RuntimeError(f"logarithmic equilibrium did not reach its free edge: {sol.message}")
    y_edge = float(sol.t_events[0][0])
    tension = float(sol.y_events[0][0, 0])
    log_b = _log_column_stretch(y_edge, tension)
    root_residual = float(np.logaddexp(4 * log_b - 2 * y_edge,
                                      math.log(-tension) + 3 * log_b))
    diagnostics = {"integration_success": bool(sol.success), "edge_reached": True,
                   "edge_tension_residual": tension - target,
                   "stretch_log_equation_residual": root_residual,
                   "far_seed": seed, "far_seed_tension": initial,
                   "rtol": rtol, "atol": atol, "nfev": sol.nfev}
    return LogEquilibriumProfile(ec, r0, y_edge, math.exp(-log_b), diagnostics,
                                 sol, seed, rtol, atol)


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
    amplitude = _positive_finite("amplitude", amplitude)
    r0 = _positive_finite("r0", r0)
    rho_max = max(1e4 * r0, 100 * np.sqrt(amplitude))
    r = rho_max + amplitude / rho_max
    a = 1 - amplitude / rho_max**2
    c = rho_max / r
    lz = c / a
    s = r * lz * (a**2 - lz**2)
    sol = solve_ivp(_rhs, [rho_max, r0], [r, s], method="LSODA", rtol=1e-11, atol=1e-13,
                    dense_output=dense)
    return _checked_solution(sol, r0)


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
    ec = _positive_finite("Ec", ec)
    r0 = _positive_finite("r0", r0)
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer >=2")
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
