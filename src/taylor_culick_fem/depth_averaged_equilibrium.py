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

Three formulations are provided. ``shoot``/``profile``/``branch`` integrate the radial
equations inward from the Lame far field and find the amplitude ``A`` for a given ``Ec``.
``log_profile`` integrates the logarithmic form, ``dT/dy = -(1 + k)`` with ``y = ln(r/rho)``,
``k = lambda_r/lambda_theta`` and ``T = N_rr/G``, from the far field to the free edge
``T = -2/Ec``. That trajectory does not depend on ``Ec``: ``universal_trajectory`` integrates
it once, in a state with an explicit right-hand side, and every ``Ec`` is then the special case
that stops it at ``T = -2/Ec``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp
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


# The universal trajectory -----------------------------------------------------------------
#
# In the variables of ``log_profile``, set zeta = -ln(lambda_z) = ln(b), so k = exp(zeta - 2y),
# and I(y) = int_0^y k dy'. The stretch equation exp(-2y) b^4 - T b^3 = 1 becomes
# T = k - exp(-3 zeta), and dT/dy = -(1 + k) gives T = -y - I and
#
#     dzeta/dy = exp(3 zeta) (k - 1) / (exp(4 zeta - 2y) + 3),    dI/dy = k,
#
# whose right-hand side needs no root solve and stays smooth for every y. The far-field
# series starts it at small y. A sheet with elastocapillary number Ec stops where
# T(Y) = -2/Ec, i.e. Y + I(Y) = 2/Ec, with Y = ln(R/r0).

#: Default numerics of the universal trajectory: series start, DOP853 tolerances and the
#: largest log radius ratio integrated (2/Ec + 20 <= 2020 covers Ec >= 1e-3).
FAR_SEED = 1e-3
UNIVERSAL_RTOL = 1e-12
UNIVERSAL_ATOL = 1e-14
LOG_RADIUS_MAX = 2020.0
#: Beyond this y, k <= exp(-2y) contributes less than exp(-80) to C = I(inf).
CONSTANT_TAIL = 40.0
#: Margin beyond 2/Ec the trajectory must cover, so the stopping remainder's tail is resolved.
REMAINDER_MARGIN = 20.0
#: Smallest y to which the far-field series continues a distance integration.
SERIES_FLOOR = 1e-300


def universal_series_start(far_seed: float = FAR_SEED) -> np.ndarray:
    """``(zeta, I)`` at ``y = far_seed`` from the far-field series of the trajectory.

    ``lambda_z = 1 + y^2/4 - y^3/16 - 53 y^4/768`` and
    ``I = y - y^2 + 7 y^3/12 - 37 y^4/192 + 133 y^5/3840``, truncated at ``O(y^5)`` and
    ``O(y^6)`` respectively.
    """
    y = _positive_finite("far_seed", far_seed)
    if y > 1e-2:
        raise ValueError("far_seed must be <=1e-2 for the far-field series")
    integral = y - y**2 + 7 * y**3 / 12 - 37 * y**4 / 192 + 133 * y**5 / 3840
    return np.array([_series_zeta(y), integral])


def _series_zeta(y: float) -> float:
    """``zeta = -ln(lambda_z)`` from the far-field series, for ``0 <= y <= 1e-2``."""
    return -math.log1p(y**2 / 4 - y**3 / 16 - 53 * y**4 / 768)


def _universal_rhs(y, state):
    zeta = state[0]
    k = math.exp(zeta - 2 * y)
    return [math.exp(3 * zeta) * (k - 1) / (math.exp(4 * zeta - 2 * y) + 3), k]


@dataclass(frozen=True)
class UniversalTrajectory:
    """The ``Ec``-independent depth-averaged trajectory on ``[far_seed, log_radius_max]``.

    ``y`` is ``ln(r/rho)``; ``tension`` is ``T = N_rr/G``, ``stretch_ratio`` is
    ``k = lambda_r/lambda_theta`` and ``thickness`` is ``lambda_z = h/h0``. Every method that
    takes ``Ec`` evaluates the sharp-edged equilibrium that stops the trajectory at
    ``T = -2/Ec``; lengths are on ``h0`` and the reference edge radius is ``r0``.
    """

    far_seed: float
    log_radius_max: float
    rtol: float
    atol: float
    max_step: float
    _solution: object = field(repr=False, compare=False)

    def _y(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < self.far_seed) or np.any(y > self.log_radius_max):
            raise ValueError("y lies outside the integrated trajectory")
        return y

    def zeta(self, y):
        """``zeta = -ln(lambda_z)``."""
        return self._solution.sol(self._y(y))[0]

    def stretch_ratio(self, y):
        """``k = lambda_r / lambda_theta = exp(zeta - 2y)``."""
        y = self._y(y)
        return np.exp(self._solution.sol(y)[0] - 2 * y)

    def tension(self, y):
        """``T = N_rr / G = -y - I(y)``."""
        y = self._y(y)
        return -y - self._solution.sol(y)[1]

    def thickness(self, y):
        """``lambda_z = h/h0``."""
        return np.exp(-self.zeta(y))

    @property
    def constant(self) -> float:
        """``C = int_0^inf k dy``, so that ``ln(R/r0) -> 2/Ec - C`` as ``Ec -> 0``."""
        return float(self._solution.sol(CONSTANT_TAIL)[1])

    def _check_ec(self, ec: float) -> float:
        ec = _positive_finite("Ec", ec)
        target = 2.0 / ec
        if target + REMAINDER_MARGIN > self.log_radius_max:
            raise ValueError(f"Ec={ec:g} needs log_radius_max >= 2/Ec + {REMAINDER_MARGIN:g}")
        if target <= 2 * self.far_seed:
            raise ValueError(f"Ec={ec:g} puts the free edge inside the far-field series start")
        return ec

    def arrest_log_radius(self, ec: float) -> float:
        """``Y = ln(R/r0)``: the root of ``Y + I(Y) = 2/Ec``, which lies in ``[2/Ec - C, 2/Ec]``."""
        target = 2.0 / self._check_ec(ec)
        lower = max(target - self.constant - 1e-9, self.far_seed)
        return brentq(lambda y: float(self.tension(y)) + target, lower, target,
                      xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)

    def log_remainder(self, log_radius: float) -> float:
        """``ln eps`` of the stopping remainder ``eps(Y) = int_Y^inf k dy = C - I(Y)``.

        Evaluated as a tail integral scaled by ``exp(-2Y)``, so it neither cancels nor
        underflows. ``Y`` must lie at least ``REMAINDER_MARGIN`` inside ``log_radius_max``, so
        the omitted tail is below ``exp(-2 REMAINDER_MARGIN)`` relative to the result.
        """
        Y = float(log_radius)
        if not self.far_seed <= Y <= self.log_radius_max - REMAINDER_MARGIN:
            raise ValueError(f"log_radius must lie in [far_seed, log_radius_max - {REMAINDER_MARGIN:g}]")
        end = min(Y + CONSTANT_TAIL, self.log_radius_max)
        integrand = lambda y: math.exp(float(self._solution.sol(y)[0]) - 2 * (y - Y))
        value, _ = quad(integrand, Y, end, epsabs=0.0, epsrel=1e-11, limit=200)
        return math.log(value) - 2 * Y

    def edge_thickness(self, ec: float) -> float:
        """``lambda_z`` at the free edge, ``h_e / h0``."""
        return float(self.thickness(self.arrest_log_radius(ec)))

    def thickness_at_distances(self, ec: float, distances, r0: float = 1.0) -> np.ndarray:
        """``h/h0`` at nonnegative deformed distances (on ``h0``) behind the free edge.

        With ``J(y) = int_y^Y k/(1-k) dy'``, ``r/R = exp(J)`` and the distance is
        ``R (exp(J) - 1)``. The scaled ``Jt = exp(Y) J`` is integrated inward from the edge,
        so neither ``R`` nor ``J`` need be representable; the distance is
        ``r0 Jt expm1(J)/J``. Below ``far_seed`` the far-field series continues the
        trajectory, integrated in ``ln y`` because ``k/(1-k) ~ 1/(2y)`` as ``y -> 0``.
        """
        ds = np.asarray(distances, dtype=float)
        if ds.ndim != 1 or not np.all(np.isfinite(ds)) or np.any(ds < 0):
            raise ValueError("distances must be a finite nonnegative one-dimensional sequence")
        r0 = _positive_finite("r0", r0)
        Y = self.arrest_log_radius(ec)
        result = np.full(ds.shape, float(self.thickness(Y)))
        with np.errstate(over="ignore"):
            scaled = ds / r0
        if not np.all(np.isfinite(scaled)):
            raise ValueError("distance / r0 is not representable as a float")
        targets = sorted(set(scaled[scaled > 0].tolist()))
        if not targets:
            return result
        found: dict[float, float] = {}

        def integrand(y, zeta):
            # dJt/dy, with 1 - k formed without cancellation as y -> 0.
            exponent = Y - 2 * y + zeta
            return -(math.exp(exponent) if exponent < 700 else math.inf) / -math.expm1(zeta - 2 * y)

        def distance(jt):
            j = jt * math.exp(-Y) if Y < 700 else 0.0
            return jt * (math.expm1(j) / j if j > 0 else 1.0)

        def advance(rhs, span, jt, to_y, zeta_of, max_step):
            pending = [target for target in targets if target not in found]
            events = []
            for target in pending:
                def event(t, state, target=target):
                    return distance(state[0]) - target
                event.direction = 1
                event.terminal = target == pending[-1]
                events.append(event)
            # Near the edge at small Ec the integrand underflows to zero; DOP853's error norm
            # is then 0/0 for the first steps, which only makes it retry with a smaller step.
            with np.errstate(invalid="ignore"):
                sol = solve_ivp(rhs, span, [jt], method="DOP853", rtol=self.rtol, atol=1e-30,
                                events=events, max_step=max_step)
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise RuntimeError(f"finite-distance integration failed: {sol.message}")
            for target, times in zip(pending, sol.t_events, strict=True):
                if len(times) == 1:
                    found[target] = math.exp(-zeta_of(to_y(float(times[0]))))
            return float(sol.y[0, -1])

        trajectory_zeta = lambda y: float(self._solution.sol(y)[0])
        jt = advance(lambda y, state: [integrand(y, trajectory_zeta(y))], (Y, self.far_seed), 0.0,
                     lambda y: y, trajectory_zeta, 0.05)
        if len(found) < len(targets):
            # Below the series start, in s = ln y: dJt/ds = y dJt/dy stays bounded as y -> 0.
            def series_rhs(s, state):
                y = math.exp(s)
                return [y * integrand(y, _series_zeta(y))]
            advance(series_rhs, (math.log(self.far_seed), math.log(SERIES_FLOOR)), jt, math.exp,
                    _series_zeta, 0.5)
        if len(found) < len(targets):
            raise RuntimeError("finite-distance integration ended before reaching a requested distance")
        for target, height in found.items():
            result[scaled == target] = height
        return result


@lru_cache(maxsize=8)
def universal_trajectory(far_seed: float = FAR_SEED, rtol: float = UNIVERSAL_RTOL,
                         atol: float = UNIVERSAL_ATOL, log_radius_max: float = LOG_RADIUS_MAX,
                         max_step: float | None = None) -> UniversalTrajectory:
    """Integrate the universal trajectory with DOP853 from its far-field series start.

    ``max_step`` defaults to 0.05 when ``log_radius_max <= 60`` and 0.25 otherwise. The result
    is cached per argument set.
    """
    far_seed = _positive_finite("far_seed", far_seed)
    rtol = _positive_finite("rtol", rtol)
    atol = _positive_finite("atol", atol)
    log_radius_max = _positive_finite("log_radius_max", log_radius_max)
    if log_radius_max < CONSTANT_TAIL + REMAINDER_MARGIN:
        raise ValueError(f"log_radius_max must be at least {CONSTANT_TAIL + REMAINDER_MARGIN:g}")
    if max_step is None:
        max_step = 0.05 if log_radius_max <= 60 else 0.25
    max_step = _positive_finite("max_step", max_step)
    sol = solve_ivp(_universal_rhs, (far_seed, log_radius_max), universal_series_start(far_seed),
                    method="DOP853", rtol=rtol, atol=atol, dense_output=True, max_step=max_step)
    _checked_solution(sol, log_radius_max)
    return UniversalTrajectory(far_seed, log_radius_max, rtol, atol, max_step, sol)


def arrest_log_radius_lower_bound(ec: float) -> float:
    """``Y*`` with ``Y* + (1 - exp(-2 Y*))/2 = 2/Ec``.

    Since ``lambda_z >= 1``, ``k <= exp(-2y)`` and ``I(Y) <= (1 - exp(-2Y))/2``, so
    ``ln(R/r0) >= Y*``.
    """
    target = 2.0 / _positive_finite("Ec", ec)
    return brentq(lambda y: y + 0.5 * (1 - math.exp(-2 * y)) - target, 0.0, target, xtol=1e-15)


def edge_thickness_from_log_radius(ec: float, log_radius: float) -> float:
    """Edge ``lambda_z``: the positive root of ``lambda^4 - (2/Ec) lambda - exp(-2Y) = 0``.

    This is ``T = k - lambda_z^3 = -2/Ec`` at the edge, an algebraic identity given ``Y``,
    solved in log space by the stretch root of ``log_profile`` (``b = 1/lambda_z``).
    """
    tension = -2.0 / _positive_finite("Ec", ec)
    return math.exp(-_log_column_stretch(_positive_finite("log_radius", log_radius), tension))


def scaled_thickness_expansion(ec, distance: float, constant: float, order: int = 2):
    """Small-``Ec`` expansion of ``Ec^(1/3) h(d)/h0`` at a fixed distance ``d`` (``r0 = h0``).

    With ``B = C - ln(2d) + ln(Ec)/3``: ``1 + (Ec/6) B`` at first order, less
    ``(Ec^2/36)(B^2 + B + 1)`` at second order.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    ec = np.asarray(ec, dtype=float)
    b = constant - math.log(2 * _positive_finite("distance", distance)) + np.log(ec) / 3
    result = 1 + ec / 6 * b
    if order == 2:
        result = result - ec**2 / 36 * (b**2 + b + 1)
    return result
