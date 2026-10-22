"""Thin-film far field for the co-moving viscoelastic window.

Far from the rim the sheet is flat and thin and moves in plane, so the 2D
Oldroyd-B (``lambda = inf``) solution there is the axisymmetric Trouton
membrane of ``simulationCases/trouton1d_ale_axi.py`` with a neo-Hookean
stress added (plane stress, ``rho = h0 = 1``, ``mu = sqrt(2) Oh``, ``G = Ec``):

    h_t + (1/r)(r h u)_r = 0
    h (u_t + u u_r) = (r N_rr)_r / r - N_thth / r
    N_rr   = 2 mu h (2 u_r + u/r) + G h (lam_r^2  - lam_z^2)
    N_thth = 2 mu h (u_r + 2 u/r) + G h (lam_th^2 - lam_z^2)
    Y_t + u Y_r = 0

with ``Y`` the reference (initial) radius of the material at ``r``,
``lam_th = r/Y``, ``lam_z = h`` and ``lam_r = 1/(lam_th lam_z)``.  Uniform
surface tension exerts an isotropic membrane force with no divergence on a flat
sheet, so it does not appear.  Linearised, the model is a Kelvin--Voigt
cylindrical dilatational wave of speed ``2 sqrt(G)``: the elastic far field is
an outgoing pulse, not a quasi-static ``c/r`` field.

The domain ``[a(t), b]`` has a moving inner end, driven with the 2D velocity
inside the window, and a fixed outer end at rest far enough away that the pulse
never reaches it.  Nodes sit at fixed fractions of the domain, so time
derivatives carry the ALE velocity ``w = (1 - xi) a'(t)``.  The inner end moves
with the window, faster than the film, so it is an outflow for ``h`` and ``Y``
and needs no condition on them.

By default the fractions are uniform (``xi = linspace(0, 1, nodes)``).  A very
long membrane can instead take graded fractions (``node_fractions``, usually
from :func:`graded_fractions`): uniform spacing next to the window, then
geometric growth towards the far end.  The graded path discretises the same
equations with the same structure on the non-uniform grid (staggered fluxes for
``(r N_rr)_r`` with local spacings, a second-order nodal gradient for the hoop
stress, second-order one-sided upwinding for advection); the uniform path is a
separate code path and is unchanged by it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import BDF
from scipy.optimize import brentq
from scipy.sparse import bmat, diags


def _geometric_sum(growth: float, cells: int) -> float:
    """``sum_{k=1..cells} growth**k`` without cancellation near ``growth = 1``."""

    if growth == 1.0:
        return float(cells)
    x = math.log(growth)
    return growth * math.expm1(cells * x) / math.expm1(x)


def graded_fractions(
    inner: float, outer: float, spacing: float, near_length: float, growth: float
) -> np.ndarray:
    """Node fractions for a membrane on ``[inner, outer]`` that coarsens outwards.

    The first ``m = ceil(near_length / spacing)`` cells beyond ``inner`` have the
    uniform width ``spacing``.  The remaining length ``S`` is covered by ``n``
    cells of width ``spacing * g**k`` (``k = 1..n``), so the cell width grows by
    the factor ``g`` from one cell to the next, starting from the last uniform
    cell.  ``n`` is the fewest cells that reach ``outer`` at ``g = growth``, and
    ``g`` in ``(1, growth]`` is then solved so that they end exactly at
    ``outer``, so every neighbour ratio equals ``g``.  For a long tail ``g`` is
    close to ``growth``.  Two degenerate cases stay (nearly) uniform: if the
    uniform part leaves less than one cell, the whole domain is uniform with
    ``ceil((outer - inner) / spacing) + 1`` nodes (the layout of
    :meth:`FarFieldMembrane.extended`); if the tail is too short for any
    ``g > 1`` (``n`` cells of ``spacing`` already overshoot), the tail is
    uniform with ``round(S / spacing)`` cells.

    Returns the fractions ``(r - inner) / (outer - inner)``: strictly increasing,
    first exactly 0 and last exactly 1, ready for ``node_fractions``.
    """

    inner, outer = float(inner), float(outer)
    spacing, near_length, growth = float(spacing), float(near_length), float(growth)
    if not (math.isfinite(inner) and math.isfinite(outer) and outer > inner):
        raise ValueError("graded fractions require a finite domain with inner < outer")
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError("graded fractions require a positive spacing")
    if not (math.isfinite(near_length) and near_length >= 0):
        raise ValueError("graded fractions require a non-negative near length")
    if not (math.isfinite(growth) and growth > 1):
        raise ValueError("graded fractions require a growth factor above 1")
    length = outer - inner
    uniform_cells = max(0, math.ceil(near_length / spacing - 1e-9))
    tail = length - uniform_cells * spacing
    if tail < spacing:
        return np.linspace(0.0, 1.0, math.ceil(length / spacing) + 1)
    # Fewest cells reaching the end: closed form, then guarded against round-off.
    cells = max(1, math.ceil(
        math.log1p(tail * (growth - 1.0) / (spacing * growth)) / math.log(growth)
    ))
    while cells > 1 and spacing * _geometric_sum(growth, cells - 1) >= tail:
        cells -= 1
    while spacing * _geometric_sum(growth, cells) < tail:
        cells += 1
    if cells * spacing >= tail:
        # No growth factor > 1 fits: a short tail, made uniform instead.
        cells = max(1, round(tail / spacing))
        widths = np.full(cells, tail / cells)
    else:
        target = tail / spacing
        g = brentq(lambda q: _geometric_sum(q, cells) - target, 1.0, growth,
                   xtol=1e-15, rtol=4 * np.finfo(float).eps)
        widths = spacing * g ** np.arange(1, cells + 1)
    positions = np.concatenate([
        spacing * np.arange(uniform_cells + 1),
        uniform_cells * spacing + np.cumsum(widths),
    ])
    fractions = positions / length
    fractions[-1] = 1.0
    if not np.all(np.diff(fractions) > 0):
        raise ValueError("graded fractions are not strictly increasing")
    return fractions


@dataclass(frozen=True)
class EdgeState:
    """Film state at one radius: velocity, thickness and principal stretches."""

    velocity: float
    thickness: float
    stretch_r: float
    stretch_theta: float

    def log_conformation(self) -> tuple[float, float, float]:
        """Return ``log C`` for (r, z, theta): ``C = diag(lam_r^2, lam_z^2, lam_th^2)``."""

        return (
            2.0 * math.log(self.stretch_r),
            2.0 * math.log(self.thickness),
            2.0 * math.log(self.stretch_theta),
        )


class FarFieldMembrane:
    """Moving-inner-end solver for the far-field membrane.

    Give either ``nodes`` (uniform fractions, the original layout) or
    ``node_fractions`` (any strictly increasing array from exactly 0 to exactly
    1, at least 8 entries; if ``nodes`` is also given it must equal its length).
    Graded fractions use the separate non-uniform discretisation.
    """

    def __init__(
        self,
        *,
        elastic_modulus: float,
        viscosity: float,
        inner: float,
        outer: float,
        nodes: int | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-10,
        node_fractions: np.ndarray | None = None,
    ) -> None:
        if not outer > inner > 0:
            raise ValueError("far-field domain requires 0 < inner < outer")
        if node_fractions is None:
            if nodes is None:
                raise ValueError("far-field membrane needs nodes or node_fractions")
            if nodes < 8:
                raise ValueError("far-field membrane needs at least 8 nodes")
            xi = np.linspace(0.0, 1.0, nodes)
            self._graded = False
        else:
            xi = np.array(node_fractions, dtype=float)
            if xi.ndim != 1 or len(xi) < 8:
                raise ValueError("far-field membrane needs at least 8 node fractions")
            if nodes is not None and nodes != len(xi):
                raise ValueError("nodes disagrees with the number of node fractions")
            if not (np.all(np.isfinite(xi)) and xi[0] == 0.0 and xi[-1] == 1.0
                    and np.all(np.diff(xi) > 0)):
                raise ValueError(
                    "node fractions must increase strictly from exactly 0 to exactly 1"
                )
            nodes = len(xi)
            self._graded = True
        self.G = float(elastic_modulus)
        self.mu = float(viscosity)
        self.outer = float(outer)
        self.xi = xi
        self.rtol, self.atol = rtol, atol
        self.time = 0.0
        self.inner = float(inner)
        r = self.radii()
        self.u = np.zeros(nodes)
        self.h = np.ones(nodes)
        self.Y = r.copy()
        n = nodes
        band = diags([np.ones(n - 2), np.ones(n - 1), np.ones(n), np.ones(n - 1), np.ones(n - 2)],
                     [-2, -1, 0, 1, 2], shape=(n, n))
        self._sparsity = bmat([[band] * 3] * 3).tocsr()

    # ------------------------------------------------------------------ state
    @property
    def graded(self) -> bool:
        """Whether this membrane uses explicit node fractions (non-uniform path)."""

        return self._graded

    def radii(self, inner: float | None = None) -> np.ndarray:
        a = self.inner if inner is None else inner
        return a + self.xi * (self.outer - a)

    def state_at(self, radius: float) -> EdgeState:
        r = self.radii()
        if not r[0] <= radius <= r[-1]:
            raise ValueError(f"radius {radius} outside the far-field domain")
        u = float(np.interp(radius, r, self.u))
        h = float(np.interp(radius, r, self.h))
        Y = float(np.interp(radius, r, self.Y))
        stretch_theta = radius / Y
        return EdgeState(u, h, 1.0 / (stretch_theta * h), stretch_theta)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """State as arrays for ``np.savez``.

        A graded membrane adds ``graded = True``; a uniform one writes exactly
        the keys it always has, so its state files are unchanged.
        """

        arrays = {
            "time": np.array(self.time), "inner": np.array(self.inner),
            "outer": np.array(self.outer), "xi": self.xi,
            "u": self.u, "h": self.h, "Y": self.Y,
            "G": np.array(self.G), "mu": np.array(self.mu),
        }
        if self._graded:
            arrays["graded"] = np.array(True)
        return arrays

    @classmethod
    def from_arrays(cls, data, **kwargs) -> FarFieldMembrane:
        """Rebuild a membrane from :meth:`to_arrays` output (or an ``np.load``).

        Stored fractions equal to ``linspace(0, 1, n)``, with no ``graded`` flag,
        give the uniform path, as every earlier state file does; otherwise the
        stored fractions are used as ``node_fractions``.
        """

        xi = np.asarray(data["xi"], dtype=float)
        graded = "graded" in data and bool(data["graded"])
        if not graded and np.array_equal(xi, np.linspace(0.0, 1.0, len(xi))):
            layout = {"nodes": len(xi)}
        else:
            layout = {"node_fractions": xi}
        self = cls(
            elastic_modulus=float(data["G"]), viscosity=float(data["mu"]),
            inner=float(data["inner"]), outer=float(data["outer"]),
            **layout, **kwargs,
        )
        if not np.array_equal(self.xi, data["xi"]):
            raise ValueError("far-field node fractions disagree with the stored state")
        self.time = float(data["time"])
        self.u, self.h, self.Y = (np.array(data[k], dtype=float) for k in ("u", "h", "Y"))
        return self

    def extended(self, outer: float, node_spacing: float) -> FarFieldMembrane:
        """Return this state on a longer domain, padded with undisturbed film.

        Used when a restart lengthens the membrane: beyond the old end the film
        is taken to be at rest, which is exact while the pulse has not reached
        that end.  Existing values are linearly interpolated onto the new nodes.
        """

        if not outer > self.outer:
            raise ValueError("extended outer end must exceed the current one")
        if not node_spacing > 0:
            raise ValueError("node spacing must be positive")
        nodes = math.ceil((outer - self.inner) / node_spacing) + 1
        new = FarFieldMembrane(
            elastic_modulus=self.G, viscosity=self.mu, inner=self.inner, outer=outer,
            nodes=nodes, rtol=self.rtol, atol=self.atol,
        )
        old_r, new_r = self.radii(), new.radii()
        beyond = new_r > self.outer
        new.u = np.where(beyond, 0.0, np.interp(new_r, old_r, self.u))
        new.h = np.where(beyond, 1.0, np.interp(new_r, old_r, self.h))
        new.Y = np.where(beyond, new_r, np.interp(new_r, old_r, self.Y))
        new.time = self.time
        return new

    def extended_graded(
        self, outer: float, spacing: float, near_length: float, growth: float
    ) -> FarFieldMembrane:
        """Return this state on a longer, graded domain, padded with undisturbed film.

        The graded analogue of :meth:`extended`, for a uniform or a graded
        membrane: the new nodes are :func:`graded_fractions` of
        ``[inner, outer]``, existing values are linearly interpolated onto them
        inside the old domain, and beyond the old end the film is at rest and
        unstretched (``u = 0``, ``h = 1``, ``Y = r``), which is exact while the
        pulse has not reached that end.  Time and inner end are kept.  Where the
        new grid is coarser than the old one the interpolation only samples the
        old state, so lengthen before the pulse reaches the region being
        coarsened.
        """

        if not outer > self.outer:
            raise ValueError("extended outer end must exceed the current one")
        fractions = graded_fractions(self.inner, outer, spacing, near_length, growth)
        new = FarFieldMembrane(
            elastic_modulus=self.G, viscosity=self.mu, inner=self.inner, outer=outer,
            node_fractions=fractions, rtol=self.rtol, atol=self.atol,
        )
        old_r, new_r = self.radii(), new.radii()
        beyond = new_r > self.outer
        new.u = np.where(beyond, 0.0, np.interp(new_r, old_r, self.u))
        new.h = np.where(beyond, 1.0, np.interp(new_r, old_r, self.h))
        new.Y = np.where(beyond, new_r, np.interp(new_r, old_r, self.Y))
        new.time = self.time
        return new

    # --------------------------------------------------------------- dynamics
    @staticmethod
    def _upwind(f: np.ndarray, a: np.ndarray, dr: float) -> np.ndarray:
        """Second-order upwind derivative of ``f`` for ``f_t = a f_r``."""

        d = np.empty_like(f)
        fwd = np.empty_like(f); bwd = np.empty_like(f)
        fwd[:-2] = (-3 * f[:-2] + 4 * f[1:-1] - f[2:]) / (2 * dr)
        fwd[-2:] = (f[-1] - f[-2]) / dr
        bwd[2:] = (3 * f[2:] - 4 * f[1:-1] + f[:-2]) / (2 * dr)
        bwd[:2] = (f[1] - f[0]) / dr
        d[:] = np.where(a > 0, fwd, bwd)
        return d

    @staticmethod
    def _upwind_graded(f: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
        """:meth:`_upwind` on a non-uniform grid with cell widths ``d = diff(r)``.

        Three-point one-sided second-order differences with the local widths
        (exact for quadratics), and the same first-order two-point fallbacks at
        the last two nodes (forward) and first two nodes (backward).
        """

        d0, d1 = d[:-1], d[1:]          # d0[j] = r[j+1] - r[j], d1[j] = r[j+2] - r[j+1]
        s = d0 + d1
        fwd = np.empty_like(f); bwd = np.empty_like(f)
        fwd[:-2] = (-(2 * d0 + d1) / (d0 * s) * f[:-2] + s / (d0 * d1) * f[1:-1]
                    - d0 / (d1 * s) * f[2:])
        fwd[-2:] = (f[-1] - f[-2]) / d[-1]
        bwd[2:] = ((2 * d1 + d0) / (d1 * s) * f[2:] - s / (d0 * d1) * f[1:-1]
                   + d1 / (d0 * s) * f[:-2])
        bwd[:2] = (f[1] - f[0]) / d[0]
        return np.where(a > 0, fwd, bwd)

    def _rhs(self, t, y, t0, a0, a1, v0, v1, dt):
        if self._graded:
            return self._rhs_graded(t, y, t0, a0, a1, v0, v1, dt)
        n = len(self.xi)
        u, h, Y = y[:n].copy(), y[n:2 * n], y[2 * n:]
        s = (t - t0) / dt
        a = a0 + s * (a1 - a0); adot = (a1 - a0) / dt
        u[0] = v0 + s * (v1 - v0)
        u[-1] = 0.0
        r = a + self.xi * (self.outer - a)
        dr = (self.outer - a) / (n - 1)
        w = (1.0 - self.xi) * adot
        G, mu = self.G, self.mu
        # midpoint fluxes
        rm = 0.5 * (r[1:] + r[:-1]); um = 0.5 * (u[1:] + u[:-1])
        hm = 0.5 * (h[1:] + h[:-1]); Ym = 0.5 * (Y[1:] + Y[:-1])
        urm = (u[1:] - u[:-1]) / dr
        ltm = rm / Ym; lrm = 1.0 / (ltm * hm)
        Nrr = 2 * mu * hm * (2 * urm + um / rm) + G * hm * (lrm**2 - hm**2)
        # nodal hoop stress
        ur = np.gradient(u, dr)
        lt = r / Y
        Ntt = 2 * mu * h * (ur + 2 * u / r) + G * h * (lt**2 - h**2)
        rel = w - u
        dudt = np.zeros(n); dhdt = np.zeros(n); dYdt = np.zeros(n)
        F = rm * Nrr
        dudt[1:-1] = rel[1:-1] * self._upwind(u, rel, dr)[1:-1] + (
            (F[1:] - F[:-1]) / dr / r[1:-1] - Ntt[1:-1] / r[1:-1]
        ) / h[1:-1]
        dudt[0] = (v1 - v0) / dt
        dhdt[:-1] = (rel * self._upwind(h, rel, dr) - h * (ur + u / r))[:-1]
        dYdt[:-1] = (rel * self._upwind(Y, rel, dr))[:-1]
        return np.concatenate([dudt, dhdt, dYdt])

    def _rhs_graded(self, t, y, t0, a0, a1, v0, v1, dt):
        """:meth:`_rhs` on non-uniform fractions, with the same structure.

        ``N_rr`` lives at cell midpoints with ``u_r = du / d_local``; its
        divergence at node ``i`` is ``(F_{i+1/2} - F_{i-1/2}) / (r_{i+1/2} -
        r_{i-1/2})``, the conservative finite-volume form (second order on a
        smoothly graded grid; at a fixed ratio ``g`` of neighbouring widths it
        keeps a local error ``~ (g - 1) d F''/4``).  The hoop stress uses the
        second-order non-uniform nodal gradient of ``np.gradient``, advection
        :meth:`_upwind_graded`, and the boundary rows are those of the uniform
        path.  Every term stays within two nodes, so the pentadiagonal
        ``_sparsity`` blocks still cover the Jacobian.
        """

        n = len(self.xi)
        u, h, Y = y[:n].copy(), y[n:2 * n], y[2 * n:]
        s = (t - t0) / dt
        a = a0 + s * (a1 - a0); adot = (a1 - a0) / dt
        u[0] = v0 + s * (v1 - v0)
        u[-1] = 0.0
        r = a + self.xi * (self.outer - a)
        d = np.diff(r)
        w = (1.0 - self.xi) * adot
        G, mu = self.G, self.mu
        # midpoint fluxes with local spacings
        rm = 0.5 * (r[1:] + r[:-1]); um = 0.5 * (u[1:] + u[:-1])
        hm = 0.5 * (h[1:] + h[:-1]); Ym = 0.5 * (Y[1:] + Y[:-1])
        urm = (u[1:] - u[:-1]) / d
        ltm = rm / Ym; lrm = 1.0 / (ltm * hm)
        Nrr = 2 * mu * hm * (2 * urm + um / rm) + G * hm * (lrm**2 - hm**2)
        # nodal hoop stress, second-order non-uniform gradient
        ur = np.gradient(u, r)
        lt = r / Y
        Ntt = 2 * mu * h * (ur + 2 * u / r) + G * h * (lt**2 - h**2)
        rel = w - u
        dudt = np.zeros(n); dhdt = np.zeros(n); dYdt = np.zeros(n)
        F = rm * Nrr
        volume = rm[1:] - rm[:-1]
        dudt[1:-1] = rel[1:-1] * self._upwind_graded(u, rel, d)[1:-1] + (
            (F[1:] - F[:-1]) / volume / r[1:-1] - Ntt[1:-1] / r[1:-1]
        ) / h[1:-1]
        dudt[0] = (v1 - v0) / dt
        dhdt[:-1] = (rel * self._upwind_graded(h, rel, d) - h * (ur + u / r))[:-1]
        dYdt[:-1] = (rel * self._upwind_graded(Y, rel, d))[:-1]
        return np.concatenate([dudt, dhdt, dYdt])

    def advance(self, t1: float, inner1: float, drive0: float, drive1: float) -> None:
        """Advance to ``t1``: inner end moves linearly to ``inner1`` while its
        velocity moves linearly from ``drive0`` to ``drive1``."""

        dt = t1 - self.time
        if not dt > 0:
            raise ValueError("far-field advance requires a later time")
        if not inner1 < self.outer:
            raise ValueError("far-field inner end reached the outer end")
        y0 = np.concatenate([self.u, self.h, self.Y])
        y0[0] = drive0
        args = (self.time, self.inner, inner1, drive0, drive1, dt)
        # The same steps solve_ivp(method="BDF") takes, with the solver held here: its
        # closures refer back to it, so once it outlives a few garbage collections its
        # Jacobian and LU are freed only by a full collection, and on a long membrane
        # that garbage grows by tens of MB per interval. Clearing it frees them now.
        solver = BDF(lambda t, y: self._rhs(t, y, *args), self.time, y0, t1,
                     jac_sparsity=self._sparsity, rtol=self.rtol, atol=self.atol)
        message = None
        while solver.status == "running":
            message = solver.step()
        if solver.status != "finished":
            raise RuntimeError(f"far-field membrane failed: {message}")
        yend = solver.y.copy()
        solver.__dict__.clear()
        n = len(self.xi)
        self.u, self.h, self.Y = yend[:n].copy(), yend[n:2 * n].copy(), yend[2 * n:].copy()
        self.u[0], self.u[-1] = drive1, 0.0
        self.time, self.inner = float(t1), float(inner1)
        if not (np.all(np.isfinite(self.u)) and np.all(self.h > 0) and np.all(self.Y > 0)):
            raise RuntimeError("far-field membrane state became unphysical")
