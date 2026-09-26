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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags


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
    """Moving-inner-end solver for the far-field membrane."""

    def __init__(
        self,
        *,
        elastic_modulus: float,
        viscosity: float,
        inner: float,
        outer: float,
        nodes: int,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> None:
        if not outer > inner > 0:
            raise ValueError("far-field domain requires 0 < inner < outer")
        if nodes < 8:
            raise ValueError("far-field membrane needs at least 8 nodes")
        self.G = float(elastic_modulus)
        self.mu = float(viscosity)
        self.outer = float(outer)
        self.xi = np.linspace(0.0, 1.0, nodes)
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
        return {
            "time": np.array(self.time), "inner": np.array(self.inner),
            "outer": np.array(self.outer), "xi": self.xi,
            "u": self.u, "h": self.h, "Y": self.Y,
            "G": np.array(self.G), "mu": np.array(self.mu),
        }

    @classmethod
    def from_arrays(cls, data, **kwargs) -> FarFieldMembrane:
        self = cls(
            elastic_modulus=float(data["G"]), viscosity=float(data["mu"]),
            inner=float(data["inner"]), outer=float(data["outer"]),
            nodes=len(data["xi"]), **kwargs,
        )
        if not np.array_equal(self.xi, data["xi"]):
            raise ValueError("far-field node fractions disagree with the stored state")
        self.time = float(data["time"])
        self.u, self.h, self.Y = (np.array(data[k], dtype=float) for k in ("u", "h", "Y"))
        return self

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

    def _rhs(self, t, y, t0, a0, a1, v0, v1, dt):
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
        sol = solve_ivp(
            self._rhs, (self.time, t1), y0, method="BDF",
            args=(self.time, self.inner, inner1, drive0, drive1, dt),
            jac_sparsity=self._sparsity, rtol=self.rtol, atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"far-field membrane failed: {sol.message}")
        n = len(self.xi)
        yend = sol.y[:, -1]
        self.u, self.h, self.Y = yend[:n].copy(), yend[n:2 * n].copy(), yend[2 * n:].copy()
        self.u[0], self.u[-1] = drive1, 0.0
        self.time, self.inner = float(t1), float(inner1)
        if not (np.all(np.isfinite(self.u)) and np.all(self.h > 0) and np.all(self.Y > 0)):
            raise RuntimeError("far-field membrane state became unphysical")
