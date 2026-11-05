"""Relaxing thin-film far field for the co-moving viscoelastic window.

This is the exterior of the opt-in ``physics.far_field`` mode
``"relaxing_membrane"``: an Oldroyd-B thin film whose conformation relaxes at a
finite Deborah number.  It is a separate model from the neo-Hookean
(``lambda = inf``) membrane of :mod:`taylor_culick_fem.farfield_membrane`, which
the ``"membrane"`` mode uses and which this module does not change.

Far from the rim the sheet is flat, thin and moves in plane.  Depth averaging
the axisymmetric solvent plus Oldroyd-B stresses gives

``h_t + (r h u)_r/r = 0`` and
``h (u_t + u u_r) = (r N_rr)_r/r - N_thth/r``, with

``N_rr = 2 mu h (2 u_r + u/r) + G h (C_rr - C_zz)`` and
``N_thth = 2 mu h (u_r + 2 u/r) + G h (C_thth - C_zz)``.

At finite relaxation time the three diagonal log-conformation components are
transported explicitly.  At infinite relaxation time the state layout is
that of the neo-Hookean membrane (``u``, ``h``, ``Y``): the conformation is
reconstructed from the material reference radius ``Y``.  A
zero relaxation time is the exact solvent-Newtonian endpoint, with ``C=I``
and no polymer feedback.

The domain ``[a(t), b]`` has a moving inner end and a fixed outer end.  Nodes
sit at fixed fractions of the domain, so time derivatives carry the ALE
velocity ``w = (1-xi) a'(t)``.  Finite-De transport is supported only while
the inner end is an outflow for membrane state, ``u(a) <= a'(t)``; otherwise
the missing 2D incoming thickness and conformation traces are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import BDF
from scipy.sparse import bmat, diags


@dataclass(frozen=True)
class EdgeState:
    """Film state at one radius, with optional explicit log conformation."""

    velocity: float
    thickness: float
    stretch_r: float
    stretch_theta: float
    log_c_rr: float | None = None
    log_c_zz: float | None = None
    log_c_th: float | None = None

    def log_conformation(self) -> tuple[float, float, float]:
        """Return ``log C`` ordered as radial, sheet-normal and azimuthal."""

        explicit = (self.log_c_rr, self.log_c_zz, self.log_c_th)
        if all(value is not None for value in explicit):
            return tuple(float(value) for value in explicit)  # type: ignore[arg-type]
        if any(value is not None for value in explicit):
            raise ValueError("explicit log conformation requires all three components")
        return (
            2.0 * math.log(self.stretch_r),
            2.0 * math.log(self.thickness),
            2.0 * math.log(self.stretch_theta),
        )


class RelaxingFarFieldMembrane:
    """Moving-inner-end thin-film membrane with optional Oldroyd-B relaxation."""

    def __init__(
        self,
        *,
        elastic_modulus: float,
        viscosity: float,
        inner: float,
        outer: float,
        nodes: int,
        relaxation_time: float = math.inf,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> None:
        if not outer > inner > 0:
            raise ValueError("far-field domain requires 0 < inner < outer")
        if nodes < 8:
            raise ValueError("far-field membrane needs at least 8 nodes")
        if not math.isfinite(elastic_modulus) or elastic_modulus < 0:
            raise ValueError("elastic modulus must be finite and non-negative")
        if not math.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and non-negative")
        if math.isnan(relaxation_time) or relaxation_time < 0:
            raise ValueError("relaxation time must be non-negative or infinity")
        if not math.isfinite(rtol) or not math.isfinite(atol) or rtol <= 0 or atol <= 0:
            raise ValueError("integration tolerances must be finite and positive")
        self.G = float(elastic_modulus)
        self.mu = float(viscosity)
        self.relaxation_time = float(relaxation_time)
        self.outer = float(outer)
        self.xi = np.linspace(0.0, 1.0, nodes)
        self.rtol, self.atol = rtol, atol
        self.time = 0.0
        self.inner = float(inner)
        radii = self.radii()
        self.u = np.zeros(nodes)
        self.h = np.ones(nodes)
        self.Y = radii.copy()
        if self.evolves_conformation:
            self.log_c_rr = np.zeros(nodes)
            self.log_c_zz = np.zeros(nodes)
            self.log_c_th = np.zeros(nodes)
        blocks = 6 if self.evolves_conformation else 3
        band = diags(
            [
                np.ones(nodes - 2),
                np.ones(nodes - 1),
                np.ones(nodes),
                np.ones(nodes - 1),
                np.ones(nodes - 2),
            ],
            [-2, -1, 0, 1, 2],
            shape=(nodes, nodes),
        )
        self._sparsity = bmat([[band] * blocks] * blocks).tocsr()

    @property
    def evolves_conformation(self) -> bool:
        """Whether finite positive De adds three transported log-C fields."""

        return 0.0 < self.relaxation_time < math.inf

    # ------------------------------------------------------------------ state
    def radii(self, inner: float | None = None) -> np.ndarray:
        a = self.inner if inner is None else inner
        return a + self.xi * (self.outer - a)

    def _logs_at(self, radius: float) -> tuple[float, float, float] | None:
        if self.relaxation_time == 0:
            return (0.0, 0.0, 0.0)
        if not self.evolves_conformation:
            return None
        radii = self.radii()
        return tuple(
            float(np.interp(radius, radii, component))
            for component in (self.log_c_rr, self.log_c_zz, self.log_c_th)
        )

    def state_at(self, radius: float) -> EdgeState:
        radii = self.radii()
        if not radii[0] <= radius <= radii[-1]:
            raise ValueError(f"radius {radius} outside the far-field domain")
        velocity = float(np.interp(radius, radii, self.u))
        thickness = float(np.interp(radius, radii, self.h))
        reference = float(np.interp(radius, radii, self.Y))
        stretch_theta = radius / reference
        stretch_r = 1.0 / (stretch_theta * thickness)
        logs = self._logs_at(radius)
        if logs is None:
            return EdgeState(velocity, thickness, stretch_r, stretch_theta)
        return EdgeState(velocity, thickness, stretch_r, stretch_theta, *logs)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Serialise state, retaining the legacy infinite-De array contract."""

        arrays = {
            "time": np.array(self.time),
            "inner": np.array(self.inner),
            "outer": np.array(self.outer),
            "xi": self.xi,
            "u": self.u,
            "h": self.h,
            "Y": self.Y,
            "G": np.array(self.G),
            "mu": np.array(self.mu),
        }
        if math.isfinite(self.relaxation_time):
            arrays["relaxation_time"] = np.array(self.relaxation_time)
        if self.evolves_conformation:
            arrays.update(
                log_c_rr=self.log_c_rr,
                log_c_zz=self.log_c_zz,
                log_c_th=self.log_c_th,
            )
        return arrays

    @classmethod
    def from_arrays(cls, data, **kwargs) -> RelaxingFarFieldMembrane:
        """Restore a state and reject parameter changes hidden in overrides."""

        stored_de = float(data["relaxation_time"]) if "relaxation_time" in data else math.inf
        requested_de = float(kwargs.pop("relaxation_time", stored_de))
        if requested_de != stored_de:
            raise ValueError("relaxation time disagrees with the stored far-field state")
        self = cls(
            elastic_modulus=float(data["G"]),
            viscosity=float(data["mu"]),
            relaxation_time=stored_de,
            inner=float(data["inner"]),
            outer=float(data["outer"]),
            nodes=len(data["xi"]),
            **kwargs,
        )
        if not np.array_equal(self.xi, data["xi"]):
            raise ValueError("far-field node fractions disagree with the stored state")
        self.time = float(data["time"])
        self.u, self.h, self.Y = (
            np.array(data[name], dtype=float) for name in ("u", "h", "Y")
        )
        if self.evolves_conformation:
            required = ("log_c_rr", "log_c_zz", "log_c_th")
            if any(name not in data for name in required):
                raise ValueError("finite-De far-field state lacks log-conformation arrays")
            self.log_c_rr, self.log_c_zz, self.log_c_th = (
                np.array(data[name], dtype=float) for name in required
            )
        self._check_state()
        return self

    def extended(self, outer: float, node_spacing: float) -> RelaxingFarFieldMembrane:
        """Return this state on a longer domain, padded with undisturbed film."""

        if not outer > self.outer:
            raise ValueError("extended outer end must exceed the current one")
        if not node_spacing > 0:
            raise ValueError("node spacing must be positive")
        nodes = math.ceil((outer - self.inner) / node_spacing) + 1
        new = RelaxingFarFieldMembrane(
            elastic_modulus=self.G,
            viscosity=self.mu,
            relaxation_time=self.relaxation_time,
            inner=self.inner,
            outer=outer,
            nodes=nodes,
            rtol=self.rtol,
            atol=self.atol,
        )
        old_r, new_r = self.radii(), new.radii()
        beyond = new_r > self.outer
        new.u = np.where(beyond, 0.0, np.interp(new_r, old_r, self.u))
        new.h = np.where(beyond, 1.0, np.interp(new_r, old_r, self.h))
        new.Y = np.where(beyond, new_r, np.interp(new_r, old_r, self.Y))
        if self.evolves_conformation:
            for name in ("log_c_rr", "log_c_zz", "log_c_th"):
                values = np.where(beyond, 0.0, np.interp(new_r, old_r, getattr(self, name)))
                setattr(new, name, values)
        new.time = self.time
        return new

    def outer_state_deviation(self, fraction: float = 0.1) -> dict[str, float]:
        """Return tail maxima for velocity, thickness, conformation and stress.

        The tail is the outer ``fraction`` of the domain plus the interior node just inside
        it, as in ``farfield_membrane._monitored_tail``. On a coarse grid the outer fraction
        can hold only the fixed outer endpoint, which stays undisturbed by construction, so
        without that bracketing node the monitor would read zero whatever reaches the end.
        """

        if not 0 < fraction <= 1:
            raise ValueError("outer-state fraction must lie in (0, 1]")
        tail = self.xi >= 1.0 - fraction
        first = int(np.argmax(tail)) if tail.any() else len(self.xi) - 1
        tail[max(first - 1, 0):] = True
        qrr, qzz, qth = self._principal_logs()
        crr, czz, cth = self._exp_logs(qrr, qzz, qth)
        return {
            "velocity": float(np.max(np.abs(self.u[tail]))),
            "thickness": float(np.max(np.abs(self.h[tail] - 1.0))),
            "log_conformation": float(
                max(np.max(np.abs(qrr[tail])), np.max(np.abs(qzz[tail])), np.max(np.abs(qth[tail])))
            ),
            "polymer_stress": float(
                self.G
                * max(np.max(np.abs(crr[tail] - czz[tail])), np.max(np.abs(cth[tail] - czz[tail])))
            ),
        }

    # --------------------------------------------------------------- dynamics
    @staticmethod
    def _upwind(f: np.ndarray, a: np.ndarray, dr: float) -> np.ndarray:
        """Second-order upwind derivative of ``f`` for ``f_t = a f_r``."""

        derivative = np.empty_like(f)
        forward = np.empty_like(f)
        backward = np.empty_like(f)
        forward[:-2] = (-3 * f[:-2] + 4 * f[1:-1] - f[2:]) / (2 * dr)
        forward[-2:] = (f[-1] - f[-2]) / dr
        backward[2:] = (3 * f[2:] - 4 * f[1:-1] + f[:-2]) / (2 * dr)
        backward[:2] = (f[1] - f[0]) / dr
        derivative[:] = np.where(a > 0, forward, backward)
        return derivative

    @staticmethod
    def _log_conformation_source(
        log_c_rr: np.ndarray,
        log_c_zz: np.ndarray,
        log_c_th: np.ndarray,
        radial_strain_rate: np.ndarray,
        hoop_strain_rate: np.ndarray,
        relaxation_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Material log-C rates for diagonal axisymmetric Oldroyd-B flow."""

        if not 0 < relaxation_time < math.inf:
            raise ValueError("finite conformation source requires 0 < De < infinity")
        try:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                relaxation = tuple(
                    -(1.0 - np.exp(-component)) / relaxation_time
                    for component in (log_c_rr, log_c_zz, log_c_th)
                )
        except FloatingPointError as exc:
            raise RuntimeError(
                "far-field relaxation left the representable log-conformation range"
            ) from exc
        return (
            2.0 * radial_strain_rate + relaxation[0],
            -2.0 * (radial_strain_rate + hoop_strain_rate) + relaxation[1],
            2.0 * hoop_strain_rate + relaxation[2],
        )

    @staticmethod
    def _exp_logs(*components: np.ndarray) -> tuple[np.ndarray, ...]:
        try:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                return tuple(np.exp(component) for component in components)
        except FloatingPointError as exc:
            raise RuntimeError("far-field conformation left the representable log range") from exc

    def _principal_logs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.relaxation_time == 0:
            zeros = np.zeros_like(self.h)
            return zeros, zeros, zeros
        if self.evolves_conformation:
            return self.log_c_rr, self.log_c_zz, self.log_c_th
        radii = self.radii()
        log_th = 2.0 * np.log(radii / self.Y)
        log_zz = 2.0 * np.log(self.h)
        return -log_th - log_zz, log_zz, log_th

    def _rhs(self, t, y, t0, a0, a1, v0, v1, dt):
        n = len(self.xi)
        u, h, reference = y[:n].copy(), y[n : 2 * n], y[2 * n : 3 * n]
        if self.evolves_conformation:
            qrr, qzz, qth = y[3 * n : 4 * n], y[4 * n : 5 * n], y[5 * n :]
        fraction = (t - t0) / dt
        inner = a0 + fraction * (a1 - a0)
        inner_speed = (a1 - a0) / dt
        u[0] = v0 + fraction * (v1 - v0)
        u[-1] = 0.0
        radii = inner + self.xi * (self.outer - inner)
        dr = (self.outer - inner) / (n - 1)
        mesh_speed = (1.0 - self.xi) * inner_speed
        relative = mesh_speed - u

        midpoint_r = 0.5 * (radii[1:] + radii[:-1])
        midpoint_u = 0.5 * (u[1:] + u[:-1])
        midpoint_h = 0.5 * (h[1:] + h[:-1])
        radial_rate_mid = (u[1:] - u[:-1]) / dr
        radial_rate = np.gradient(u, dr)
        hoop_rate = u / radii

        if self.evolves_conformation:
            qrr_mid = 0.5 * (qrr[1:] + qrr[:-1])
            qzz_mid = 0.5 * (qzz[1:] + qzz[:-1])
            crr_mid, czz_mid = self._exp_logs(qrr_mid, qzz_mid)
            _crr, czz, cth = self._exp_logs(qrr, qzz, qth)
        elif self.relaxation_time == 0:
            crr_mid = czz_mid = np.ones_like(midpoint_h)
            czz = cth = np.ones_like(h)
        else:
            midpoint_reference = 0.5 * (reference[1:] + reference[:-1])
            stretch_th_mid = midpoint_r / midpoint_reference
            stretch_r_mid = 1.0 / (stretch_th_mid * midpoint_h)
            crr_mid, czz_mid = stretch_r_mid**2, midpoint_h**2
            stretch_th = radii / reference
            czz, cth = h**2, stretch_th**2

        elastic_modulus = 0.0 if self.relaxation_time == 0 else self.G
        nrr = 2 * self.mu * midpoint_h * (
            2 * radial_rate_mid + midpoint_u / midpoint_r
        ) + elastic_modulus * midpoint_h * (crr_mid - czz_mid)
        ntt = 2 * self.mu * h * (
            radial_rate + 2 * hoop_rate
        ) + elastic_modulus * h * (cth - czz)
        force = midpoint_r * nrr

        dudt = np.zeros(n)
        dhdt = np.zeros(n)
        dYdt = np.zeros(n)
        dudt[1:-1] = relative[1:-1] * self._upwind(u, relative, dr)[1:-1] + (
            (force[1:] - force[:-1]) / dr / radii[1:-1] - ntt[1:-1] / radii[1:-1]
        ) / h[1:-1]
        dudt[0] = (v1 - v0) / dt
        dhdt[:-1] = (
            relative * self._upwind(h, relative, dr) - h * (radial_rate + hoop_rate)
        )[:-1]
        dYdt[:-1] = (relative * self._upwind(reference, relative, dr))[:-1]
        result = [dudt, dhdt, dYdt]
        if self.evolves_conformation:
            sources = self._log_conformation_source(
                qrr, qzz, qth, radial_rate, hoop_rate, self.relaxation_time
            )
            result.extend(
                relative * self._upwind(component, relative, dr) + source
                for component, source in zip((qrr, qzz, qth), sources, strict=True)
            )
        return np.concatenate(result)

    def _check_state(self) -> None:
        arrays = [self.u, self.h, self.Y]
        if self.evolves_conformation:
            arrays.extend((self.log_c_rr, self.log_c_zz, self.log_c_th))
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("far-field membrane state became non-finite")
        if not np.all(self.h > 0) or not np.all(self.Y > 0):
            raise RuntimeError("far-field membrane state became unphysical")
        if self.evolves_conformation:
            self._exp_logs(self.log_c_rr, self.log_c_zz, self.log_c_th)

    def advance(self, t1: float, inner1: float, drive0: float, drive1: float) -> None:
        """Advance the membrane to ``t1`` with linearly varying inner data."""

        dt = t1 - self.time
        if not dt > 0:
            raise ValueError("far-field advance requires a later time")
        if not inner1 < self.outer:
            raise ValueError("far-field inner end reached the outer end")
        inner_speed = (inner1 - self.inner) / dt
        roundoff = 32 * np.finfo(float).eps * max(
            1.0, abs(inner_speed), abs(drive0), abs(drive1)
        )
        if self.evolves_conformation and (
            drive0 > inner_speed + roundoff or drive1 > inner_speed + roundoff
        ):
            raise RuntimeError(
                "finite-De membrane inner end became inflow; "
                "provide 2D thickness and conformation traces"
            )
        initial = [self.u, self.h, self.Y]
        if self.evolves_conformation:
            initial.extend((self.log_c_rr, self.log_c_zz, self.log_c_th))
        y0 = np.concatenate(initial)
        y0[0] = drive0
        args = (self.time, self.inner, inner1, drive0, drive1, dt)
        # The steps solve_ivp(method="BDF") takes, with the solver held here: its closures
        # refer back to it, so a finished solver is otherwise freed only by a full garbage
        # collection, and its Jacobian and LU accumulate over many advances. Clearing it
        # once the endpoint is copied releases them at once (as in ``farfield_membrane``).
        solver = BDF(lambda t, y: self._rhs(t, y, *args), float(self.time), y0, float(t1),
                     jac_sparsity=self._sparsity, rtol=self.rtol, atol=self.atol)
        message = None
        while solver.status == "running":
            message = solver.step()
        if solver.status != "finished":
            solver.__dict__.clear()
            raise RuntimeError(f"far-field membrane failed: {message}")
        final = solver.y.copy()
        solver.__dict__.clear()
        n = len(self.xi)
        self.u = final[:n].copy()
        self.h = final[n : 2 * n].copy()
        self.Y = final[2 * n : 3 * n].copy()
        if self.evolves_conformation:
            self.log_c_rr = final[3 * n : 4 * n].copy()
            self.log_c_zz = final[4 * n : 5 * n].copy()
            self.log_c_th = final[5 * n :].copy()
        self.u[0], self.u[-1] = drive1, 0.0
        self.time, self.inner = float(t1), float(inner1)
        self._check_state()
