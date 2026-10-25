"""Axisymmetric co-moving Taylor--Culick retraction with Oldroyd-B elasticity.

This is deliberately a thin extension of :mod:`axisymmetric_comoving`.  The
mesh, ALE kinematics, sharp free-surface traction, tip control and laboratory-
frame velocity are inherited unchanged.  The only new bulk contribution is
the polymer conformation equation and its stress ``G (C - I)``.

The dimensionless modulus used here is ``G = G_dim h0 / sigma``, on the same
full-thickness ``h0`` and surface-tension scale as the inherited problem.
``passive_conformation=True`` retains the log-conformation transport and
diagnostic output while omitting polymer feedback in the momentum equation;
it is required when ``G=0``.

``relaxation_time=math.inf`` is an exact non-relaxing limit: no large finite
surrogate is used.  The incoming film is unstretched (``log(C)=0``), while the
midplane imposes only the tensor symmetry condition.  With the opt-in
``far_field="membrane"`` the incoming film instead carries the stretch of the
coupled far-field membrane, ``log(C) = diag(2 log lam_r, 2 log lam_z,
2 log lam_th)``, supplied through global parameters.  The free surface has no
conformation boundary condition because it is a material boundary.
"""

from __future__ import annotations

import math

import numpy as np
from pyoomph import DirichletBC, IntegralObservables, TemporalErrorEstimator, WeakContribution
from pyoomph.expressions import dot, piecewise_geq0, var, var_and_test

from taylor_culick_fem.absorbing_layer import (
    RayleighSpongeParameters,
    rayleigh_coefficient,
)
from taylor_culick_fem.axisymmetric_comoving import AxisymmetricComovingProblem
from taylor_culick_fem.viscoelastic import (
    GLambdaOldroydBParameters,
    make_g_lambda_oldroyd_b_equations,
)


class AxisymmetricViscoelasticProblem(AxisymmetricComovingProblem):
    """Co-moving axisymmetric sheet with solvent viscosity and polymer elasticity.

    ``solvent_oh`` is the project's Ohnesorge number, so the viscosity passed
    to Navier--Stokes is ``mu_s = sqrt(2) * solvent_oh``.  ``elastic_modulus``
    is ``G = G_dim h0 / sigma`` and ``relaxation_time`` is ``lambda`` in
    capillary units.  ``passive_conformation`` selects diagnostic-only
    conformation transport; active polymer feedback remains the default.
    """

    far_edge_history_fields = (
        "velocity_x", "log_conformation_xx", "log_conformation_yy", "log_conformation_aa",
    )

    def __init__(self):
        super().__init__()
        self.solvent_oh = 0.0
        self.elastic_modulus = 1.0
        self.relaxation_time = math.inf
        self.passive_conformation = False
        self.conformation_space = "C2"
        self.conformation_stabilization = "SUPG"
        self.conformation_supg_factor = 1.0
        self.viscoelastic_parameters: GLambdaOldroydBParameters | None = None
        self.sponge: RayleighSpongeParameters | None = None
        # Fields whose step-to-step change drives the adaptive time step, with weights.
        # Empty keeps the old behaviour: no estimator, so every step runs at the cap.
        # ``X`` and ``U`` are the global rim position and speed; the rest live on the
        # liquid.
        self.temporal_error_factors: dict[str, float] = {}

    def define_problem(self):
        if not math.isfinite(self.solvent_oh) or self.solvent_oh < 0:
            raise ValueError("solvent_oh must be finite and non-negative")
        if not math.isfinite(self.elastic_modulus) or self.elastic_modulus < 0:
            raise ValueError("elastic_modulus must be finite and non-negative")
        if not (self.relaxation_time > 0):
            raise ValueError("relaxation_time must be positive (or infinity)")
        if type(self.passive_conformation) is not bool:
            raise TypeError("passive_conformation must be a boolean")
        if self.elastic_modulus == 0 and not self.passive_conformation:
            raise ValueError("elastic_modulus=0 requires passive_conformation=True")

        # NavierStokesEquations in the inherited problem carries the solvent
        # stress only.  Velocities remain laboratory-frame velocities.
        self.mu = math.sqrt(2.0) * self.solvent_oh
        super().define_problem()

        if self.sponge is not None:
            self.sponge.validate_window(self.window_length)
            if self.sponge.gamma_max > 0:
                # The ALE mesh translates, but ``velocity`` is the laboratory
                # velocity.  With f=-rho*gamma*u and rho=1, Navier--Stokes'
                # ``-weak(f,u_test)`` convention gives this positive residual.
                velocity, velocity_test = var_and_test("velocity")
                x_rel = var("mesh_x") - var("X", domain="globals")
                gamma = rayleigh_coefficient(x_rel, self.sponge)
                core = piecewise_geq0(self.sponge.start - x_rel, 1, 0)
                sponge_eqs = WeakContribution(gamma * velocity, velocity_test)
                sponge_eqs += IntegralObservables(
                    sponge_power=-gamma * dot(velocity, velocity),
                    sponge_core_velocity_squared=core * dot(velocity, velocity),
                    sponge_core_volume=core,
                )
                self += sponge_eqs @ "liquid"

        params = GLambdaOldroydBParameters(
            elastic_modulus=self.elastic_modulus,
            relaxation_time=self.relaxation_time,
        )
        self.viscoelastic_parameters = params
        eqs = make_g_lambda_oldroyd_b_equations(
            params,
            formulation="log-conf",
            space=self.conformation_space,
            stabilization=self.conformation_stabilization,
            supg_factor=self.conformation_supg_factor,
            add_polymer_stress_to_momentum=not self.passive_conformation,
            output_conformation=True,
        )

        # Undisturbed material enters through the translating far boundary.
        # In axisymmetry the azimuthal tensor component is named ``aa``.
        if self.far_field == "membrane":
            if self.passive_conformation or not math.isinf(self.relaxation_time):
                raise ValueError("far_field='membrane' requires active lambda=inf elasticity")
            # Pure stretching of a thin film: C = diag(lam_r^2, lam_z^2,
            # lam_th^2) from the far-field membrane. Parameters, not unknowns.
            eqs += (
                DirichletBC(
                    log_conformation_xx=self.define_global_parameter(far_field_log_c_xx=0.0),
                    log_conformation_xy=0,
                    log_conformation_yy=self.define_global_parameter(far_field_log_c_yy=0.0),
                    log_conformation_aa=self.define_global_parameter(far_field_log_c_aa=0.0),
                )
                @ "far"
            )
        else:
            eqs += (
                DirichletBC(
                    log_conformation_xx=0,
                    log_conformation_xy=0,
                    log_conformation_yy=0,
                    log_conformation_aa=0,
                )
                @ "far"
            )
        # Reflection symmetry forbids only the r-z shear component.  The
        # normal components are evolved, not prescribed.  There is
        # deliberately no conformation condition on ``interface``.
        eqs += DirichletBC(log_conformation_xy=0) @ "symmetry"
        self += eqs @ "liquid"

        factors = dict(self.temporal_error_factors)
        if not all(math.isfinite(v) and v > 0 for v in factors.values()):
            raise ValueError("temporal error factors must be finite and positive")
        global_factors = {k: factors.pop(k) for k in ("X", "U") if k in factors}
        if global_factors and (len(global_factors) != 2 or global_factors["X"] != global_factors["U"]):
            # See comphy-lab/pyoomph#24: unequal global weights are silently misapplied.
            raise ValueError("temporal error factors must weight X and U together, equally")
        if factors:
            self += TemporalErrorEstimator(**factors) @ "liquid"
        if global_factors:
            self += TemporalErrorEstimator(**global_factors) @ "globals"

    def set_far_field(
        self, velocity: float, thickness: float, log_conformation: tuple[float, float, float],
        *, start=None, interval: tuple[float, float] | None = None,
    ) -> None:
        """Set the edge film velocity, full thickness and (r, z, theta) log C.

        With ``far_field_interpolation == "linear"`` these are the interval-end values,
        ``start`` is the (velocity, thickness, log C) at the interval start and
        ``interval`` its (start, end) times; the edge then moves linearly between them,
        evaluated at each step's new time level.
        """

        if self.far_field != "membrane":
            raise RuntimeError("set_far_field requires far_field='membrane'")
        linear = self.far_field_interpolation == "linear"
        if linear != (start is not None and interval is not None):
            raise ValueError("start state and interval are given exactly with linear interpolation")
        states = [(velocity, thickness, log_conformation)]
        if linear:
            states.append((start[0], start[1], start[2]))
        for u, h, logc in states:
            values = (u, h, *logc)
            if len(logc) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError("far-field state must be finite with three log C components")
            if not h > 0:
                raise ValueError("far-field thickness must be positive")
        names = ("far_field_velocity", "far_field_half_thickness",
                 "far_field_log_c_xx", "far_field_log_c_yy", "far_field_log_c_aa")
        end_values = (velocity, 0.5 * thickness, *log_conformation)
        if not linear:
            for name, value in zip(names, end_values, strict=True):
                self.get_global_parameter(name).value = value
            return
        if not (math.isfinite(interval[0]) and interval[1] > interval[0]):
            raise ValueError("far-field interval must be finite with end after start")
        start_values = (start[0], 0.5 * start[1], *start[2])
        self.set_far_field_ramp(
            interval[0], interval[1],
            {name: (a, b) for name, a, b in zip(names, start_values, end_values, strict=True)},
        )

    def sponge_diagnostics(self) -> dict[str, float]:
        """Return instantaneous damping power and undamped-core velocity RMS."""

        if self.sponge is None or self.sponge.gamma_max == 0:
            return {"sponge_power": 0.0, "sponge_core_velocity_rms": 0.0}
        mesh = self.get_mesh("liquid")
        power = float(mesh.evaluate_observable("sponge_power"))
        velocity_squared = float(mesh.evaluate_observable("sponge_core_velocity_squared"))
        volume = float(mesh.evaluate_observable("sponge_core_volume"))
        if volume <= 0:
            raise RuntimeError("sponge core has non-positive axisymmetric volume")
        return {
            "sponge_power": power,
            "sponge_core_velocity_rms": math.sqrt(max(velocity_squared / volume, 0.0)),
        }

    def polymer_diagnostic_extrema(self) -> dict[str, float]:
        """Return nodal conformation/stress extrema available for a smoke run.

        The primary unknown is ``Psi=log(C)``.  Reconstructing ``C=exp(Psi)``
        at nodes makes these diagnostics independent of optional VTK local-
        expression output.  For this Oldroyd-B parameterisation
        ``tau_p=G(C-I)`` exactly, including at infinite relaxation time.
        """
        if self.viscoelastic_parameters is None:
            return {}

        mesh = self.get_mesh("liquid")
        indices = mesh.get_nodal_field_indices()
        names = [
            "log_conformation_xx",
            "log_conformation_xy",
            "log_conformation_yy",
            "log_conformation_aa",
        ]
        missing = [name for name in names if name not in indices]
        if missing:
            raise RuntimeError(f"missing conformation fields: {', '.join(missing)}")

        psi_abs_max = 0.0
        c_min = math.inf
        c_max = -math.inf
        stress_min = math.inf
        stress_max = -math.inf
        for node in mesh.nodes():
            psi = np.array(
                [
                    [node.value(indices[names[0]]), node.value(indices[names[1]]), 0.0],
                    [node.value(indices[names[1]]), node.value(indices[names[2]]), 0.0],
                    [0.0, 0.0, node.value(indices[names[3]])],
                ],
                dtype=float,
            )
            psi_eigenvalues = np.linalg.eigvalsh(psi)
            c_eigenvalues = np.exp(psi_eigenvalues)
            stress_eigenvalues = self.elastic_modulus * (c_eigenvalues - 1.0)
            psi_abs_max = max(psi_abs_max, float(np.max(np.abs(psi_eigenvalues))))
            c_min = min(c_min, float(c_eigenvalues[0]))
            c_max = max(c_max, float(c_eigenvalues[-1]))
            stress_min = min(stress_min, float(stress_eigenvalues[0]))
            stress_max = max(stress_max, float(stress_eigenvalues[-1]))

        return {
            "log_conformation_abs_max": psi_abs_max,
            "conformation_eigenvalue_min": c_min,
            "conformation_eigenvalue_max": c_max,
            "polymer_stress_eigenvalue_min": stress_min,
            "polymer_stress_eigenvalue_max": stress_max,
        }
