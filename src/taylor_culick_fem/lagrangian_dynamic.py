"""Lagrangian dynamics of the opening hole in a Kelvin--Voigt neo-Hookean sheet.

With an infinite relaxation time the Oldroyd-B sheet of the dynamic runs is a Kelvin--Voigt
solid: the conformation tensor is the left Cauchy--Green tensor ``B = F F^T`` of the
deformation from the initial sheet, and the Cauchy stress is

    sigma = Ec (B - I) + mu_s (L + L^T) - p I,     L = grad_x v = (Grad v) F^-1,

with ``mu_s = sqrt(2) Oh`` (the project's Ohnesorge convention) and unit density, together
with surface tension 1 on the free surface. These are the equations of
:mod:`taylor_culick_fem.axisymmetric_viscoelastic` (polymer stress ``G (C - I)``, solvent
stress, Navier--Stokes inertia) written in the reference configuration, so no conformation
is transported: ``B`` is exact by construction.

The reference body and its mesh are those of the static solver
(:mod:`taylor_culick_fem.elastocapillary_equilibrium`): the initial half sheet with its
quarter-circle cap of radius 1/2 at ``r0``, nine-node quadrilaterals on structured blocks.
The unknowns are the displacement ``u``, the velocity ``v`` (both Q2) and a discontinuous
linear pressure enforcing ``det F = 1``. The equations, on the reference volume, are

    int rho dv/dt . phi + P : Grad phi dV0 + (surface tension) = 0,
    int (du/dt - v) . psi dV0 = 0,      int (J - 1) q dV0 = 0,

with ``P = Ec (F - cof F) - p cof F + sigma_v F^-T`` (``J = 1``). The elastic, pressure and
surface terms are the static solver's, unchanged, so a static equilibrium on the same mesh
is an exact discrete rest state of these equations.

The far end of the reference domain is either the static solver's Robin traction of the
linear far field (``far_edge="robin"``, with the pull of the continuing free surface) or a
clamped radius (``"clamped"``). Either reflects the outgoing elastic signal, so the domain
must be long enough that no reflection returns to the rim within the horizon; see
:func:`reflection_free_time`.

Large openings crush the edge region, so the mesh is redesigned in the deformed state and
pulled back to the reference body (:func:`taylor_culick_fem.equilibrium_geometry.design_grids`).
:func:`transfer_state` moves the displacement, the velocity and their BDF2 history to the new
mesh by exact Q2 interpolation in reference coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyoomph.expressions import (
    grad,
    matproduct,
    matrix,
    partial_t,
    testfunction,
    transpose,
    var,
)

from taylor_culick_fem.cases import MAX_STEP_HALVINGS
from taylor_culick_fem.elastocapillary_equilibrium import (
    DeformedSurfaceEnergy,
    ElastocapillaryEquilibriumProblem,
    FarFieldTraction,
    NeoHookeanDisplacementEquations,
    ReferenceSheetMesh,
    SurfaceContinuation,
)
from taylor_culick_fem.equilibrium_geometry import (
    HALF_THICKNESS,
    FieldMap,
    design_grids,
    mesh_mismatch,
    radial_nodes,
    reference_grids,
)

V_TAYLOR_CULICK = math.sqrt(2.0)


def deformation_gradient(g, eta):
    """Axisymmetric ``F = I + Grad u`` from the meridional gradient ``g`` and ``eta = u_r / X``."""
    return matrix([[1 + g[0, 0], g[0, 1], 0], [g[1, 0], 1 + g[1, 1], 0], [0, 0, 1 + eta]])


def inverse_deformation_gradient(g, eta):
    """Exact inverse of the block-diagonal axisymmetric ``F``."""
    a, b, c, d = 1 + g[0, 0], g[0, 1], g[1, 0], 1 + g[1, 1]
    det = a * d - b * c
    return matrix([[d / det, -b / det, 0], [-c / det, a / det, 0], [0, 0, 1 / (1 + eta)]])


def viscous_first_piola(grad_v, v_r, radius, g, eta, viscosity):
    """First Piola--Kirchhoff viscous stress ``J sigma_v F^-T`` with ``J = 1``.

    ``grad_v`` is the meridional reference gradient of the velocity, ``v_r / radius`` its
    hoop component (``radius`` the reference radius). The spatial velocity gradient is
    ``L = (Grad v) F^-1`` and ``sigma_v = viscosity (L + L^T)``.
    """
    grad_v = matrix([[grad_v[0, 0], grad_v[0, 1], 0], [grad_v[1, 0], grad_v[1, 1], 0],
                     [0, 0, v_r / radius]])
    f_inv = inverse_deformation_gradient(g, eta)
    spatial = matproduct(grad_v, f_inv)
    sigma = viscosity * (spatial + transpose(spatial))
    return matproduct(sigma, transpose(f_inv))


class KelvinVoigtLagrangianEquations(NeoHookeanDisplacementEquations):
    """Inertia and Kelvin--Voigt viscosity added to the static neo-Hookean solid.

    The static equations supply the elastic and pressure stress and the volume constraint;
    ``damping`` is zero, so the static mass used for its stability problem is absent. This
    class adds ``rho dv/dt`` and the viscous stress to the momentum rows (the displacement
    test functions) and the kinematic condition ``du/dt = v`` on the velocity test functions.
    """

    def __init__(self, shear_modulus, viscosity, density=1.0, displacement_space="C2",
                 pressure_space="DL"):
        super().__init__(shear_modulus, displacement_space=displacement_space,
                         pressure_space=pressure_space, damping=0)
        self.viscosity = viscosity
        self.density = density

    def define_fields(self):
        super().define_fields()
        self.define_vector_field("velocity", self.displacement_space)

    def define_residuals(self):
        super().define_residuals()
        u, v = var("displacement"), var("velocity")
        g = grad(u)
        radius = var("coordinate_x")
        eta = u[0] / radius
        stress = viscous_first_piola(grad(v), v[0], radius, g, eta, self.viscosity)
        test_u = testfunction("displacement")
        self.add_weak(stress, grad(test_u))
        self.add_weak(self.density * partial_t(v), test_u)
        self.add_weak(partial_t(u) - v, testfunction("velocity"))


class LagrangianSurfaceEnergy(DeformedSurfaceEnergy):
    required_parent_type = KelvinVoigtLagrangianEquations


class LagrangianFarFieldTraction(FarFieldTraction):
    required_parent_type = KelvinVoigtLagrangianEquations


def reflection_free_time(length: float, radius: float, elastic_modulus: float) -> float:
    """Earliest time at which a signal reflected at the far end can reach the rim.

    The fastest signal in the sheet is the dilatational membrane wave of the neo-Hookean
    film, speed ``2 sqrt(Ec)`` (plane stress, incompressible: ``E/(1 - nu^2) = 4 G``). A
    signal leaving the rim at reference radius ``radius`` at t = 0 returns after
    ``2 (length - radius) / (2 sqrt(Ec))``. The Kelvin--Voigt viscosity smooths the pulse but
    does not make it faster than this; it is a lower bound on the clean horizon only in that
    the viscous precursor is exponentially small ahead of the front.
    """
    if not (length > radius > 0 and elastic_modulus > 0):
        raise ValueError("need length > radius > 0 and a positive modulus")
    return (length - radius) / math.sqrt(elastic_modulus)


class LagrangianRetractionProblem(ElastocapillaryEquilibriumProblem):
    """Transient Lagrangian sheet: inertia, Kelvin--Voigt viscosity, surface tension.

    Reuses the static problem's reference mesh, readers (``hole_radius``, ``rim_height``,
    ``deformed_surface``) and far-edge options. ``elastic_modulus`` is ``Ec = G h0 / sigma``
    and ``oh`` the project's Ohnesorge number; the coded viscosity is ``sqrt(2) oh``.
    """

    def __init__(self):
        super().__init__()
        self.elastic_modulus = 1.0
        self.oh = 1.0
        self.density = 1.0
        self.quadratic_grids = True
        self.with_stability_mass = False
        self.newton_tolerance_at_unit_radius = 1e-8
        self.newton_tolerance_safety = 10.0
        # Radius of the moving part of the sheet, used in the round-off floor of Newton's
        # residual; the run sets it to the end of the finely held region after every output.
        self.newton_scale_radius = 1.0
        # When set, every transient step first keeps its start state (``step_start``), so a
        # step whose Newton solve fails can be restored and retried.
        self.keep_step_start = False
        self.step_start: dict | None = None

    @property
    def viscosity(self) -> float:
        return math.sqrt(2.0) * self.oh

    def define_problem(self):
        if self.far_edge not in ("robin", "clamped"):
            raise ValueError("far_edge must be 'robin' or 'clamped'")
        if not (math.isfinite(self.elastic_modulus) and self.elastic_modulus > 0):
            raise ValueError("elastic_modulus must be finite and positive")
        if not (math.isfinite(self.oh) and self.oh >= 0):
            raise ValueError("oh must be finite and non-negative")
        from pyoomph import DirichletBC

        self.set_coordinate_system("axisymmetric")
        if self.grids is None:
            self.grids = reference_grids(self.r0, self.length, self.elements_per_quarter, self.growth,
                                         self.first_spacing, self.max_spacing, self.core_fraction,
                                         quadratic=True)
        self.add_mesh(ReferenceSheetMesh(self.grids, self.r0, False, quadratic=True))
        modulus = self.elastic_modulus
        eqs = KelvinVoigtLagrangianEquations(modulus, self.viscosity, self.density,
                                             pressure_space=self.pressure_space)
        eqs += LagrangianSurfaceEnergy(self.surface_tension) @ "interface"
        eqs += DirichletBC(displacement_y=0, velocity_y=0) @ "symmetry"
        if self.far_edge == "clamped":
            eqs += DirichletBC(displacement_x=0, velocity_x=0) @ "far"
        else:
            eqs += LagrangianFarFieldTraction(modulus) @ "far"
            eqs += SurfaceContinuation(self.surface_tension) @ "interface/far"
        self.add_equations(eqs @ "solid")

    @property
    def elastocapillary_number(self) -> float:
        return float(self.elastic_modulus)

    def actions_before_newton_solve(self):
        """Newton tolerance: 1e-8, or the round-off floor ``eps r^2 / dt`` if that is larger.

        ``r`` is ``newton_scale_radius``, the extent of the moving part of the sheet, not the
        reference length: the far film is at rest, and scaling the floor with the 3e4-long
        domain would accept one Newton step with residuals near 1e-6 (measured; the floor at
        rest is about 6e-10).
        """
        super().actions_before_newton_solve()
        try:
            dt = float(self.time_pt().dt(0))
        except Exception:  # noqa: BLE001 -- no timestepper yet
            dt = 0.0
        base = self.newton_tolerance_at_unit_radius
        if not dt > 0:
            self.newton_solver_tolerance = base
            return
        floor = 2.2e-16 * self.newton_scale_radius ** 2 / dt
        self.newton_solver_tolerance = max(base, self.newton_tolerance_safety * floor)

    # ------------------------------------------------------------------ step start state

    def actions_before_transient_solve(self, *args, **kwargs):
        result = super().actions_before_transient_solve(*args, **kwargs)
        # After the parent, so the dof numbering it may set up is the one kept.
        self.step_start = self.step_state() if self.keep_step_start else None
        return result

    def step_state(self) -> dict:
        """Everything a transient step reads: every unknown at every stored time level, the
        time, the step history and the step count. Pinned values are constant here."""
        time_pt = self.time_pt()
        levels = self.time_stepper_pt().ntstorage()
        return {"dofs": [np.array(self.get_history_dofs(i), dtype=float) for i in range(levels)],
                "time": float(self.get_current_time(as_float=True, dimensional=False)),
                "dts": [float(time_pt.dt(i)) for i in range(time_pt.ndt())],
                "steps_done": int(self.timestepper.get_num_unsteady_steps_done())}

    def restore_step_state(self, state: dict) -> None:
        """Undo a step that failed part-way: its time shift, step history and Newton iterates."""
        for level, dofs in enumerate(state["dofs"]):
            self.set_history_dofs(level, dofs)
        self.set_current_time(state["time"], dimensional=False)
        time_pt = self.time_pt()
        for i, dt in enumerate(state["dts"]):
            time_pt.set_dt(i, dt)
        self.time_stepper_pt().set_weights()
        self.timestepper.set_num_unsteady_steps_done(state["steps_done"])

    # ------------------------------------------------------------------ readers

    def _tip_node(self):
        mesh = self.get_mesh("solid/interface")
        return min(mesh.nodes(), key=lambda nd: (nd.x(1), nd.x(0)))

    def tip_state(self) -> dict[str, float]:
        """Material tip on the midplane: reference radius, current radius and radial speed."""
        mesh = self.get_mesh("solid")
        idx = mesh.get_nodal_field_indices()
        node = self._tip_node()
        return {"X": float(node.x(0)), "R": float(node.x(0) + node.value(idx["displacement_x"])),
                "U": float(node.value(idx["velocity_x"]))}

    def nodal_fields(self) -> dict[str, int]:
        return self.get_mesh("solid").get_nodal_field_indices()


# ---------------------------------------------------------------------- meshes and remeshing

def sweep_radii(start: float, first: float, far: float, *, sweep_end: float, sweep_spacing: float,
                growth: float, max_spacing: float) -> np.ndarray:
    """Column radii: graded from ``first`` up to ``sweep_spacing``, held there to ``sweep_end``,
    then growing geometrically to ``max_spacing`` and the far radius ``far``.

    The held stretch covers the rim and the film it will sweep up before the next remesh, so
    material enters the rim already resolved at ``sweep_spacing``.
    """
    if not (far > start and sweep_spacing > 0 and growth > 1 and max_spacing >= sweep_spacing):
        raise ValueError("sweep radii need far > start, a positive sweep spacing, growth > 1 "
                         "and max_spacing >= sweep_spacing")
    end = min(max(sweep_end, start + 2 * sweep_spacing), far)
    near = radial_nodes(start, min(first, sweep_spacing), growth, end, sweep_spacing)
    if near[-1] >= far:
        return near
    if len(near) > 2:
        # radial_nodes snaps its last node to ``end``, stretching or shrinking the final
        # interval; drop it so the grading runs on smoothly into the tail.
        near = near[:-1]
    last = near[-1] - near[-2] if len(near) > 1 else sweep_spacing
    tail = radial_nodes(near[-1], min(last * growth, max_spacing), growth, far, max_spacing)
    return np.concatenate([near, tail[1:]])


def initial_grids(r0: float, length: float, elements_per_quarter: int, *, sweep_end: float,
                  sweep_spacing: float, growth: float, max_spacing: float,
                  core_fraction: float = 0.5) -> dict[str, np.ndarray]:
    """Nine-node reference grids of the initial sheet with a held sweep stretch.

    The cap blocks are the static solver's (:func:`reference_grids`); the sheet columns follow
    :func:`sweep_radii` from the cap end, with the cap's first spacing.
    """
    a, n = HALF_THICKNESS, elements_per_quarter
    # Only the cap blocks are kept; the sheet built here is replaced below, so its spacing
    # cap must not reject a fine ``max_spacing`` below the cap's own first spacing.
    grids = reference_grids(r0, length, elements_per_quarter, growth, None,
                            max(max_spacing, a / (2 * n)), core_fraction, quadratic=True)
    m = 2 * n
    c = core_fraction * a
    radii = sweep_radii(r0 + a, a / (2 * n), length, sweep_end=sweep_end,
                        sweep_spacing=sweep_spacing, growth=growth, max_spacing=max_spacing)
    radii = np.insert(radii, np.arange(1, len(radii)), 0.5 * (radii[1:] + radii[:-1]))
    heights = np.concatenate([np.linspace(0.0, c, m + 1), np.linspace(c, a, m + 1)[1:]])
    sheet = np.empty((len(radii), 2 * m + 1, 2))
    sheet[:, :, 0] = radii[:, None]
    sheet[:, :, 1] = heights[None, :]
    grids["sheet"] = sheet
    return grids


class StateMap(FieldMap):
    """Q2 element data of a transient state: displacement and velocity at every time level.

    ``levels`` has shape ``(time levels, 2 fields, elements, 9, 2)``; level 0 is the current
    value and field 0 the displacement, so the inherited :class:`FieldMap` methods
    (``forward``, ``pullback``) describe the current deformation.
    """

    FIELDS = ("displacement", "velocity")

    def __init__(self, Xe: np.ndarray, ij: np.ndarray, levels: np.ndarray):
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 5 or levels.shape[1] != 2 or levels.shape[2:] != Xe.shape:
            raise ValueError("levels must have shape (time levels, 2, elements, 9, 2)")
        super().__init__(Xe, levels[0, 0], ij)
        self.levels = levels

    @classmethod
    def from_problem(cls, problem, time_levels: int = 3) -> StateMap:
        mesh = problem.get_mesh("solid")
        idx = mesh.get_nodal_field_indices()
        index = [[idx[f"{name}_{axis}"] for axis in ("x", "y")] for name in cls.FIELDS]
        Xe, ij, values = [], [], []
        for element in mesh.elements():
            if element.nnode() != 9:
                raise RuntimeError("StateMap expects Q2 quadrilaterals")
            xs, ks, vs = [], [], []
            for k in range(9):
                node = element.node_pt(k)
                loc = element.local_coordinate_of_node(k)
                xs.append((node.x(0), node.x(1)))
                ks.append((round(loc[0]) + 1, round(loc[1]) + 1))
                levels = min(time_levels, node.ntstorage())
                vs.append([[[node.value_at_t(min(t, levels - 1), i) for i in field] for field in index]
                           for t in range(time_levels)])
            Xe.append(xs)
            ij.append(ks)
            values.append(vs)
        values = np.asarray(values)  # (elements, 9, levels, fields, 2)
        return cls(np.asarray(Xe), np.asarray(ij), np.transpose(values, (2, 3, 0, 1, 4)))

    def save(self, path, **extra):
        np.savez(path, Xe=self.Xe, ij=self.ij, levels=self.levels, **extra)

    @classmethod
    def load(cls, path) -> StateMap:
        data = np.load(path)
        return cls(data["Xe"], data["ij"], data["levels"])

    def values_at(self, X) -> np.ndarray:
        """All levels and fields at reference point ``X``: shape ``(levels, 2, 2)``."""
        e, s, t = self.locate(X, "ref", tol=1e-6)
        N, _, _ = self._shape(e, s, t)
        return np.einsum("k,lfkc->lfc", N, self.levels[:, :, e])


def install_state(problem, state: StateMap, time: float, dts: list[float],
                  steps_done: int = 2) -> dict[str, float]:
    """Interpolate ``state`` onto ``problem``'s nodes at every time level and set its clock.

    Every free displacement and velocity value at every stored time level is the Q2
    interpolant of ``state`` at the node's reference position; pinned values are left to
    their boundary conditions. The time, the BDF step history ``dts`` and the step count are
    set so that the next step is an ordinary second-order step, not a restart. The pressure is
    not transferred: it carries no history, and Newton recovers it from the constraint.
    Returns the largest interpolated value change of a pinned dof (zero on a consistent
    state) and the number of nodes.
    """
    mesh = problem.get_mesh("solid")
    idx = mesh.get_nodal_field_indices()
    index = [[idx[f"{name}_{axis}"] for axis in ("x", "y")] for name in StateMap.FIELDS]
    pinned_mismatch = 0.0
    count = 0
    for node in mesh.nodes():
        values = state.values_at(np.array([node.x(0), node.x(1)]))
        levels = min(values.shape[0], node.ntstorage())
        for f, field in enumerate(index):
            for c, i in enumerate(field):
                if node.is_pinned(i):
                    pinned_mismatch = max(pinned_mismatch, abs(values[0, f, c] - node.value(i)))
                    continue
                for level in range(levels):
                    node.set_value_at_t(level, i, float(values[level, f, c]))
        count += 1
    problem.set_current_time(float(time), dimensional=False)
    problem.initialise_dt(float(dts[0]))
    time_pt = problem.time_pt()
    for i in range(min(time_pt.ndt(), len(dts))):
        time_pt.set_dt(i, float(dts[i]))
    stepper = problem.time_stepper_pt()
    stepper.set_weights()
    stepper.undo_make_steady()
    problem._taken_already_an_unsteady_step = True
    problem.timestepper.set_num_unsteady_steps_done(max(2, int(steps_done)))
    problem._last_step_was_stationary = False
    return {"pinned_mismatch": float(pinned_mismatch), "nodes": count}


_GAUSS = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
_WEIGHTS = np.array([5 / 9, 8 / 9, 5 / 9])


def state_integrals(state: StateMap, elastic_modulus: float) -> dict:
    """Volume error ``int (J - 1) dV0``, kinetic and neo-Hookean energies of the current level.

    Third-order Gauss quadrature on every Q2 element with the axisymmetric measure
    ``2 pi X``; a remap compares these before and after to measure its transfer error.
    """
    volume_error = kinetic = elastic = 0.0
    for e in range(len(state.Xe)):
        u = state.levels[0, 0, e]
        for a, wa in zip(_GAUSS, _WEIGHTS, strict=True):
            for b, wb in zip(_GAUSS, _WEIGHTS, strict=True):
                N, Ns, Nt = state._shape(e, a, b)
                X = N @ state.Xe[e]
                dX = np.array([Ns @ state.Xe[e], Nt @ state.Xe[e]]).T
                v = N @ state.levels[0, 1, e]
                du = np.array([Ns @ u, Nt @ u]).T @ np.linalg.inv(dX)
                hoop = 1 + (N @ u)[0] / X[0]
                F = np.eye(2) + du
                weight = wa * wb * abs(np.linalg.det(dX)) * 2 * math.pi * X[0]
                volume_error += (np.linalg.det(F) * hoop - 1) * weight
                kinetic += 0.5 * float(v @ v) * weight
                elastic += 0.5 * elastic_modulus * (float(np.sum(F * F)) + hoop ** 2 - 3) * weight
    return {"volume_error": volume_error, "kinetic": kinetic, "elastic": elastic}


def rim_extent(surface: np.ndarray, tolerance: float = 0.01) -> float:
    """Outer radius of the rim: the last surface point, ordered from the tip, whose height
    exceeds the far film's ``(1 + tolerance)`` times.

    ``surface`` holds deformed ``(r, z)`` in its first two columns, ordered by reference
    arclength from the tip.
    """
    film = float(surface[-1, 1])
    raised = np.where(surface[:, 1] > film * (1 + tolerance))[0]
    return float(surface[raised[-1], 0]) if len(raised) else float(surface[0, 0])


def reference_arclength(points: np.ndarray, r0: float) -> np.ndarray:
    """Reference arclength from the tip of reference surface points ``(X, Z)``."""
    a = HALF_THICKNESS
    cx = r0 + a
    X, Z = np.asarray(points[:, 0], float), np.asarray(points[:, 1], float)
    on_arc = X <= cx + 1e-12
    angle = np.arctan2(Z, X - cx)  # pi at the tip, pi/2 at the shoulder
    return np.where(on_arc, a * (math.pi - angle), 0.5 * math.pi * a + (X - cx))


@dataclass
class LagrangianSettings:
    """Physics and numerics of a Lagrangian run; see the runner's case for the meaning."""

    oh: float
    elastic_modulus: float
    r0: float = 1.0
    surface_tension: float = 1.0
    far_edge: str = "robin"
    elements_per_quarter: int = 8
    length: float = 30000.0
    growth: float = 1.1
    max_spacing: float = 300.0
    sweep_spacing: float = 0.25
    sweep_ahead: float = 20.0
    sweep_ahead_fraction: float = 0.3
    columns: str = "fan"
    shoulder_angle: float = 10.0
    blend: float = 4.0
    # Remesh surface sampling: the chord sag allowed between surface samples, in h0. The column
    # nodes nearest the surface lie 1 / (4 elements_per_quarter) of the local height below it.
    surface_sag_tolerance: float = 5e-4
    remesh_log_radius_growth: float = 0.1
    remesh_front_margin: float = 5.0
    dt: float = 0.01
    max_newton: int = 10
    newton_tolerance: float = 1e-8
    newton_tolerance_safety: float = 10.0
    linear_solver: str | None = None
    max_step_halvings: int = MAX_STEP_HALVINGS

    def sweep_end(self, radius: float, extent: float) -> float:
        return extent + max(self.sweep_ahead, self.sweep_ahead_fraction * radius)


class LagrangianRun:
    """A transient Lagrangian run across remeshes.

    Holds the current pyoomph problem and rebuilds it on a redesigned mesh whenever the hole
    radius has grown by ``remesh_log_radius_growth`` in ``ln R`` since the last mesh or the rim
    has come within ``remesh_front_margin`` of the end of the held sweep stretch. Every remap
    is recorded with its transfer diagnostics.
    """

    def __init__(self, settings: LagrangianSettings, folder):
        from pathlib import Path

        self.settings = settings
        self.folder = Path(folder)
        self.problem = None
        self.grids = None
        self.mesh_index = -1
        self.mesh_radius = None
        self.sweep_end_radius = None
        self.remaps: list[dict] = []
        self.retries: list[dict] = []

    # -- problem construction
    def build(self, grids: dict) -> LagrangianRetractionProblem:
        s = self.settings
        self.mesh_index += 1
        problem = LagrangianRetractionProblem()
        problem.ignore_command_line = True
        problem.set_output_directory(str(self.folder / f"mesh{self.mesh_index:03d}"))
        problem.elastic_modulus = s.elastic_modulus
        problem.oh = s.oh
        problem.r0 = s.r0
        problem.surface_tension = s.surface_tension
        problem.far_edge = s.far_edge
        problem.length = s.length
        problem.grids = grids
        problem.max_newton_iterations = s.max_newton
        problem.newton_tolerance_at_unit_radius = s.newton_tolerance
        problem.newton_tolerance_safety = s.newton_tolerance_safety
        if s.linear_solver:
            problem.set_linear_solver(s.linear_solver)
        problem.initialise()
        self.grids = grids
        return problem

    def release(self) -> None:
        if self.problem is not None:
            import gc

            self.problem.release()
            self.problem = None
            gc.collect()

    def start_flat(self) -> None:
        """The dynamic runs' initial state: the flat sheet at rest."""
        s = self.settings
        grids = initial_grids(s.r0, s.length, s.elements_per_quarter,
                              sweep_end=s.sweep_end(s.r0, s.r0 + 1.0), sweep_spacing=s.sweep_spacing,
                              growth=s.growth, max_spacing=s.max_spacing)
        self.problem = self.build(grids)
        self.mesh_radius = s.r0
        self.sweep_end_radius = s.sweep_end(s.r0, s.r0 + 1.0)
        self.problem.newton_scale_radius = self.sweep_end_radius

    def start_from(self, grids: dict, state: StateMap, time: float, dts: list[float],
                   steps_done: int = 2, mesh_radius: float | None = None,
                   sweep_end_radius: float | None = None, exact: bool = False) -> dict:
        """Build on ``grids`` and install ``state`` (exact when the grids are the state's own).

        With ``exact``, the built mesh must be the state's own node for node (curved cap
        included), or nothing is installed: an exact transfer, never an interpolation.
        """
        self.problem = self.build(grids)
        if exact:
            built = StateMap.from_problem(self.problem, time_levels=1)
            mismatch = mesh_mismatch(built.Xe, built.ij, state.Xe, state.ij)
            if mismatch:
                raise ValueError(f"the seed state is not on the mesh its grids build: {mismatch}")
        info = install_state(self.problem, state, time, dts, steps_done)
        radius = self.tip()["R"]
        self.mesh_radius = radius if mesh_radius is None else mesh_radius
        self.sweep_end_radius = (self.settings.sweep_end(radius, self.extent())
                                 if sweep_end_radius is None else sweep_end_radius)
        self.problem.newton_scale_radius = self.sweep_end_radius
        return info

    # -- readers
    def tip(self) -> dict[str, float]:
        return self.problem.tip_state()

    def surface(self) -> np.ndarray:
        """Deformed surface ``(r, z, X, Z)`` ordered by reference arclength from the tip."""
        points = self.problem.deformed_surface()
        order = np.argsort(reference_arclength(points[:, 2:4], self.settings.r0))
        return points[order]

    def extent(self) -> float:
        return rim_extent(self.surface())

    def time(self) -> float:
        return float(self.problem.get_current_time(as_float=True))

    def dts(self) -> list[float]:
        time_pt = self.problem.time_pt()
        return [float(time_pt.dt(i)) for i in range(time_pt.ndt())]

    def steps_done(self) -> int:
        return int(self.problem.timestepper.get_num_unsteady_steps_done())

    # -- stepping
    def advance(self, target: float) -> None:
        """Step to ``target`` with steps of ``dt``, ending on a shorter step if needed.

        pyoomph's ``run`` takes its first step at ``startstep`` and does not shorten a step
        to land on ``endtime``, so whole steps and any remainder are run separately. A step
        whose Newton solve fails is retried in halves (:meth:`_retry`); stepping then resumes
        at ``dt`` from the end of that step, so the output times are unchanged. Steps that
        converge are taken exactly as without the retry.
        """
        dt = self.settings.dt
        target = float(target)
        tolerance = 1e-9 * max(1.0, abs(target))
        while True:
            remaining = target - self.time()
            if remaining <= tolerance:
                return
            whole = math.floor(remaining / dt + 1e-9)
            if whole >= 1:
                end, step = self.time() + whole * dt, dt
                if abs(end - target) <= tolerance:
                    end = target
            else:
                end, step = target, remaining
            before = self.time()
            self._run_to(end, step)
            if not self.time() > before:
                raise RuntimeError(f"no progress from t = {before!r} towards {target!r}")

    def _run_to(self, end: float, step: float) -> None:
        """Fixed steps of ``step`` to ``end``, the last one the exact remainder.

        One ``problem.run`` per step: over several steps pyoomph's ``run`` stretches a final
        step that falls a round-off short of ``step`` to 1.00001 times the remainder, which
        put grouped runs up to ~1e-7 past successive cadence boundaries. A one-step ``run``
        ends on its own end time (pyoomph drops a remaining sliver below 1e-8 of the step).
        """
        tolerance = 1e-9 * max(1.0, abs(end))
        while end - self.time() > tolerance:
            remaining = end - self.time()
            size = remaining if remaining - step <= tolerance else step
            before = self.time()
            self._run_step(before + size, size)
            if not self.time() > before:
                raise RuntimeError(f"no progress from t = {before!r} towards {end!r}")
        if self.time() > end + tolerance:
            raise RuntimeError(f"stepping overshot t = {end!r}: reached {self.time()!r}")

    def _run_step(self, end: float, step: float) -> None:
        """One ``problem.run`` step to ``end``; recover a step that fails."""
        problem = self.problem
        if self.settings.max_step_halvings <= 0:
            problem.run(end, startstep=step, maxstep=step, outstep=False)
            return
        problem.keep_step_start = True
        try:
            problem.run(end, startstep=step, maxstep=step, outstep=False)
        except RuntimeError as exc:  # pyoomph's OomphException: Newton did not converge
            start = problem.step_start
            if start is None:
                raise
            size = float(problem.time_pt().dt(0))
            self._retry(start, start["time"] + size, size, 1, exc)
        finally:
            problem.keep_step_start = False
            problem.step_start = None

    def _retry(self, start: dict, step_end: float, size: float, depth: int,
               failure: BaseException) -> None:
        """Retake the failed step ``[start, step_end]`` of length ``size`` as two halves.

        The problem is first restored to the step's start state. A half that fails in turn is
        split again, down to ``max_step_halvings`` levels; a failure below that is raised. The
        second half ends exactly on ``step_end``. Every retry is recorded in ``self.retries``
        with the residuals of the failed attempt.

        When Newton stalls on its round-off floor rather than diverging, a shorter step helps
        only through the acceptance: its ``newton_tolerance_safety * eps r^2 / dt`` term rises
        as ``dt`` falls, once it exceeds ``newton_tolerance``, while the floor itself does not
        fall. Accepting near the floor is the job of ``newton_tolerance_safety``; the retry is
        the safety net.
        """
        if depth > self.settings.max_step_halvings:
            raise failure
        problem = self.problem
        residuals = [float(r) for r in problem.get_last_residual_convergence()]
        problem.restore_step_state(start)
        self.retries.append({"t": start["time"], "dt": size, "depth": depth,
                             "residuals": residuals})
        for sub_end in (start["time"] + 0.5 * size, step_end):
            step = sub_end - self.time()
            try:
                problem.run(sub_end, startstep=step, maxstep=step, outstep=False)
            except RuntimeError as exc:
                sub_start = problem.step_start
                if sub_start is None:
                    raise
                self._retry(sub_start, sub_end, step, depth + 1, exc)

    def remesh_reason(self) -> str | None:
        s = self.settings
        radius = self.tip()["R"]
        if math.log(radius / self.mesh_radius) > s.remesh_log_radius_growth:
            return "radius_growth"
        if self.extent() > self.sweep_end_radius - s.remesh_front_margin:
            return "sweep_front"
        return None

    def remesh(self, reason: str) -> dict:
        """Redesign the mesh in the deformed state and move the state onto it."""
        import time as clock

        s = self.settings
        started = clock.time()
        old = StateMap.from_problem(self.problem)
        time, dts, steps = self.time(), self.dts(), self.steps_done()
        tip_old = self.tip()
        extent = self.extent()
        sweep_end = s.sweep_end(tip_old["R"], extent)
        before = state_integrals(old, s.elastic_modulus)
        grids = design_grids(old, s.elements_per_quarter, s.length, growth=s.growth,
                             max_spacing=s.max_spacing, r0=s.r0, columns=s.columns,
                             shoulder_angle=s.shoulder_angle, blend=s.blend,
                             surface_sag_tolerance=s.surface_sag_tolerance,
                             column_radii=lambda start, first, far: sweep_radii(
                                 start, first, far, sweep_end=sweep_end,
                                 sweep_spacing=s.sweep_spacing, growth=s.growth,
                                 max_spacing=s.max_spacing))
        self.release()
        info = self.start_from(grids, old, time, dts, steps, mesh_radius=tip_old["R"],
                               sweep_end_radius=sweep_end)
        new = StateMap.from_problem(self.problem)
        after = state_integrals(new, s.elastic_modulus)
        tip_new = self.tip()
        record = {
            "mesh": self.mesh_index, "reason": reason, "t": time, "R_before": tip_old["R"],
            "R_after": tip_new["R"], "U_before": tip_old["U"], "U_after": tip_new["U"],
            "elements": len(new.Xe), "ndof": int(self.problem.ndof()),
            "sweep_end": sweep_end, "rim_extent": extent,
            "volume_error_before": before["volume_error"],
            "volume_error_after": after["volume_error"],
            "kinetic_rel_change": ((after["kinetic"] - before["kinetic"]) / before["kinetic"]
                                   if before["kinetic"] > 0 else 0.0),
            "elastic_rel_change": ((after["elastic"] - before["elastic"]) / before["elastic"]
                                   if before["elastic"] > 0 else 0.0),
            "pinned_mismatch": info["pinned_mismatch"], "seconds": clock.time() - started,
        }
        self.remaps.append(record)
        return record

    def checkpoint(self, path, **extra) -> None:
        state = StateMap.from_problem(self.problem)
        np.savez(path, Xe=state.Xe, ij=state.ij, levels=state.levels,
                 grids_keys=np.array(sorted(self.grids)),
                 **{f"grid_{k}": v for k, v in self.grids.items()},
                 time=self.time(), dts=np.array(self.dts()), steps_done=self.steps_done(),
                 mesh_index=self.mesh_index, mesh_radius=self.mesh_radius,
                 sweep_end_radius=self.sweep_end_radius, **extra)

    @staticmethod
    def read_checkpoint(path) -> dict:
        data = np.load(path)
        grids = {str(k): data[f"grid_{k}"] for k in data["grids_keys"]}
        return {"grids": grids, "state": StateMap(data["Xe"], data["ij"], data["levels"]),
                "time": float(data["time"]), "dts": [float(x) for x in data["dts"]],
                "steps_done": int(data["steps_done"]), "mesh_index": int(data["mesh_index"]),
                "mesh_radius": float(data["mesh_radius"]),
                "sweep_end_radius": float(data["sweep_end_radius"]), "data": data}

    def start_checkpoint(self, path) -> dict:
        record = self.read_checkpoint(path)
        self.mesh_index = record["mesh_index"] - 1
        info = self.start_from(record["grids"], record["state"], record["time"], record["dts"],
                               record["steps_done"], record["mesh_radius"],
                               record["sweep_end_radius"])
        return {**record, **info}

    def start_static_seed(self, grids: dict, static: FieldMap) -> dict:
        """Start at rest from a static equilibrium on its own grids (exact nodal transfer)."""
        u = np.asarray(static.ue, float)
        levels = np.stack([np.stack([u, np.zeros_like(u)])] * 3)
        state = StateMap(static.Xe, static.ij, levels)
        dt = self.settings.dt
        return self.start_from(grids, state, 0.0, [dt, dt], steps_done=2, exact=True)


HOOP_BANDS = (1e-3, 3e-3, 0.015, 0.06, 0.5)


def hoop_bands(state: StateMap, elastic_modulus: float, radius: float,
               bands=HOOP_BANDS) -> list[dict[str, float]]:
    """Hoop force ``Ec int (C_thth - 1) dA`` within ``r - R <= d`` of the tip, per band ``d``.

    As the static diagnostic: third-order Gauss points of the current state, the current
    meridional area ``dA`` (half domain, no ``2 pi r``), ``C_thth = (r / X)^2``. A Gauss point
    counts when its own current radius lies within the band. ``area`` is the area counted;
    the incompressible kinematic estimate of the hoop force carried by the material that
    occupies that area is ``(Ec R / 2) ln(1 + 2 R area / r0)``.
    """
    widest = max(bands)
    current = state.Xe + state.ue
    near = np.where(current[:, :, 0].min(axis=1) <= radius + widest)[0]
    rows = [{"d": float(d), "hoop": 0.0, "area": 0.0} for d in bands]
    for e in near:
        for a, wa in zip(_GAUSS, _WEIGHTS, strict=True):
            for b, wb in zip(_GAUSS, _WEIGHTS, strict=True):
                N, Ns, Nt = state._shape(e, a, b)
                X = N @ state.Xe[e]
                x = N @ current[e]
                jac = np.array([Ns @ current[e], Nt @ current[e]]).T
                area = abs(np.linalg.det(jac)) * wa * wb
                hoop = elastic_modulus * ((x[0] / X[0]) ** 2 - 1) * area
                for row in rows:
                    if x[0] - radius <= row["d"]:
                        row["hoop"] += hoop
                        row["area"] += area
    return rows
