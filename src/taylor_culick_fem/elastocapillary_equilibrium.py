"""Static equilibria of a hole in an elastic sheet held open by surface tension.

The elastic runs use an Oldroyd-B sheet with infinite relaxation time. That material is a
Kelvin--Voigt solid: the conformation tensor is the left Cauchy--Green tensor of the
deformation from the initial sheet, and the polymer stress ``Ec (C - I)`` is that of an
incompressible neo-Hookean solid of shear modulus ``Ec = G h0 / sigma``. Once the sheet is
at rest the viscous stress vanishes, so an arrested hole is a static hyperelastic
equilibrium in which the neo-Hookean stress balances surface tension on the deformed free
surface. Neither ``Oh`` nor inertia enters it.

This module solves for those equilibria directly, in the reference (Lagrangian)
configuration, which is the initial sheet of the dynamic runs: a sheet of full thickness
``h0 = 1`` with a hole of radius ``r0`` on the midplane and a quarter-circle edge cap of
radius ``h0 / 2``. Only the upper half is modelled. The midplane is a symmetry plane. The
far edge at reference radius ``length`` carries the traction of the linear-elastic far
field, ``sigma_rr = -2 Ec u_r / r`` (the incompressible plane-stress Lame field
``u_r ~ 1/r``), together with the pull of the free surface that continues beyond it;
``far_edge="clamped"`` instead fixes the edge radius. The continuation parameter is
``log_Ec``, the natural logarithm of the shear modulus.

The unknowns are the displacement ``u = x - X`` from the reference position ``X`` on a
static reference mesh, and a pressure enforcing ``det F = 1``. Displacements, not
positions, are the unknowns because the far film is thin and reaches radii of order
``1e4``: with positions, the round-off of the large radial coordinates appears as shear
strain across the thin far elements and puts a floor under the Newton residual. The bulk
and surface terms are the variations of the energy
``E = int Ec/2 (I1 - 3) dV0 + int sigma dA``, with the pressure as a Lagrange multiplier,
written out explicitly.

Stability is assessed with an overdamped gradient flow, ``M du/dt = -dE/du``, whose mass
matrix is the reference-volume mass of the displacement weighted by ``1/X^2``; static solves
ignore it. The sign of its eigenvalues is that of the constrained energy Hessian, so an
equilibrium is a local minimum of the energy (and an attractor of the damped dynamics) when
every eigenvalue is negative. Only axisymmetric perturbations are represented.
"""

from __future__ import annotations

import math

import numpy as np
from pyoomph import DirichletBC, Problem
from pyoomph import _pyoomph_core as _pyoomph
from pyoomph.expressions import (
    dot,
    exp,
    grad,
    matproduct,
    matrix,
    partial_t,
    square_root,
    testfunction,
    transpose,
    var,
    vector,
)
from pyoomph.generic.codegen import Equations, InterfaceEquations
from pyoomph.meshes.mesh import MeshTemplate

from taylor_culick_fem.equilibrium_geometry import HALF_THICKNESS, reference_grids


class NeoHookeanDisplacementEquations(Equations):
    """Incompressible neo-Hookean solid in displacement form on the reference mesh.

    ``F = I + grad(u)`` is block diagonal in axisymmetry, ``[[a, b], [c, d]]`` in ``(r, z)``
    and the hoop stretch ``h = 1 + u_r / X``, so ``J = (ad - bc) h`` and the cofactor
    ``J F^-T`` are written out. The first Piola--Kirchhoff stress is
    ``P = mu (F - J F^-T) - p J F^-T`` (``J = 1``): the Cauchy stress is ``mu (B - I) - p I``,
    stress-free at ``u = 0, p = 0``. The pressure enforces ``J = 1``. ``damping`` adds
    ``damping * du/dt``, an overdamped mass used only for the stability eigenproblem.
    """

    def __init__(self, shear_modulus, displacement_space="C2", pressure_space="DL", damping=1,
                 error_estimators=False, stiffening=None, volume_penalty=0.0):
        super().__init__()
        self.shear_modulus = shear_modulus
        self.stiffening = stiffening
        self.volume_penalty = volume_penalty
        self.error_estimators = error_estimators
        self.displacement_space = displacement_space
        self.pressure_space = pressure_space
        self.damping = damping

    def define_fields(self):
        self.define_vector_field("displacement", self.displacement_space)
        self.define_scalar_field("pressure", self.pressure_space)

    def define_residuals(self):
        u, pressure = var("displacement"), var("pressure")
        g = grad(u)
        g00, g01, g10, g11 = g[0, 0], g[0, 1], g[1, 0], g[1, 1]
        eta = u[0] / var("coordinate_x")
        minor = g00 * g11 - g01 * g10
        # F - cof(F) and det(F) - 1 written in the small displacement gradients, so the far
        # field does not subtract two O(mu) numbers (that round-off floors the residual).
        excess = matrix([[g00 - g11 - eta - g11 * eta, g01 + g10 * (1 + eta), 0],
                         [g10 + g01 * (1 + eta), g11 - g00 - eta - g00 * eta, 0],
                         [0, 0, eta - g00 - g11 - minor]])
        cofactor = matrix([[(1 + g11) * (1 + eta), -g10 * (1 + eta), 0],
                           [-g01 * (1 + eta), (1 + g00) * (1 + eta), 0],
                           [0, 0, 1 + g00 + g11 + minor]])
        volume_change = (g00 + g11 + minor) * (1 + eta) + eta
        stress = self.shear_modulus * excess - pressure * cofactor
        if self.volume_penalty:
            # Consistent augmented-Lagrangian term: its variation vanishes when J = 1.
            # It controls volume modes that the discrete pressure space does not resolve.
            stress = stress + self.shear_modulus * self.volume_penalty * volume_change * cofactor
        if self.stiffening is not None:
            # Optional Mooney-Rivlin term beta mu / 2 (I2 - 3) - 2 beta mu (J - 1), weighted by a
            # field w(X); its stress beta mu w (I1 F - F F^T F - 2 cof F) vanishes at F = I.
            deformation = matrix([[1 + g00, g01, 0], [g10, 1 + g11, 0], [0, 0, 1 + eta]])
            first_invariant = (1 + g00) ** 2 + g01 ** 2 + g10 ** 2 + (1 + g11) ** 2 + (1 + eta) ** 2
            cubic = matproduct(deformation, matproduct(transpose(deformation), deformation))
            stress = stress + self.shear_modulus * self.stiffening * (
                first_invariant * deformation - cubic - 2 * cofactor)
        self.add_weak(stress, grad(testfunction("displacement")))
        self.add_weak(volume_change, testfunction("pressure"))
        if self.damping:
            self.add_weak(self.damping * partial_t(u), testfunction("displacement"))

    def define_error_estimators(self):
        """Z2 estimators on the displacement gradient and the hoop strain, for local refinement."""
        if not self.error_estimators:
            return
        u = var("displacement")
        g = grad(u)
        for component in (g[0, 0], g[0, 1], g[1, 0], g[1, 1], u[0] / var("coordinate_x")):
            self.add_spatial_error_estimator(component)


class DeformedSurfaceEnergy(InterfaceEquations):
    """Variation of the surface energy ``sigma`` per unit deformed area.

    With the reference unit tangent ``t0`` and ``tau = dx/dS0 = t0 + grad(u) t0``, the
    deformed area element is ``(x_r / X_r) |tau|`` times the reference one, so the
    variation is ``sigma [ |tau| du_r / X_r + (x_r / X_r) tau . d(du)/dS0 / |tau| ]``.
    """

    required_parent_type = NeoHookeanDisplacementEquations

    def __init__(self, surface_tension=1):
        super().__init__()
        self.surface_tension = surface_tension

    def define_residuals(self):
        u, test = var("displacement"), testfunction("displacement")
        normal = var("normal")
        t0 = (-normal[1], normal[0])
        g, gt = grad(u), grad(test)
        tau = vector(t0[0] + g[0, 0] * t0[0] + g[0, 1] * t0[1],
                     t0[1] + g[1, 0] * t0[0] + g[1, 1] * t0[1])
        dtau = vector(gt[0, 0] * t0[0] + gt[0, 1] * t0[1], gt[1, 0] * t0[0] + gt[1, 1] * t0[1])
        length = square_root(dot(tau, tau))
        radius = var("coordinate_x")
        hoop = 1 + u[0] / radius
        sigma = self.surface_tension
        self.add_weak(vector(sigma * length / radius, 0), test)
        self.add_weak(sigma * hoop / length * tau, dtau)


class FarFieldTraction(InterfaceEquations):
    """Traction of the linear far field, ``sigma_rr = -2 mu u_r / r`` (Lame field ``u_r ~ 1/r``)."""

    required_parent_type = NeoHookeanDisplacementEquations

    def __init__(self, shear_modulus):
        super().__init__()
        self.shear_modulus = shear_modulus

    def define_residuals(self):
        u = var("displacement")
        self.add_weak(vector(2 * self.shear_modulus * u[0] / var("coordinate_x"), 0),
                      testfunction("displacement"))


class SurfaceContinuation(InterfaceEquations):
    """Pull of the free surface beyond a truncation point: ``sigma (x_r / X_r)`` along the conormal."""

    def __init__(self, surface_tension=1):
        super().__init__()
        self.surface_tension = surface_tension

    def define_residuals(self):
        u = var("displacement")
        hoop = 1 + u[0] / var("coordinate_x")
        self.add_weak(-self.surface_tension * hoop * self.get_normal(), testfunction("displacement"))


class ReferenceSheetMesh(MeshTemplate):
    """Structured mesh of the half sheet from block grids in reference coordinates.

    ``grids`` holds the ``core``, ``lower``, ``upper`` and ``sheet`` blocks of
    :func:`reference_grids`, or any grids with the same topology (for example pull-backs of
    a mesh designed in the deformed state). Shared block edges must coincide. The free
    surface is the left column of ``lower``, the top row of ``upper`` and the top row of
    ``sheet``; its facets that lie on the reference cap arc follow the arc exactly, the rest
    are straight. Boundaries: ``interface`` (free surface), ``symmetry`` (midplane), ``far``.
    """

    def __init__(self, grids: dict[str, np.ndarray], r0: float, triangles: bool = False,
                 domain_name: str = "solid", quadratic: bool = False):
        super().__init__()
        self.grids = grids
        self.quadratic = quadratic
        self.r0 = r0
        self.triangles = triangles
        self.domain_name = domain_name

    def define_geometry(self):
        a = HALF_THICKNESS
        cx = self.r0 + a
        domain = self.new_domain(self.domain_name)
        registry: dict[tuple[float, float], int] = {}

        def node(p) -> int:
            key = (round(float(p[0]), 11), round(float(p[1]), 11))
            if key not in registry:
                registry[key] = self.add_node_unique(float(p[0]), float(p[1]))
            return registry[key]

        step = 2 if self.quadratic else 1

        def block(grid: np.ndarray) -> list[list[int]]:
            ids = [[node(grid[i, j]) for j in range(grid.shape[1])] for i in range(grid.shape[0])]
            if self.quadratic:
                if (grid.shape[0] - 1) % 2 or (grid.shape[1] - 1) % 2:
                    raise ValueError("quadratic block grids need an even number of intervals")
                for i in range(0, grid.shape[0] - 1, 2):
                    for j in range(0, grid.shape[1] - 1, 2):
                        domain.add_quad_2d_C2(*(ids[i + di][j + dj] for dj in range(3) for di in range(3)))
                return ids
            for i in range(grid.shape[0] - 1):
                for j in range(grid.shape[1] - 1):
                    n00, n10, n01, n11 = ids[i][j], ids[i + 1][j], ids[i][j + 1], ids[i + 1][j + 1]
                    if self.triangles:
                        domain.add_tri_2d_C1(n00, n10, n11)
                        domain.add_tri_2d_C1(n00, n11, n01)
                    else:
                        domain.add_quad_2d_C1(n00, n10, n01, n11)
            return ids

        tip, shoulder = (cx - a, 0.0), (cx, a)
        cap_arc = _pyoomph.CurvedEntityCircleArc([cx, 0.0, 0.0], [tip[0], tip[1], 0.0],
                                                 [shoulder[0], shoulder[1], 0.0])
        self._cap_arc = cap_arc  # keep the entity alive with the template
        core, lower, upper, sheet = (block(self.grids[k]) for k in ("core", "lower", "upper", "sheet"))

        def on_cap_arc(point) -> bool:
            return point[0] <= cx + 1e-12 and abs(math.hypot(point[0] - cx, point[1]) - a) < 1e-9

        def edge(name, nodes, points=None):
            # A free-surface facet whose nodes all lie on the reference cap arc follows the
            # arc exactly; any other facet is straight in reference coordinates.
            for k in range(0, len(nodes) - 1, step):
                facet_nodes = nodes[k:k + step + 1]
                vertices = [facet_nodes[0], facet_nodes[-1]]
                if points is not None and all(on_cap_arc(p) for p in points[k:k + step + 1]):
                    self.add_facet_to_boundary(name, facet_nodes, vertices, cap_arc)
                else:
                    self.add_facet_to_boundary(name, facet_nodes, vertices)

        g = self.grids
        edge("symmetry", [row[0] for row in lower])
        edge("symmetry", [row[0] for row in core])
        edge("interface", lower[0], g["lower"][0])
        edge("interface", [row[-1] for row in upper], g["upper"][:, -1])
        edge("symmetry", [row[0] for row in sheet])
        edge("interface", [row[-1] for row in sheet], g["sheet"][:, -1])
        edge("far", sheet[-1])


class ElastocapillaryEquilibriumProblem(Problem):
    """Axisymmetric equilibrium of the half sheet; continuation parameter ``log_Ec``."""

    def __init__(self):
        super().__init__()
        self.r0 = 1.0
        self.length = 1.0e5
        self.elements_per_quarter = 3
        self.growth = 1.1
        self.first_spacing: float | None = None
        self.max_spacing = math.inf
        self.core_fraction = 0.5
        self.triangles = False
        self.adaptive = False
        self.grids: dict[str, np.ndarray] | None = None
        self.quadratic_grids = False
        # Numerical regularisation of the crushed edge cap: Mooney-Rivlin weight beta inside
        # reference radius cap_stiffening_radius (smooth step of width cap_stiffening_width).
        self.cap_stiffening = 0.0
        self.cap_stiffening_radius = 1.5
        self.cap_stiffening_width = 0.05
        self.volume_penalty = 0.0
        self.far_edge = "robin"
        self.pressure_space = "DL"
        self.surface_tension = 1.0
        self.with_stability_mass = True

    def define_problem(self):
        if self.far_edge not in ("robin", "clamped"):
            raise ValueError("far_edge must be 'robin' or 'clamped'")
        self.set_coordinate_system("axisymmetric")
        if self.grids is None:
            self.grids = reference_grids(self.r0, self.length, self.elements_per_quarter, self.growth,
                                         self.first_spacing, self.max_spacing, self.core_fraction,
                                         quadratic=self.quadratic_grids)
        self.add_mesh(ReferenceSheetMesh(self.grids, self.r0, self.triangles,
                                         quadratic=self.quadratic_grids))
        modulus = exp(self.define_global_parameter(log_Ec=0.0))
        # The stability mass is weighted by 1/X^2 so that slow far-field modes, whose rate
        # would otherwise scale as 1/X^2, do not crowd zero; the eigenvalue signs are unchanged.
        damping = var("coordinate_x") ** -2 if self.with_stability_mass else 0
        stiffening = None
        if self.cap_stiffening > 0:
            # A global parameter, so a run can ramp it; the value set here is the initial one.
            beta = self.define_global_parameter(cap_beta=self.cap_stiffening)
            step = (var("coordinate_x") - self.cap_stiffening_radius) / self.cap_stiffening_width
            stiffening = beta / (1 + exp(step))
        eqs = NeoHookeanDisplacementEquations(modulus, displacement_space="C2TB" if self.triangles else "C2",
                                              pressure_space=self.pressure_space, damping=damping,
                                              error_estimators=self.adaptive, stiffening=stiffening,
                                              volume_penalty=self.volume_penalty)
        eqs += DeformedSurfaceEnergy(self.surface_tension) @ "interface"
        eqs += DirichletBC(displacement_y=0) @ "symmetry"
        if self.far_edge == "clamped":
            eqs += DirichletBC(displacement_x=0) @ "far"
        else:
            eqs += FarFieldTraction(modulus) @ "far"
            eqs += SurfaceContinuation(self.surface_tension) @ "interface/far"
        self.add_equations(eqs @ "solid")

    @property
    def elastocapillary_number(self) -> float:
        return math.exp(self.get_global_parameter("log_Ec").value)

    def _displacement(self, mesh_name: str):
        mesh = self.get_mesh(mesh_name)
        indices = mesh.get_nodal_field_indices()
        return mesh, indices["displacement_x"], indices["displacement_y"]

    def deformed_surface(self) -> np.ndarray:
        """Deformed free-surface nodes ``(r, z)``, ordered by reference arclength."""
        mesh, ix, iy = self._displacement("solid/interface")
        points = [(nd.x(0) + nd.value(ix), nd.x(1) + nd.value(iy), nd.x(0), nd.x(1)) for nd in mesh.nodes()]
        return np.asarray(points)

    def hole_radius(self) -> float:
        """Smallest radius of the deformed free surface (the dynamic runs' ``R``)."""
        return float(self.deformed_surface()[:, 0].min())

    def rim_height(self) -> float:
        """Full rim height: twice the largest height of the deformed free surface.

        Each quadratic surface element is rebuilt in its local coordinate so that a
        maximum between nodes counts, as in the dynamic runs.
        """
        mesh, _, iy = self._displacement("solid/interface")
        maximum = -math.inf
        for element in mesh.elements():
            points = []
            for local_node in range(element.nnode()):
                local = element.local_coordinate_of_node(local_node)
                node = element.node_pt(local_node)
                points.append((float(local[0]), float(node.x(1) + node.value(iy))))
            maximum = max(maximum, *(height for _, height in points))
            if len(points) >= 3:
                s = np.asarray([p[0] for p in points])
                h = np.asarray([p[1] for p in points])
                constant, linear, quadratic = np.polynomial.polynomial.polyfit(s, h, 2)
                if quadratic < 0:
                    stationary = -linear / (2 * quadratic)
                    if s.min() <= stationary <= s.max():
                        maximum = max(maximum, constant + linear * stationary
                                      + quadratic * stationary ** 2)
        return 2.0 * maximum

    def far_edge_displacement(self) -> float:
        """Radial displacement of the far edge at the midplane."""
        mesh, ix, _ = self._displacement("solid/far")
        bottom = min(mesh.nodes(), key=lambda nd: nd.x(1))
        return float(bottom.value(ix))
