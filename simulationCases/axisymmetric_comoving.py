"""Axisymmetric Taylor-Culick retraction: co-moving WINDOW DNS (semi-infinite sheet).

Design authority: ``docs/axi-design.md``, the paragraph at the end of the "2D
DNS" section describing a moving window ``[X(t), X(t) + L]`` with one global
bookkeeping dof ``X`` slaved to the tip. That paragraph was written as a
fallback; this module promotes it to a first-class sibling of
``axisymmetric_lab.py``, because the two answer different questions:

  * ``axisymmetric_lab.py`` puts a FIXED wall at ``R_out``, so ``R_out`` must
    contain the entire trajectory (``R0 + V_TC*t_end`` plus a disturbance
    footprint). ``t_end`` is therefore bounded by the mesh budget: doubling
    ``t*`` grows the annulus, and with it the far-film element count.
  * This module keeps the WINDOW LENGTH ``L`` fixed and drags it outward with
    the rim, so the dof count is independent of ``t_end``. That is the proper
    semi-infinite-sheet formulation: the film ahead of the rim is unbounded,
    and the run length is limited only by wall clock.

WHAT IS AND IS NOT CO-MOVING HERE
---------------------------------
This is an ALE *window translation*, not a change of variables. The
coordinate system stays ``axisymmetric`` and the mesh coordinates carry the
TRUE radius, exactly as in the lab case, so the ``2*pi*r`` measure and the
hoop terms of every operator see the physical ``r``. Velocities are LAB-frame
throughout. Consequently:

  * there is **no bulkforce**. A radial translation is not a Galilean boost,
    so the planar case's ``-rho*partial_t(U)`` non-inertial correction has no
    axisymmetric analogue and must never be ported. Nothing in this module is
    solved in a moving frame; only the *outer boundary of the meshed region*
    moves.
  * the far boundary is a genuine inflow: as the window sweeps outward it
    swallows quiescent film. In the laboratory frame that film is at rest
    (``velocity_x = velocity_y = 0``), which is exactly the undisturbed
    semi-infinite far field. Contrast the planar co-moving case, where the
    far film enters at ``-U`` because THERE the frame really does move.
  * volume is therefore **not** conserved (the lab case's sharpest
    diagnostic). It is replaced by an exact modified check -- see
    ``expected_volume()`` below, which is just as sharp.

THE LAGRANGE-MULTIPLIER PATTERN, AND ITS TWO AXISYMMETRIC CORRECTIONS
---------------------------------------------------------------------
The planar case closes its system with

    WeakContribution(var("mesh_x"), Utest) @ "interface/symmetry"

i.e. a point constraint at the tip whose test function belongs to a global
dof. This module uses the same device -- one global dof, one point constraint
at ``interface/symmetry``, the constrained quantity fed back into a boundary
condition through ``EnforcedBC`` -- with two changes that are load-bearing:

(a) The residual is ``mesh_x - X``, not ``mesh_x``. The planar form works
    because ``U`` does not appear in it: the row reads "tip is at x = 0" and
    ``U`` is determined only through the far-field coupling. Writing the same
    thing here (``weak(mesh_x, Xtest)``, enforcing ``mesh_x_tip = 0``) would
    be wrong twice over -- the tip is at ``r ~ R0 > 0``, and under the
    ``2*pi*r`` point measure the constraint ``2*pi*r_tip*mesh_x_tip = 0`` is a
    DOUBLE ROOT in ``mesh_x_tip``, whose derivative vanishes at the solution,
    giving a singular Jacobian row. ``mesh_x - X`` has ``dResidual/dX = -1``
    identically, so the row is well posed everywhere, and it says the true
    thing: ``X`` IS the hole radius, not a bookkeeping offset.

(b) ``coordinate_system=cartesian``. The recorded trap (``AGENTS.md``) is that
    a point contribution on the symmetry AXIS vanishes because the measure
    carries ``2*pi*r`` with ``r = 0``. That exact failure cannot fire here --
    the tip sits at ``r ~ R0``, not at ``r = 0`` -- but the same factor is
    still present, and it multiplies this row by ``2*pi*R(t)``, which GROWS
    without bound over a long window run. The pin is a geometric statement
    ("this node is at this radius"), not an integral of anything, so it is
    written in the cartesian measure and its row keeps an O(1) scale for the
    whole run.

``X`` then drags the far boundary through

    EnforcedBC(mesh_x = var("mesh_x") - (X + L)) @ "far"

-- ``EnforcedBC`` and never ``DirichletBC``, because the prescribed position
depends on an unknown (the same trap ``trouton1d_ale.py`` records for its
kinematic condition, and the reason planar's inflow uses
``EnforcedDirichlet``). ``EnforcedBC`` takes its argument in RESIDUAL form:
the expression is driven to zero, so ``mesh_x - (X + L)`` and not ``X + L``.
The remaining far-field conditions (``mesh_y``, both velocity components) are
plain constants and stay ordinary ``DirichletBC``.

TRAP: THE FAR/INTERFACE CORNER NEEDS ITS KINEMATIC MULTIPLIER PINNED.
----------------------------------------------------------------------
Replacing the lab case's ``DirichletBC(mesh_x=R_out, ...)`` by an
``EnforcedBC`` makes the Jacobian EXACTLY singular, and the symptom points
nowhere near the cause: PARDISO reports a backward error of ~1e47, the first
Newton step returns ``inf``, every timestep is rejected and the run dies with
"Tried to reduce dt to 7.5e-13 ... less than the minimum dt" -- which reads
like the ``dt``-underflow cascade ``axisymmetric_lab.py`` documents for the
Newton floor, and is a completely different failure. ``debug_jacobian_by_fd_
epsilon`` does NOT catch it: the assembled Jacobian is correct, it is the
system that is rank-deficient.

The cause is one node. At the corner where ``"far"`` meets ``"interface"``,
``mesh_y`` is Dirichlet-pinned and ``mesh_x`` is now a free unknown -- and TWO
Lagrange multipliers act on that single free dof: the free surface's kinematic
multiplier ``_kin_bc``, and the ``EnforcedBC``'s ``_lagr_enf_bc_mesh_x``. Two
constraints on one unknown is a null space. The lab case never sees it because
a plain ``DirichletBC`` REMOVES the position dofs, and pyoomph then deals with
the orphaned kinematic multiplier itself.

Measured, at ``X0 = 1``, ``L = 40``, ``resolution`` 0.4, ``dt = 1e-4``
(2117 dofs, no zero rows or columns -- the deficiency is structural, not a
missing entry). Singular values of the assembled Jacobian, smallest first:

    as written above, without the corner pin:   4.8e-13, 2.3e-12, 4.6e-4, ...
    with DirichletBC(_kin_bc=0) @ "interface/far":        2.3e-12, 4.6e-4, ...
    lab case, same mesh, R_out = 41 (CONTROL):            2.3e-12, 4.7e-4, ...

The 4.8e-13 mode is entirely ``liquid/interface/_kin_bc``, dominated by a
single node with an exponentially decaying tail along the interface -- the
signature of one redundant multiplier, not of a physical soft mode. The
2.3e-12 mode that survives is NOT introduced here: the lab case, which runs
this campaign to ``t* = 30`` without complaint, carries the identical mode
(``interface/mesh_y`` uniform along the flat film with two
``interface/velocity_x`` partners) at the same magnitude against a largest
singular value of 5.3e3. Pinning the corner multiplier therefore restores the
working case's conditioning exactly, and nothing more.

Physically the pin says what the far boundary already says: the inflow
cross-section is undisturbed film, so its surface node is not free to move
under the kinematic condition. That is the same statement as planar's
``mesh_y=True`` at its inflow, applied to the multiplier the axisymmetric
window leaves behind.

THE RETRACTION SPEED IS A SOLVED DOF, NOT A FINITE DIFFERENCE
--------------------------------------------------------------
A second global dof ``U`` carries ``partial_t(X) - U = 0``. The lab case has
no such dof and its driver differences ``R`` between output rows, which is
forced there: ``R`` is read off the mesh (``min mesh_x`` on the interface),
and there is no unknown holding it. Here ``X`` already IS an unknown, so its
rate is available at no extra cost beyond one scalar equation, differentiated
by the same BDF2 scheme as everything else rather than by a first-order
difference across a geometrically stretching output schedule. The driver logs
BOTH (``U`` from the dof, ``Rdot`` by difference); they agree to the
time-discretisation error, which makes their difference a free accuracy
monitor on the output cadence.

THE NEWTON FLOOR, WITH THE LIVE RADIUS
---------------------------------------
``axisymmetric_lab.py``'s TRAP section is the reference for why an ABSOLUTE
``newton_solver_tolerance`` cannot survive a large radius: the residual
carries ``2*pi*r`` from the measure and ``1/dt`` from the mass rows, so the
roundoff floor of a direct solve goes like ``eps * R^2 / dt``. That module
scales with ``R0`` and notes the limitation explicitly ("the scale used is
``R0``, not the live ``R(t)``... a campaign in which ``R`` grows by an order
of magnitude from a large ``R0`` should re-measure rather than assume").

This module is exactly that campaign: ``R`` grows without bound by
construction, which is the whole point of a window. The scale is therefore
read from the LIVE value of the global dof ``X`` every step, and the radius
used is the largest one present in the mesh, ``X + L`` (the far boundary),
since the vector norms in the floor estimate are global. With the default
``L = 40`` the resulting tolerance is ``max(1e-8, 10 * eps * (X+L)^2 / dt)``,
which for ordinary steps (``dt >~ 4e-4``) is simply pyoomph's default 1e-8
and only loosens on the small output-landing slivers the driver's schedule
produces. ``newton_scale_uses_window`` = False falls back to the tip radius
``X`` alone, which the lab module's own measurements support better (its
``R0 = 1``, ``R_out = 32`` row reached 2.8e-11, i.e. the ``R0`` law, not the
``R_out`` law); it is provided so the choice can be measured rather than
argued.

Physical scaling is unchanged: ``rho = sigma = h0 = 1`` on the FULL thickness,
half-thickness 0.5, ``V_TC = sqrt(2)``. Both constants are imported from
``planar_comoving`` so the three geometries cannot drift apart on them.
"""

import os

# Must precede the first pyoomph (hence MPI) import; see planar_comoving.py's
# header for why (hwloc probes a display socket that a Wayland compositor
# never completes the handshake for, hanging MPI_Init indefinitely).
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

import math

import numpy as np
import planar_comoving
from axisymmetric_lab import _MACHINE_EPSILON, AnnularSheetMesh
from pyoomph import *
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions import *

try:
    # pyoomph >= 65a4e87c moved generic boundary equations here.
    from pyoomph.equations.generic import EnforcedBC
except ImportError:  # compatibility with the older develop pin
    from pyoomph.meshes.bcs import EnforcedBC
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)


class ComovingAnnularSheetMesh(AnnularSheetMesh):
    """``AnnularSheetMesh`` whose outer edge follows the window, not the lab.

    Only one thing changes relative to the parent, and it only shows up on a
    REMESH. The parent rebuilds the interface from its current node positions
    (so its far-top corner already tracks the moved boundary) but then places
    the far-bottom corner at the problem's fixed ``R_out``. That is right for
    a lab-frame annulus and wrong here: after the window has translated, the
    two corners of the ``"far"`` edge would sit at different radii, shearing
    the outflow boundary into a slanted line and, worse, silently changing the
    window length. The far radius is therefore read from the live geometry --
    the last interface node, which by construction is the top corner of the
    ``"far"`` edge -- rather than from a stored constant.

    The initial (non-remeshing) mesh is the parent's unchanged, with
    ``pr.R_out`` supplied as a property equal to ``X0 + L``; the graded size
    callback is inherited as-is and re-anchors on the live tip every remesh.
    """

    def define_geometry(self):
        pr = self.get_problem()
        assert isinstance(pr, AxisymmetricComovingProblem)
        self.mesh_mode = "tris"
        self.default_resolution = pr.resolution

        cap_r = planar_comoving.HALF_THICKNESS

        if self.is_remeshing():
            segments = self.get_boundary_coordinates(
                "liquid/interface", sort_along_axis="x+"
            )
            pts = [self.point(x, y) for x, y in segments[0]]
            self.spline(pts, name="interface")
            p_tip, p_far_top = pts[0], pts[-1]
            tip_x = float(segments[0][0][0])
            # The live outer radius: X(t) + L, read off the geometry rather
            # than off a constant (see the class docstring).
            far_x = float(segments[0][-1][0])
        else:
            p_tip = self.point(pr.X0, 0.0)
            p_shoulder = self.point(pr.X0 + cap_r, planar_comoving.HALF_THICKNESS)
            p_far_top = self.point(
                pr.R_out, planar_comoving.HALF_THICKNESS, size=pr.far_resolution
            )
            self.circle_arc(
                p_tip, p_shoulder, center=(pr.X0 + cap_r, 0.0), name="interface"
            )
            self.line(p_shoulder, p_far_top, name="interface")
            tip_x = pr.X0
            far_x = pr.R_out

        # Distance-from-tip grading, re-anchored at the live tip every remesh
        # (inherited implementation).
        self._current_tip_x = tip_x
        self._mesh_size_callback = self._graded_size

        p_far_bot = self.point(far_x, 0.0, size=pr.far_resolution)
        self.line(p_far_top, p_far_bot, name="far")
        self.line(p_far_bot, p_tip, name="symmetry")
        self.plane_surface("interface", "far", "symmetry", name="liquid")


class AxisymmetricComovingProblem(Problem):
    """Axisymmetric hole opening in a fixed-length window dragged by the rim.

    ``self.mu`` is the CODED dynamic viscosity handed to
    ``NavierStokesEquations``. The project's Ohnesorge number is derived from
    it as ``Oh = mu / sqrt(2)``; performing that conversion is the driver's
    job, matching both sibling cases.
    """

    def __init__(self):
        super().__init__()
        self.mu = 0.05  # coded dynamic viscosity; Oh = mu/sqrt(2) = 0.035
        self.X0 = 1.0  # initial hole radius; matches the Basilisk IC
        self.window_length = 40.0  # L: fixed radial extent of the window, in h0
        self.resolution = 0.06  # near-tip element size, ~8 across HALF_THICKNESS
        self.far_resolution = 0.25  # far-film element size, ~2 across HALF_THICKNESS
        self.mesh_grading_length = 6.0  # inherited grading rule; see AnnularSheetMesh
        self.newton_tolerance_at_unit_radius = 1e-8
        self.newton_tolerance_safety = 10.0
        # Which radius enters the eps*R^2/dt floor: the window's outer edge
        # (X + L, the largest radius in the mesh) or the tip alone (X). See the
        # module docstring's Newton-floor section.
        self.newton_scale_uses_window = True
        self.remesh_active = True

    @property
    def R_out(self) -> float:
        """Initial outer radius of the window; only the t = 0 mesh uses it.

        Named ``R_out`` so the inherited ``AnnularSheetMesh`` machinery (grading
        callback, corner sizes) needs no change. It is NOT a wall: after t = 0
        the outer edge is wherever ``X(t) + L`` is, enforced weakly.
        """
        return self.X0 + self.window_length

    def define_problem(self):
        # The metric switch. The mesh coordinates are TRUE radii, exactly as in
        # the lab case; the window translation is ALE, not a change of variables.
        self.set_coordinate_system("axisymmetric")
        self.newton_solver_tolerance = self.newton_tolerance_at_unit_radius

        self += ComovingAnnularSheetMesh()

        # X: the hole radius, as a global unknown. Its own equation is the tip
        # pin below. U: its rate, so dR/dt is solved rather than differenced.
        X, Xtest = self.add_global_dof("X", initial_condition=self.X0)
        # U's own residual is partial_t(X) - U, so U is referred to inside its
        # own equation_contribution by name; nothing else uses the returned
        # handles, hence the underscore names.
        _U, _Utest = self.add_global_dof(
            "U",
            equation_contribution=partial_t(X) - var("U", domain="globals"),
            initial_condition=0.0,
        )

        # No bulkforce: see the module docstring. The frame does not move; only
        # the meshed window does, and its motion is ALE, already accounted for
        # by partial_t(..., ALE="auto") once mesh equations are present.
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.mu)
        eqs += HyperelasticSmoothedMesh()  # default cartesian mesh metric -- deliberate
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions(active=self.remesh_active))

        # Sharp free surface; the surface-divergence form carries both
        # principal curvatures under the axisymmetric measure.
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        # Midplane symmetry, unchanged from both siblings.
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "symmetry"

        # Far edge of the window. mesh_x follows the unknown X, so it must go
        # through a Lagrange-multiplier field (EnforcedBC), in RESIDUAL form.
        eqs += EnforcedBC(mesh_x=var("mesh_x") - (X + self.window_length)) @ "far"
        # ...everything else there is a constant: the film being swallowed is
        # undisturbed and at rest in the laboratory frame, and its thickness is
        # held at the half-thickness rather than left free under the kinematic
        # condition (same reasoning as planar's pinned inflow cross-section).
        eqs += DirichletBC(mesh_y=True, velocity_x=0, velocity_y=0) @ "far"
        # TRAP (measured, see the module docstring's CORNER section): the
        # free-surface kinematic multiplier at the far/interface CORNER must be
        # pinned, or the Jacobian is exactly singular.
        eqs += DirichletBC(_kin_bc=0) @ "interface/far"

        # THE CLOSING CONSTRAINT: the tip node's radius IS X. Residual form
        # mesh_x - X (not planar's bare mesh_x -- double root here), cartesian
        # measure (not scaled by the growing 2*pi*R). See the module docstring.
        eqs += WeakContribution(
            var("mesh_x") - X, Xtest, coordinate_system=cartesian
        ) @ "interface/symmetry"

        # Keep interface interpolation sane across remeshing.
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="x+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "symmetry"

        # Volume is NOT conserved here (the window swallows film through
        # "far"); expected_volume() below is the exact replacement check.
        eqs += IntegralObservables(volume=1)

        self += eqs @ "liquid"
        self += ODEFileOutput() @ "globals"

    def actions_before_newton_solve(self):
        """Track this step's roundoff floor with the LIVE window radius.

        Same law and the same reason as ``axisymmetric_lab.py``'s TRAP section
        (``floor ~ eps*R^2/dt``), but the radius is read from the global dof
        every step instead of frozen at ``R0``, because here ``R`` grows
        without bound by construction. Before the ODE domain exists (the very
        first setup) the initial value ``X0`` is used.
        """
        super().actions_before_newton_solve()
        base = self.newton_tolerance_at_unit_radius
        try:
            dt = float(self.time_pt().dt(0))
        except Exception:  # noqa: BLE001 -- no timestepper yet: stationary solve
            dt = 0.0
        if not dt > 0.0:
            self.newton_solver_tolerance = base
            return
        scale = self.window_position()
        if self.newton_scale_uses_window:
            scale += self.window_length
        floor = _MACHINE_EPSILON * scale * scale / dt
        self.newton_solver_tolerance = max(base, self.newton_tolerance_safety * floor)

    # ---------------------------------------------------------------- readers

    def window_position(self) -> float:
        """Current value of the global dof X (the hole radius the pin enforces)."""
        try:
            return float(self.get_ode("globals").get_value("X", as_float=True))
        except Exception:  # noqa: BLE001 -- ODE domain not built yet
            return float(self.X0)

    def retraction_speed(self) -> float:
        """Current value of the global dof U = dX/dt, solved, not differenced."""
        try:
            return float(self.get_ode("globals").get_value("U", as_float=True))
        except Exception:  # noqa: BLE001
            return 0.0

    def hole_radius(self) -> float:
        """R(t) read off the mesh, identically to ``axisymmetric_lab.py``.

        Kept alongside ``window_position()`` deliberately: the two are equal to
        the accuracy with which the pin is satisfied, so their difference is a
        direct residual check on the constraint, and it also makes the
        comparison against the lab case an apples-to-apples one (same
        estimator, same definition of "the tip").
        """
        mesh = self.get_mesh("liquid/interface")
        return min(n.x(0) for n in mesh.nodes())

    def rim_height(self) -> float:
        """Return full rim height from the quadratic free-surface geometry.

        Interface ``nodes()`` includes midside nodes, but a quadratic
        coordinate can attain its maximum between all three nodes.  Rebuild
        each element's sheet-normal coordinate in its local coordinate and
        include any interior stationary point.  The mesh covers the upper half
        of a unit-full-thickness sheet, hence the factor two.
        """

        interface = self.get_mesh("liquid/interface")
        maximum = -math.inf
        for element in interface.elements():
            coordinates = []
            for local_node in range(element.nnode()):
                local = element.local_coordinate_of_node(local_node)
                if len(local) != 1:
                    raise RuntimeError("interface element is not one-dimensional")
                coordinates.append((float(local[0]), float(element.node_pt(local_node).x(1))))
            if not coordinates:
                continue
            maximum = max(maximum, *(height for _, height in coordinates))
            if len(coordinates) >= 3:
                local = np.asarray([item[0] for item in coordinates], dtype=float)
                height = np.asarray([item[1] for item in coordinates], dtype=float)
                constant, linear, quadratic = np.polynomial.polynomial.polyfit(
                    local, height, 2
                )
                if abs(quadratic) > np.finfo(float).eps * max(1.0, abs(linear)):
                    stationary = -linear / (2.0 * quadratic)
                    if local.min() < stationary < local.max():
                        maximum = max(
                            maximum,
                            float(
                                np.polynomial.polynomial.polyval(
                                    stationary, [constant, linear, quadratic]
                                )
                            ),
                        )
        if not math.isfinite(maximum):
            raise RuntimeError("interface contains no finite geometry")
        return 2.0 * maximum

    def volume(self) -> float:
        """Current liquid volume of the meshed window (half sheet, z >= 0)."""
        return float(self.get_mesh("liquid").evaluate_observable("volume"))

    def expected_volume(self, volume0: float, x0: float | None = None) -> float:
        """Volume the window MUST hold, given how far it has swept outward.

        The lab case's exact global volume conservation is unavailable here by
        design: the ``"far"`` edge is an inflow. It is replaced by an equally
        sharp identity, because the inflow is completely known -- the film
        entering is undisturbed (half-thickness 0.5, at rest in the lab frame),
        and the boundary moves at ``dX/dt``:

            dV/dt = (mesh speed) x (area of the "far" edge)
                  = Xdot * 2*pi*(X+L)*HALF_THICKNESS
                  = pi*(X+L)*Xdot ,

        which integrates in closed form -- no time integration is needed, and
        no error accumulates from the output cadence:

            V(t) = V(0) + (pi/2) * [ (X(t)+L)^2 - (X0+L)^2 ] .

        A drift in ``V - V_expected`` is therefore exactly as diagnostic as a
        drift in ``V`` is for the lab case: it indicts the ALE mass balance,
        the ``EnforcedBC`` on the far edge, or the free-surface kinematics.
        """
        x_start = self.X0 if x0 is None else x0
        outer_now = self.window_position() + self.window_length
        outer_0 = x_start + self.window_length
        return volume0 + 0.5 * math.pi * (outer_now**2 - outer_0**2)
