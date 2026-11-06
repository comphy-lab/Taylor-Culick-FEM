"""Axisymmetric Taylor-Culick retraction: co-moving WINDOW DNS (semi-infinite sheet).

Design authority: ``the axisymmetric design note (axi-design.md)``, the paragraph at the end of the "2D
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
    ``far_field="membrane"`` is an opt-in alternative for elastic films,
    whose far field is an outgoing pulse that reaches the edge: the edge then
    takes its radial velocity and thickness from a coupled thin-film model
    (``taylor_culick_fem.farfield_membrane``) through global parameters (see
    ``define_problem``). ``far_field="relaxing_membrane"`` sets the same edge
    conditions from a finite-relaxation exterior
    (``taylor_culick_fem.relaxing_farfield_membrane``). The default ``"rest"`` is
    the formulation above, unchanged.
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

import itertools
import math

import numpy as np
from pyoomph import *
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions import *

from taylor_culick_fem import planar_comoving
from taylor_culick_fem.axisymmetric_lab import _MACHINE_EPSILON, AnnularSheetMesh

try:
    # pyoomph >= 65a4e87c moved generic boundary equations here.
    from pyoomph.equations.generic import EnforcedBC
except ImportError:  # compatibility with the older develop pin
    from pyoomph.meshes.bcs import EnforcedBC
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

INITIAL_GUESSES = ("previous", "extrapolate")


def extrapolated_guess(level1, level2, dt: float, dt_previous: float):
    """Linear extrapolation ``level1 + (dt/dt_previous) (level1 - level2)`` of two dof vectors.

    Returns None when the levels differ in length or are not finite, so the caller keeps
    the default guess.
    """
    level1 = np.asarray(level1, dtype=float)
    level2 = np.asarray(level2, dtype=float)
    if level1.shape != level2.shape or not (np.all(np.isfinite(level1)) and np.all(np.isfinite(level2))):
        return None
    return level1 + (dt / dt_previous) * (level1 - level2)


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
        # Opt-in graded layer normal to the hole-facing surface; None keeps the mesh unchanged.
        layer = getattr(pr, "face_layer", None)

        if self.is_remeshing():
            segments = self.get_boundary_coordinates(
                "liquid/interface", sort_along_axis="x+"
            )
            pts = [self.point(x, y) for x, y in segments[0]]
            if layer is None:
                self.spline(pts, name="interface")
            else:
                # The hole-facing part of the free surface, from the tip to the rim top,
                # carries the face layer. gmsh closes a boundary layer cleanly only against
                # straight curves (see _face_layer_regions), so the first surface segment
                # beyond the face end is a line: there the slope is below
                # FACE_END_SLOPE_DEGREES, and the chord stays within ~2e-4 of the spline.
                end = face_end_index(segments[0])
                face = self.spline(pts[: end + 1], name="interface")
                film = [self.line(pts[end], pts[end + 1], name="interface"),
                        self.spline(pts[end + 1 :], name="interface")]
                p_end = pts[end]
                face_xy = [(float(x), float(y)) for x, y in segments[0][: end + 1]]
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
            face = self.circle_arc(
                p_tip, p_shoulder, center=(pr.X0 + cap_r, 0.0), name="interface"
            )
            film = [self.line(p_shoulder, p_far_top, name="interface")]
            p_end = p_shoulder
            face_xy = [(pr.X0 + cap_r - cap_r * math.cos(a), cap_r * math.sin(a))
                       for a in np.linspace(0.0, 0.5 * math.pi, 65)]
            tip_x = pr.X0
            far_x = pr.R_out

        # Distance-from-tip grading, re-anchored at the live tip every remesh
        # (inherited implementation).
        self._current_tip_x = tip_x
        self._mesh_size_callback = self._graded_size

        p_far_bot = self.point(far_x, 0.0, size=pr.far_resolution)
        far = self.line(p_far_top, p_far_bot, name="far")
        if layer is None:
            self.line(p_far_bot, p_tip, name="symmetry")
            self.plane_surface("interface", "far", "symmetry", name="liquid")
        else:
            self._face_layer_regions(layer, face, film, far, p_tip, p_end, face_xy, p_far_bot)

    def _face_layer_regions(self, layer, face, film, far, p_tip, p_end, face_xy, p_far_bot):
        """Two regions split at the face end, and wall-normal layers grown from the face.

        The layers have first height ``spacing``, geometric ratio ``growth`` and total
        ``thickness``; their tangential size is the face's 1D mesh. gmsh (4.15) closes a 2D
        boundary layer cleanly only at an end listed in its ``PointsList`` whose other curves
        are straight: the end column is then laid along that line and its graded nodes are
        shared with the mesh beyond it. A layer end without that
        leaves the unstructured mesher to join a column of first height ``spacing`` to a
        neighbourhood sized ``resolution``, and it produces flat triangles (interior angles to
        179.9 degrees at spacing 2.5e-4). Both ends are therefore made straight-line ends:

        - at the tip the face meets the symmetry axis at a right angle, and the end column
          lies on the axis;
        - at the face end a straight, unnamed internal line runs down to the axis and splits
          the window into the rim region and the film region; the end column lies on it.

        The size callback then grades four things (_face_layer_size), each at the layer's own
        rate ``growth - 1`` per unit distance, never above the tip rule:

        - the film region and its curves, from ``spacing`` at the face end. Otherwise the
          film side of the internal line joins its graded layer nodes to a neighbourhood sized
          by the tip rule (flat triangles to 156 degrees on a formed rim at spacing 2.5e-4);
        - the face, whose tangential spacing is capped at ``sqrt(4 rho spacing)`` with rho the
          face height (the cap radius 1/2 initially, the rim radius later). The sagitta of a
          boundary segment of length l on a radius rho is l^2 / (8 rho), so the cap keeps it
          below half the first layer. Under the tip rule a rim of radius 4 has face segments
          of ~0.15, whose sagitta exceeds a 2.5e-4 layer and distorts the curved elements
          (minimum scaled Jacobian 0.07);
        - the rim region, from that cap at the face, so the layer top meets a matching size;
        - the face and the rim region near the face end, from the tip ``resolution``: the
          last layer columns are then as wide as those at the tip, and the rim side of the
          internal line meets sizes that match its graded nodes (else 148 degrees on a rim
          of radius 4 at spacing 1e-3).

        (A gmsh Threshold field would not do this: gmsh passes its size to the callback,
        which ignores it.)
        """
        end_xy = face_xy[-1]
        p_foot = self.point(end_xy[0], 0.0)
        cut = self.line(p_end, p_foot)
        axis_rim = self.line(p_foot, p_tip, name="symmetry")
        axis_film = self.line(p_far_bot, p_foot, name="symmetry")
        rim_regions = self.plane_surface(face, cut, axis_rim, name="liquid")
        film_regions = self.plane_surface(*film, far, axis_film, cut, name="liquid")
        spacing, growth = float(layer["spacing"]), float(layer["growth"])
        self._face_grading = {
            "end": end_xy, "spacing": spacing, "rate": growth - 1.0,
            "resolution": float(self.get_problem().resolution),
            "face": face_polyline(face_xy, 0.01),
            "face_cap": math.sqrt(4.0 * max(end_xy[1], planar_comoving.HALF_THICKNESS) * spacing),
            "film": {(1, c._id) for c in (*film, cut)} | {(2, s._id) for s in film_regions},
            "rim": {(1, face._id)} | {(2, s._id) for s in rim_regions},
        }
        self._mesh_size_callback = self._face_layer_size
        field = self.add_mesh_size_field(
            "BoundaryLayer", CurvesList=[face], PointsList=[p_tip, p_end],
            Size=float(layer["spacing"]), Ratio=float(layer["growth"]),
            Thickness=float(layer["thickness"]), Quads=0,
        )
        self.set_mesh_size_boundary_layer_field(field)

    def _face_layer_size(self, dim, tag, x, y, z, *_ignored_lc):
        """The tip rule, graded down towards the face (see _face_layer_regions)."""
        size = self._graded_size(dim, tag, x, y, z)
        g = self._face_grading
        end_x, end_y = g["end"]
        to_end = math.hypot(x - end_x, y - end_y)
        if (dim, tag) in g["film"]:
            size = min(size, g["spacing"] + g["rate"] * to_end)
        elif (dim, tag) in g["rim"]:
            face = g["face"]
            to_face = float(np.min(np.hypot(face[:, 0] - x, face[:, 1] - y)))
            size = min(size, g["face_cap"] + g["rate"] * to_face,
                       g["resolution"] + g["rate"] * to_end)
        return size


def face_polyline(points, max_gap: float) -> np.ndarray:
    """The polyline through ``points``, with segments longer than ``max_gap`` subdivided."""
    pts = np.asarray(points, dtype=float)
    out = [pts[:1]]
    for a, b in itertools.pairwise(pts):
        n = max(1, math.ceil(float(np.hypot(*(b - a))) / max_gap))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


# Largest chord slope, in degrees from horizontal, at which the hole-facing surface ends.
FACE_END_SLOPE_DEGREES = 1.0


def face_end_index(points, max_slope_degrees: float = FACE_END_SLOPE_DEGREES) -> int:
    """Index where the hole-facing surface ends, on a free surface ordered from the tip.

    From the tip on the midplane the face rises until the surface turns horizontal. Its end is
    the first point whose outgoing chord is within ``max_slope_degrees`` of horizontal, or
    descends: the rim top once a rim has formed, and the end of the cap before. A height rule
    ("the highest point") is ill-posed before a rim forms: the flat film then carries bumps of
    order 1e-6 whose maximum can lie anywhere along it. The index is kept between 2 and
    ``len(points) - 3``, so the face has two segments and the film beyond its straight first
    segment keeps a spline of at least two points.
    """
    pts = np.asarray([(float(p[0]), float(p[1])) for p in points])
    if len(pts) < 5:
        raise ValueError("the free surface needs at least five points to split")
    chords = np.diff(pts, axis=0)
    slopes = np.degrees(np.arctan2(chords[:, 1], chords[:, 0]))
    flat = np.flatnonzero(slopes[2:] < max_slope_degrees)
    index = int(flat[0]) + 2 if flat.size else len(pts) - 3
    return min(index, len(pts) - 3)


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
        # Opt-in graded mesh layer normal to the hole-facing surface: a mapping with
        # ``spacing`` (first layer), ``growth`` (ratio) and ``thickness``. None: no layer.
        self.face_layer: dict | None = None
        self.newton_tolerance_at_unit_radius = 1e-8
        self.newton_tolerance_safety = 10.0
        # Largest dt used in the round-off floor below. The eps*R^2/dt law was tuned at
        # dt <= 0.025; at the much larger adaptive steps the true residual floor stops
        # shrinking with dt, so a cap keeps the tolerance reachable. None: no cap.
        self.newton_floor_max_dt: float | None = None
        # Newton's starting point at each time step. "previous" (oomph-lib's default) starts
        # from the last solution; "extrapolate" starts from the linear extrapolation of the
        # last two, which keeps Newton convergent at much larger adaptive steps.
        self.initial_guess = "previous"
        # Which radius enters the eps*R^2/dt floor: the window's outer edge
        # (X + L, the largest radius in the mesh) or the tip alone (X). See the
        # module docstring's Newton-floor section.
        self.newton_scale_uses_window = True
        self.remesh_active = True
        # "rest": the swallowed film is at rest in the laboratory. "membrane":
        # its radial velocity and thickness are global parameters, updated by
        # the driver from the coupled far-field membrane between intervals.
        self.far_field = "rest"
        # Membrane edge data within an output interval: "hold" applies the interval-end
        # state for the whole interval (first order in the interval length); "linear"
        # moves from the start state to the end state in time (second order). Linear
        # updates the same global parameters before every Newton solve, so the state
        # file, and restarts from earlier checkpoints, are unchanged.
        self.far_field_interpolation = "hold"
        self._far_field_ramp: dict | None = None
        # pyoomph re-applies every Dirichlet value before each Newton solve and writes
        # it to time levels 0 AND 1, evaluated at each level's time. A parameter-driven
        # value has no time dependence, so the previous step's edge value is replaced by
        # the current one and BDF2 sees the wrong rate of change at the far edge.
        # "keep" restores level 1 of those far-edge dofs after the update; "overwrite"
        # is pyoomph's behaviour, kept as the default so existing runs reproduce.
        self.far_field_history = "overwrite"

    def set_far_field_ramp(self, start_time: float, end_time: float, values: dict) -> None:
        """Move the named global parameters linearly from their start to their end
        values over [start_time, end_time]; ``values`` maps a name to (start, end)."""

        if not end_time > start_time:
            raise ValueError("far-field ramp needs an end after its start")
        self._far_field_ramp = {"start": float(start_time), "end": float(end_time),
                                "values": dict(values)}
        self._apply_far_field_ramp(float(start_time))

    def _apply_far_field_ramp(self, time: float) -> None:
        ramp = self._far_field_ramp
        if ramp is None:
            return
        fraction = (time - ramp["start"]) / (ramp["end"] - ramp["start"])
        weight = min(1.0, max(0.0, fraction))
        for name, (start, end) in ramp["values"].items():
            self.get_global_parameter(name).value = start + weight * (end - start)

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
        if self.far_field == "rest":
            eqs += DirichletBC(mesh_y=True, velocity_x=0, velocity_y=0) @ "far"
        elif self.far_field in ("membrane", "relaxing_membrane"):
            # A thin film in plane motion: u_r uniform across the thickness,
            # shear-free (velocity_y natural), and the thickness set by the far
            # field through the corner height; edge nodes slide vertically
            # under the mesh equations. The values are parameters updated
            # between output intervals, never unknowns, so DirichletBC is legal.
            far_velocity = self.define_global_parameter(far_field_velocity=0.0)
            half = self.define_global_parameter(far_field_half_thickness=0.5)
            if self.far_field_interpolation not in ("hold", "linear"):
                raise ValueError("far_field_interpolation must be 'hold' or 'linear'")
            if self.far_field_history not in ("overwrite", "keep"):
                raise ValueError("far_field_history must be 'overwrite' or 'keep'")
            eqs += DirichletBC(velocity_x=far_velocity) @ "far"
            eqs += DirichletBC(mesh_y=half) @ "interface/far"
        else:
            raise ValueError("far_field must be 'rest', 'membrane' or 'relaxing_membrane', "
                             f"not {self.far_field!r}")
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
        if self._far_field_ramp is not None:
            # The step's new time level is already set: evaluate the edge there, before
            # the Dirichlet values are applied.
            self._apply_far_field_ramp(float(self.get_current_time(as_float=True)))
        kept = self._far_edge_history() if self.far_field_history == "keep" else []
        super().actions_before_newton_solve()
        self._restore_far_edge_history(kept)
        if self.initial_guess == "extrapolate":
            self._extrapolate_initial_guess()
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
        floor_dt = dt if self.newton_floor_max_dt is None else min(dt, self.newton_floor_max_dt)
        floor = _MACHINE_EPSILON * scale * scale / floor_dt
        self.newton_solver_tolerance = max(base, self.newton_tolerance_safety * floor)

    def _extrapolate_initial_guess(self) -> None:
        """Start Newton from ``u_n + (dt/dt_prev) (u_n - u_n-1)`` instead of ``u_n``.

        After the solve's shift, history level 1 holds ``u_n`` and level 2 ``u_n-1``, and the
        current values are still ``u_n``. Only the unknowns change: Dirichlet data, including the
        far edge set just above, are not in the dof vector. The converged solution is unchanged;
        what changes is whether Newton reaches it. From ``u_n`` the first residual is dominated by
        the volume constraint, about the window inflow times dt, and above dt ~ 0.3 the first
        updates diverge; from the extrapolation it is O(dt^2). A step without two history levels
        (no timestepper yet, a stationary solve) keeps the default guess.
        """
        try:
            dt = float(self.time_pt().dt(0))
            dt_previous = float(self.time_pt().dt(1))
        except Exception:  # noqa: BLE001 -- no timestepper yet
            return
        if not (dt > 0.0 and dt_previous > 0.0):
            return
        guess = extrapolated_guess(self.get_history_dofs(1), self.get_history_dofs(2), dt, dt_previous)
        if guess is not None:
            self.set_current_dofs(guess)

    #: Nodal fields that the membrane edge prescribes through global parameters.
    far_edge_history_fields: tuple[str, ...] = ("velocity_x",)

    def _far_edge_history(self) -> list:
        """Level-1 values of the pinned far-edge dofs that the Dirichlet update rewrites.

        Covers the prescribed nodal fields and the pinned corner height (``mesh_y``).
        Constant Dirichlet values on the same nodes are saved too; restoring them is a
        no-op. Empty before the first time step (no history level 1 yet).
        """
        if self.far_field != "membrane":
            return []
        try:
            mesh = self.get_mesh("liquid")
            nodes = list(mesh.boundary_nodes("far"))
        except Exception:  # noqa: BLE001 -- mesh not built yet
            return []
        if not nodes or nodes[0].ntstorage() < 2:
            return []
        indices = mesh.get_nodal_field_indices()
        fields = [indices[name] for name in self.far_edge_history_fields if name in indices]
        kept = []
        for node in nodes:
            for index in fields:
                if node.is_pinned(index):
                    kept.append((node, index, None, node.value_at_t(1, index)))
            if node.position_is_pinned(1):
                kept.append((node, None, 1, node.x_at_t(1, 1)))
        return kept

    @staticmethod
    def _restore_far_edge_history(kept: list) -> None:
        """Put back the level-1 values saved by ``_far_edge_history``."""
        for node, index, coordinate, value in kept:
            if coordinate is None:
                node.set_value_at_t(1, index, value)
            else:
                node.set_x_at_t(1, coordinate, value)

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

    def midplane_radial_velocity(self, radius: float) -> float:
        """Radial velocity on the midplane at ``radius``, linearly interpolated.

        Reads nodal values on ``"symmetry"``: the far-field membrane is driven
        with this velocity inside the window.
        """

        index = self.get_mesh("liquid").get_nodal_field_indices()["velocity_x"]
        samples = sorted(
            (float(n.x(0)), float(n.value(index)))
            for n in self.get_mesh("liquid/symmetry").nodes()
        )
        radii = np.asarray([item[0] for item in samples])
        if not radii[0] <= radius <= radii[-1]:
            raise ValueError(f"probe radius {radius} lies outside the window")
        return float(np.interp(radius, radii, [item[1] for item in samples]))

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

    def expected_volume(
        self, volume0: float, x0: float | None = None, far_outflow: float = 0.0
    ) -> float:
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

        With ``far_field="membrane"`` the edge film has thickness ``h_e`` and
        moves at ``u_e``, so the inflow is ``pi*(X+L)*h_e*(Xdot - u_e)``.
        ``far_outflow`` is the caller's running integral of
        ``(X+L)*(h_e*u_e - (h_e-1)*Xdot)`` dt, subtracted as
        ``pi*far_outflow``; it is zero for ``"rest"``.
        """
        x_start = self.X0 if x0 is None else x0
        outer_now = self.window_position() + self.window_length
        outer_0 = x_start + self.window_length
        return volume0 + 0.5 * math.pi * (outer_now**2 - outer_0**2) - math.pi * far_outflow
