"""Axisymmetric Taylor-Culick retraction: lab-frame DNS on a graded annulus.

Design authority: ``docs/axi-design.md``, section "2D DNS (lab-frame
axisymmetric ALE on a graded annulus)". This module is a direct structural
edit of ``planar_comoving.py``: same physics (single-phase, sharp free
surface, ``NavierStokesFreeSurface``'s surface-divergence form for capillary
traction, ``HyperelasticSmoothedMesh`` for mesh motion), but the hole opens
outward from a finite radius ``R0`` into a fixed lab frame instead of
retracting toward a co-moving tip pinned at ``x = 0``.

Why lab frame, not co-moving:

  * A radial translation of an axisymmetric geometry is not a Galilean boost
    (there is no direction to translate a point that stays fixed relative to
    the hole centre), so the co-moving trick that works for the planar sheet
    -- freeze the tip, blow the far film in at ``-U`` -- has no analogue here.
    There is consequently no global frame-velocity dof, no bulk force for
    frame acceleration, and no tip-pinning constraint. None of the recorded
    pyoomph traps (unknown-dependent ``DirichletBC``, vanishing point
    constraints under the axisymmetric ``2*pi*r`` measure) can fire, because
    none of the machinery that trips them is present.
  * The metric itself carries axisymmetry: ``set_coordinate_system("axisymmetric")``
    switches the integration measure to ``2*pi*r`` and adds the hoop terms to
    every differential operator pyoomph generates, and
    ``NavierStokesFreeSurface``'s surface-divergence form then carries BOTH
    principal curvatures of the free surface automatically. No curvature is
    ever written by hand (see ``validation/axi_smoketests.py``'s ``s0`` for
    the discriminating check).
  * The domain shrinks from the inside as the hole opens (``r`` in
    ``[R(t), R_out]``), so the annulus is graded: fine near the tip and rim,
    coarse in the quiescent far film, which only needs ~2 elements across the
    half-thickness. ``R(t)`` is a pure observable (the minimum ``mesh_x`` on
    the free surface), read the same way the planar driver reads its control
    velocity from the mesh/ODE state.

Physical scaling is unchanged from the planar case: ``rho = sigma = h0 = 1``
with ``h0`` the FULL thickness, so the half-thickness is 0.5 and
``V_TC = sqrt(2)``. ``HALF_THICKNESS`` and ``V_TAYLOR_CULICK`` are imported
from ``planar_comoving`` rather than redefined, so the two geometries can
never silently drift apart on that constant.

TRAP: THE ABSOLUTE NEWTON TOLERANCE DOES NOT SURVIVE A LARGE ``R0``.
--------------------------------------------------------------------
``newton_solver_tolerance`` is an ABSOLUTE bound on the max-norm of the
residual vector (pyoomph's default 1e-8), while ``set_coordinate_system(
"axisymmetric")`` multiplies every residual and Jacobian entry by ``2*pi*r``.
The residual magnitude of the *same* physical imbalance therefore grows
linearly with the hole radius, and the roundoff floor Newton can reach grows
with it. Past ``R0 ~ 30`` that floor crosses 1e-8 and every timestep is
rejected even though the solve is, in relative terms, fully converged.

Observed signature (the thing that sends the diagnosis in the wrong
direction): a run at ``R0 = 100`` aborts in the FIRST output interval with
Newton residuals oscillating at O(0.1) with no descent, and the adaptive
controller halving ``dt`` until it underflows ("Tried to reduce dt to
6.3e-13 ... less than the minimum dt"). That O(0.1) oscillation is not the
failure; it is the tail of the cascade. The residual trace of the FIRST
attempt, at the driver's own starting step, descends cleanly:

    dt = 1.06e-5, R0 = 100, resolution 0.25:
        160.5 -> 2.75e-6 -> 7.0e-8 -> 3.4e-8, 3.5e-8, 3.5e-8, ... (10 iters)

i.e. quadratic convergence to a floor of ~3.5e-8, then a random-sign
wander at that level, rejected only because 3.5e-8 > 1e-8. Each rejection
halves ``dt``, which stiffens the system (mass terms scale as ``1/dt``,
the algebraic free-surface/kinematic-multiplier rows do not), so the floor
RISES with every retry -- 2.5e-7, 6e-7, 1.7e-6, ... -- reaching O(0.1) by
``dt ~ 1e-13``. The oscillation therefore looks ``dt``-invariant and
structural, and is neither.

Discriminating evidence, one variable at a time. The whole annulus was
translated in ``r`` with its length, element count and every solver setting
held fixed (identical 5275 dofs, ``R_out - R0 = 31.4``, ``dt = 1.06e-5``,
``Oh_SB = 0.5``, uniform 0.25 mesh), so the ONLY difference between rows is
the radial offset:

    R0     initial residual   last residual      outcome
      1        2.09               2.8e-11          converged, 2 iters
     10       16.49               5.3e-10          converged, 2 iters
     30       48.49               7.6e-9           converged, 2 iters (marginal)
    100      160.49               3.7e-8 - 5.2e-8  never below 1e-8: FAIL

For the first three rows Newton stopped on tolerance, so the last residual
is an upper bound on the floor, not the floor itself; only the ``R0 = 100``
row, which never stops, measures a floor directly. The initial residual is
exactly linear in ``R0`` (the ``2*pi*r`` weight); the achievable residual
grows monotonically and smoothly with it and crosses the fixed 1e-8
tolerance between ``R0 = 30`` and ``R0 = 100``. A smooth onset that tracks
the coordinate offset is scaling, not a defect that switches on.

What it is NOT (each checked, at ``R0 = 100``, coarse):

  * Not a residual-generation or Jacobian error at large coordinates:
    ``debug_jacobian_by_fd_epsilon = 1e-7`` runs clean at ``R0 = 100``
    (same check that passes at ``R0 = 1``), and that same configuration
    converges to 6.1e-10 when ``dt`` is 1e-3 instead of 1e-5.
  * Not the graded mesh: the repro above is a UNIFORM 0.25 mesh
    (``far_resolution == resolution``), so grading is not in the loop.
  * Not a Newton-iteration budget: ``max_newton_iterations = 40`` sits at
    3.4e-8 for 31 further iterations; a globally convergent (line-search)
    Newton sits at 3.5e-8 for the same.
  * Not a BDF2 startup pathology: BDF1 for the first step plateaus at
    1.9e-7, i.e. worse.
  * Not fixed by a larger starting step, though the direction is
    informative: ``startstep`` 1e-1 (not 1e-2) of the first output
    interval lowers the floor to 5.9e-9 and squeaks past 1e-8 with a
    factor 1.7 of margin. That is the conditioning story again (larger
    ``dt`` -> smaller ``1/dt`` mass weights -> better-conditioned saddle
    system), and far too thin to rely on.

The stalled rows are the free-surface kinematic multipliers
(``liquid/interface/_kin_bc``) on the cap at ``r = 100.01 - 100.08``, with
the sign flipping between iterations at a fixed node -- roundoff, not a
systematic imbalance -- and NOT at the far boundary (``r = 131``), which is
why the tip radius, and not ``R_out``, sets the scale.

THE FLOOR OBEYS A LAW, AND THE FIX IS THAT LAW.
------------------------------------------------
A direct solve cannot drive a residual below roughly
``eps * ||J|| * ||u||``. Here ``||u||`` is dominated by the mesh-position
dofs, which are absolute radii ~ ``R``, and ``||J||`` by the mass rows,
which carry the measure weight ``2*pi*r ~ R`` divided by ``dt``. So

    floor ~ eps * R^2 / dt .

Measured at ``R0 = 100`` (coarse, ``eps = 2.2e-16``, so the law predicts
``2.2e-12/dt``), across three decades of ``dt`` taken from real runs:

    dt        6.8e-4   1.6e-4   6.4e-5   1.06e-5   4e-6    3.4e-7
    floor     9.2e-10  5.0e-9   4.8e-8   3.5e-8    3.6e-7  2.0e-6
    law       3.2e-9   1.4e-8   3.4e-8   2.1e-7    5.5e-7  6.5e-6

-- i.e. the law holds to a factor of ~5 either way over that range, and it
also reproduces the ``R0`` scan above at fixed ``dt``.

This matters because ``dt`` is NOT bounded below in this driver. Each output
row calls ``run()`` afresh, and the last sub-step of a row is whatever is
left over to land exactly on the output time. That leftover is generically a
fair fraction of the previous step, but occasionally a tiny sliver: at
``R0 = 100``, coarse, the SECOND output row ends in a ``dt = 3.4e-7``
fragment (0.4% of the interval), whose floor is ~2e-6. Any fixed tolerance
is therefore a lottery -- a plain ``1e-8 * R0`` (= 1e-6 at ``R0 = 100``)
carried a production-resolution run to ``t* = 25`` without complaint and
still died on output row 2 at coarse resolution.

So ``actions_before_newton_solve`` sets, for ``R0 > 1`` only:

    newton_solver_tolerance = max(1e-8, safety * eps * R0^2 / dt)

with ``safety = 10``, giving ~30x margin over the measured floor uniformly
in ``dt`` (tight, ~2e-8, on the ordinary 1e-3-scale steps; loose only on the
sliver steps, where the solution is not moving anyway). ``R0 <= 1`` returns
pyoomph's default 1e-8 for every step, so the ``R0 = 1`` campaign is
untouched by construction: verified, R and volume over the first five
outputs match a pre-fix run to ~5e-15 relative, which is this solver's own
run-to-run reproducibility (two identical post-fix runs differ by the same
amount). Rescaling the test space by ``1/R`` instead would move one of the
two factors into the residual and leave the Newton step itself identical; it
would not touch the ``1/dt`` factor, which is the one that actually kills
runs. This is the same trap, and a strict generalisation of the measured
remedy, that ``trouton1d_ale_axi.py`` records for the 1D ALE solver -- there
it showed up as a plateau at a few 1e-8 rather than an O(0.1) oscillation,
because that driver has no adaptive-``dt`` cascade to disguise it.

Limits: the scale used is ``R0``, not the live ``R(t)``, which grows by
``V_TC * t_end`` over a run (15% at ``R0 = 100, t* = 30``; a factor ~16 at
``R0 = 1``, where the tolerance is pinned to the default anyway and the
campaign demonstrably converges). A campaign in which ``R`` grows by an
order of magnitude from a large ``R0`` should re-measure rather than assume.
"""

import os

# Must precede the first pyoomph (hence MPI) import; see planar_comoving.py's
# header for why (hwloc probes a display socket that a Wayland compositor
# never completes the handshake for, hanging MPI_Init indefinitely).
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

import planar_comoving
from pyoomph import *
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions import *
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

# Double-precision unit roundoff, the prefactor of the Newton residual floor
# measured in the module docstring's TRAP section. Named rather than inlined
# because it is the one number in that estimate that is not a property of this
# problem.
_MACHINE_EPSILON = 2.2e-16


class AnnularSheetMesh(GmshTemplate):
    """Annular translate of ``planar_comoving.PlanarSheetMesh``.

    Geometry (half sheet in ``r >= 0``, midplane symmetry at ``z = 0``):

        z
        ^
        |            ,------------------------------  <- free surface ("interface")
    0.5 |          ,'
        |         |     liquid
        |        |
      0 +-------+------------------------------------  <- symmetry (midplane, z = 0)
        r=0    r=R0                              r=R_out
                 ^
                 tip, cap radius = HALF_THICKNESS, matches Basilisk IC

    The hole (``r < R(t)``) is not meshed; the liquid domain shrinks as it
    opens. On remeshing (``is_remeshing()``), the interface is rebuilt as a
    spline through its current node positions, exactly as in the planar
    template.

    Mesh grading: fine (``default_resolution``, typically 0.06 -- ~8 elements
    across the half-thickness) near the tip/rim, coarsening to
    ``far_resolution`` (default 0.25 -- ~2 elements across the half-thickness,
    matching ``docs/axi-design.md``) in the quiescent far film. Grading is
    keyed to the CURRENT tip position (``self._current_tip_x``, recomputed at
    the top of every ``define_geometry`` call, including every remesh), not
    to the original geometry points -- the remeshing branch below rebuilds
    every boundary point through a single generic loop with no per-point
    size, so a static point-size assignment alone would not survive a remesh:
    the entire rebuilt interface spline would silently revert to the fine
    default resolution all the way out to ``R_out``. The graded
    ``_mesh_size_callback`` is therefore the primary grading mechanism; the
    explicit ``size=`` on the far corner points is a redundant first-pass
    pin, consistent with the callback's coarse asymptote there.
    """

    def define_geometry(self):
        pr = self.get_problem()
        assert isinstance(pr, AxisymmetricTaylorCulickProblem)
        self.mesh_mode = "tris"
        self.default_resolution = pr.resolution

        cap_r = planar_comoving.HALF_THICKNESS

        if self.is_remeshing():
            # Rebuild the deformed interface from its current node positions.
            segments = self.get_boundary_coordinates(
                "liquid/interface", sort_along_axis="x+"
            )
            pts = [self.point(x, y) for x, y in segments[0]]
            self.spline(pts, name="interface")
            p_tip, p_far_top = pts[0], pts[-1]
            tip_x = float(segments[0][0][0])
        else:
            p_tip = self.point(pr.R0, 0.0)
            p_shoulder = self.point(pr.R0 + cap_r, planar_comoving.HALF_THICKNESS)
            p_far_top = self.point(
                pr.R_out, planar_comoving.HALF_THICKNESS, size=pr.far_resolution
            )
            # Semicircular cap, centre on the midplane at the shoulder station.
            self.circle_arc(p_tip, p_shoulder, center=(pr.R0 + cap_r, 0.0), name="interface")
            self.line(p_shoulder, p_far_top, name="interface")
            tip_x = pr.R0

        # Distance-from-tip mesh grading; recomputed fresh every remesh, so it
        # keeps tracking the rim as R(t) grows instead of freezing at t = 0.
        self._current_tip_x = tip_x
        self._mesh_size_callback = self._graded_size

        p_far_bot = self.point(pr.R_out, 0.0, size=pr.far_resolution)
        self.line(p_far_top, p_far_bot, name="far")
        self.line(p_far_bot, p_tip, name="symmetry")
        self.plane_surface("interface", "far", "symmetry", name="liquid")

    def _graded_size(self, dim, tag, x, y, z, *_ignored_lc):
        """Target element size at ``(x, y)``, graded by radial distance from the tip.

        gmsh calls this with either 5 (older) or 6 (``..., lc``) positional
        arguments; the trailing ``*_ignored_lc`` absorbs the extra one so both
        arities work (mirrors the defensive wrapping
        ``GmshSizeCallback._setup_for_mesh`` applies to its own callbacks).
        """
        pr = self.get_problem()
        dist = max(0.0, x - self._current_tip_x)
        frac = min(1.0, dist / pr.mesh_grading_length)
        return pr.resolution + (pr.far_resolution - pr.resolution) * frac


class AxisymmetricTaylorCulickProblem(Problem):
    """Lab-frame axisymmetric hole-opening DNS on a graded annulus.

    ``Oh`` is the CODED dynamic viscosity (the ``mu = sqrt(2) * Oh_SB``
    conversion, when needed, is the driver's job -- see
    ``validation/run_validation_axi.py`` -- exactly as
    ``planar_comoving.PlanarTaylorCulickProblem.Oh`` takes the coded value).
    """

    def __init__(self):
        super().__init__()
        self.Oh = 0.05  # matches the Basilisk reference study (coded, not Oh_SB)
        self.R0 = 1.0  # initial hole radius, in h0; matches the Basilisk IC
        self.R_out = 12.0  # fixed outer (lab-frame) boundary
        self.resolution = 0.06  # near-tip/rim element size, ~8 elements across HALF_THICKNESS
        self.far_resolution = 0.25  # far-film element size, ~2 elements across HALF_THICKNESS
        # Radial extent, from the current tip, over which the mesh grades from
        # `resolution` to `far_resolution`. A mesh-economy choice, not a
        # physical scale -- picked to comfortably span the near-tip curvature
        # and the collecting rim without regenerating the whole grading rule
        # every remesh (it is re-anchored at the live tip position each time
        # `define_geometry` runs, so it re-covers the rim as R(t) grows).
        self.mesh_grading_length = 6.0
        # Absolute Newton tolerance AT UNIT HOLE RADIUS: pyoomph's own default,
        # and the tolerance actually used whenever R0 <= 1. For larger R0 it is
        # the LOWER BOUND of a per-step tolerance that tracks the roundoff floor
        # of the direct solve, ~ eps*R0^2/dt (module docstring, TRAP section).
        self.newton_tolerance_at_unit_radius = 1e-8
        # Margin between that tolerance and the measured floor. 10 gives ~30x
        # on the measurements in the docstring, uniformly in dt.
        self.newton_tolerance_safety = 10.0
        # Toggle for the s2 volume-conservation smoke test's with/without-remesh
        # comparison (docs/axi-design.md's smoke ladder). Production runs leave
        # this at its default; RemeshWhen is what keeps the graded annulus valid
        # as the tip moves.
        self.remesh_active = True

    def define_problem(self):
        # The metric switch: everything else (2*pi*r measure, hoop terms in
        # every operator, both principal curvatures in the free-surface term)
        # follows from this one call. See docs/axi-design.md and
        # pyoomph.expressions.coordsys.AxisymmetricCoordinateSystem.
        self.set_coordinate_system("axisymmetric")

        # ...and its one unavoidable consequence for the solver, applied once
        # here so that even a stationary solve starts from the right bound;
        # actions_before_newton_solve() re-tunes it per timestep.
        self.newton_solver_tolerance = self.newton_tolerance_at_unit_radius

        self += AnnularSheetMesh()

        # No bulkforce: a radial translation is not a Galilean boost, so there
        # is no frame acceleration to correct for (contrast
        # PlanarTaylorCulickProblem, which feeds -rho*dU/dt through bulkforce).
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.Oh)
        eqs += HyperelasticSmoothedMesh()  # default cartesian mesh metric -- deliberate
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions(active=self.remesh_active))

        # Sharp free surface. No gas phase, so surface tension enters only as
        # the surface-divergence term; no curvature is ever evaluated by hand,
        # and under the axisymmetric measure this term carries both principal
        # curvatures (verified by validation/axi_smoketests.py's s0).
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        # Midplane symmetry, unchanged from planar.
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "symmetry"

        # Far boundary: fixed lab-frame wall, quiescent undisturbed film. All
        # values are plain (R_out is a python float, not an unknown), so this
        # is DirichletBC, never EnforcedDirichlet -- there is no global dof
        # for it to depend on. No -U inflow: the film really is at rest here.
        eqs += DirichletBC(mesh_x=self.R_out, mesh_y=True, velocity_x=0, velocity_y=0) @ "far"

        # Keep interface interpolation sane across remeshing.
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="x+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "symmetry"

        # Exact global volume conservation (no flux through any boundary: the
        # far wall is impermeable and quiescent, the symmetry line is a
        # reflection plane) is a free, sharp correctness diagnostic.
        eqs += IntegralObservables(volume=1)

        self += eqs @ "liquid"

    def actions_before_newton_solve(self):
        """Re-tune the absolute Newton tolerance to this step's roundoff floor.

        pyoomph's ``newton_solver_tolerance`` is an absolute bound on the
        max-norm of a residual that this problem scales by ``2*pi*r`` (measure)
        and by ``1/dt`` (mass rows), so the floor a direct solve can reach is
        ``~ eps*R0^2/dt`` -- measured, and tabulated in the module docstring's
        TRAP section. A tolerance below that floor rejects every timestep at a
        solution that is fully converged in relative terms, and the adaptive
        controller then halves ``dt``, which RAISES the floor, until ``dt``
        underflows. Tracking the floor per step is what makes an arbitrarily
        small output-landing sliver survivable.

        ``R0 <= 1`` returns pyoomph's default unchanged, so nothing about the
        R0 = 1 campaign moves.
        """
        super().actions_before_newton_solve()
        base = self.newton_tolerance_at_unit_radius
        if self.R0 <= 1.0:
            self.newton_solver_tolerance = base
            return
        # dt(0) is the step being taken; it is zero/absent for a stationary
        # solve, where the 1/dt amplification does not exist.
        try:
            dt = float(self.time_pt().dt(0))
        except Exception:  # noqa: BLE001 -- no timestepper yet: stationary solve
            dt = 0.0
        if not dt > 0.0:
            self.newton_solver_tolerance = base
            return
        floor = _MACHINE_EPSILON * self.R0 * self.R0 / dt
        self.newton_solver_tolerance = max(base, self.newton_tolerance_safety * floor)

    def hole_radius(self) -> float:
        """Current hole edge radius R(t): the minimum radial position on the free surface.

        Reads live node positions on the "interface" boundary (the tip is,
        by construction, its smallest-``r`` node -- the cap curves away from
        r=R0 in +r for every other point on it) rather than tracking a single
        node index across mesh rebuilds, mirroring how the planar driver reads
        its control velocity from mesh/ODE state rather than caching it.
        """
        mesh = self.get_mesh("liquid/interface")
        return min(n.x(0) for n in mesh.nodes())

    def volume(self) -> float:
        """Current liquid volume, from the exact-conservation observable."""
        return float(self.get_mesh("liquid").evaluate_observable("volume"))
