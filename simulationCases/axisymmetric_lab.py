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
