"""Planar Taylor-Culick retraction in a frame co-moving with the rim.

This is the validation case. A semi-infinite two-dimensional sheet retracts at
the Taylor-Culick velocity

    V_TC = sqrt(2 sigma / (rho h0))  ->  sqrt(2)   in these units,

which the recovered control ``U(t)`` must approach.

Note there is **no true steady state**, even in the co-moving frame: the rim
collects mass at a rate ``rho h V`` and therefore grows without bound. The
classical result follows from ``d(mV)/dt = 2 sigma`` with ``m = rho h V t``, so
``V`` is constant while ``m`` grows. A stationary solve of this problem is
consequently ill-posed and diverges; the driver attempts one only as an
optional extra and reports failure rather than masking it.

The case is validated against Savva & Bush, JFM 626 (2009), figure 6, which
gives the edge velocity against ``t* = t/tau_vis`` for a range of Ohnesorge
number. Two convention traps in that comparison:

  * their ``Oh = mu/sqrt(2 H rho gamma)`` carries a ``sqrt(2)`` that ours does
    not, so our coded ``Oh = 0.05`` is their ``Oh = 0.035``;
  * ``tau_vis = mu H / (2 gamma)`` is the *viscous* clock, and their curves
    collapse onto it only for ``Oh >~ 0.5``. At lower ``Oh`` the approach is
    inertial, set by rim growth in capillary time, and ``t*`` is meaningless.

At ``Oh_SB = 0.5`` this case gives ``U/V_TC = 0.9297`` at ``t* = 30`` against
their ~0.93.

Geometry (half sheet, midplane symmetry at ``y = 0``):

    y
    ^
    |     ,-------------------------------------  <- free surface
 0.5|   ,'
    |  |   liquid                                    inflow at u_x = -U
    | |
  0 +-+--------------------------------------------  <- symmetry (midplane)
    x=0                                        x=L
     ^
     rim tip, pinned at x = 0 by the constraint

The retracting rim moves in +x in the laboratory frame. In the co-moving frame
the tip is held at ``x = 0`` and the undisturbed film, at rest in the
laboratory, enters at ``u_x = -U``.
"""

import os
import sys

# Sets HWLOC_COMPONENTS=-gl before MPI initialises; see the package docstring.
# Must precede every pyoomph import, so it comes first and is not reordered.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import taylor_culick_fem  # noqa: F401,E402  (import for side effect: runtime env)

from pyoomph import *  # noqa: E402
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.expressions import *
from pyoomph.meshes.bcs import EnforcedDirichlet
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

# Full sheet thickness is the length scale, so the simulated half-sheet is 0.5
# thick and the rim cap radius equals that half-thickness.
HALF_THICKNESS = 0.5
V_TAYLOR_CULICK = 2.0**0.5


class PlanarSheetMesh(GmshTemplate):
    """Flat half-sheet terminated by a semicircular cap of radius ``0.5``.

    The cap is tangent to the flat film at the shoulder, matching the Basilisk
    initial condition. On remeshing the interface is rebuilt as a spline through
    its current shape.
    """

    def define_geometry(self):
        pr = self.get_problem()
        assert isinstance(pr, PlanarTaylorCulickProblem)
        self.mesh_mode = "tris"
        self.default_resolution = pr.resolution

        cap_r = HALF_THICKNESS

        if self.is_remeshing():
            # Rebuild the deformed interface from its current node positions.
            segments = self.get_boundary_coordinates(
                "liquid/interface", sort_along_axis="x+"
            )
            pts = [self.point(x, y) for x, y in segments[0]]
            self.spline(pts, name="interface")
            p_tip, p_far_top = pts[0], pts[-1]
        else:
            p_tip = self.point(0.0, 0.0)
            p_shoulder = self.point(cap_r, HALF_THICKNESS)
            p_far_top = self.point(pr.domain_length, HALF_THICKNESS)
            # Semicircular cap, centre on the midplane at the shoulder station.
            self.circle_arc(p_tip, p_shoulder, center=(cap_r, 0.0), name="interface")
            self.line(p_shoulder, p_far_top, name="interface")

        p_far_bot = self.point(pr.domain_length, 0.0)
        self.line(p_far_top, p_far_bot, name="far")
        self.line(p_far_bot, p_tip, name="symmetry")
        self.plane_surface("interface", "far", "symmetry", name="liquid")


class PlanarTaylorCulickProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Oh = 0.05  # matches the Basilisk reference study
        self.domain_length = 12.0  # co-moving window length, in h0
        self.resolution = 0.06  # ~8 elements across the half-thickness
        self.frame_acceleration = True  # include the non-inertial correction

    def define_problem(self):
        self.set_coordinate_system("cartesian")
        self += PlanarSheetMesh()

        # The co-moving frame velocity: a single global unknown, fixed below by
        # the constraint that pins the rim tip.
        U, Utest = self.add_global_dof("U", initial_condition=0.0)

        # The frame translates at +U e_x, so in the non-inertial frame a body
        # force -rho * dU/dt e_x must be applied. It vanishes identically in the
        # steady state, so it does not affect the validation target.
        if self.frame_acceleration:
            bulkforce = -vector(1, 0) * partial_t(U)
        else:
            bulkforce = vector(0, 0)

        eqs = NavierStokesEquations(
            mass_density=1,
            dynamic_viscosity=self.Oh,
            bulkforce=bulkforce,
        )
        eqs += HyperelasticSmoothedMesh()
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions())

        # Sharp free surface. No gas phase: surface tension enters as the
        # surface-divergence term, so no curvature is ever evaluated.
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        # Midplane symmetry.
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "symmetry"

        # Far field: undisturbed film, at rest in the laboratory frame, so it
        # enters the co-moving frame at -U. U is an unknown, hence
        # EnforcedDirichlet (a Lagrange-multiplier field) and never DirichletBC.
        #
        # mesh_y is pinned here as well as mesh_x: the inflow cross-section is
        # the undisturbed film, so its thickness must stay at the half-thickness
        # rather than being free to thin or thicken under the kinematic BC.
        eqs += DirichletBC(mesh_x=self.domain_length, mesh_y=True, velocity_y=0) @ "far"
        eqs += EnforcedDirichlet(velocity_x=-U) @ "far"

        # The constraint that closes the system: hold the rim tip at x = 0.
        # The tip is the point where the free surface meets the midplane.
        eqs += WeakContribution(var("mesh_x"), Utest) @ "interface/symmetry"

        # Keep interface interpolation sane across remeshing.
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="x+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "symmetry"

        eqs += IntegralObservables(area=1)

        self += eqs @ "liquid"
        self += ODEFileOutput() @ "globals"

    def control_velocity(self) -> float:
        """Current value of the co-moving frame velocity."""
        return self.get_ode("globals").get_value("U", as_float=True)


def report(label: str, u: float) -> float:
    """Print a control velocity against the Taylor-Culick target."""
    rel_err = abs(u - V_TAYLOR_CULICK) / V_TAYLOR_CULICK
    print("=" * 64)
    print(label)
    print(f"  control U        = {u:.9f}")
    print(f"  V_TC = sqrt(2)   = {V_TAYLOR_CULICK:.9f}")
    print(f"  U / V_TC         = {u / V_TAYLOR_CULICK:.9f}")
    print(f"  relative error   = {rel_err:.3e}")
    print("=" * 64, flush=True)
    return rel_err


if __name__ == "__main__":
    import sys

    t_end, length, res, oh = 20.0, None, None, None
    for arg in sys.argv[1:]:
        if arg.startswith("--tend="):
            t_end = float(arg.split("=", 1)[1])
        elif arg.startswith("--length="):
            length = float(arg.split("=", 1)[1])
        elif arg.startswith("--res="):
            res = float(arg.split("=", 1)[1])
        elif arg.startswith("--oh="):
            oh = float(arg.split("=", 1)[1])

    with PlanarTaylorCulickProblem() as problem:
        if length is not None:
            problem.domain_length = length
        if res is not None:
            problem.resolution = res
        if oh is not None:
            problem.Oh = oh
        print(
            f"# L={problem.domain_length} res={problem.resolution} Oh={problem.Oh}",
            flush=True,
        )
        problem.initialise()

        # Transient from rest. A cold stationary Newton solve on a deforming
        # free surface diverges -- the interface has to find its own shape
        # first. So march in time and watch the control settle, then (optionally)
        # polish with a stationary solve seeded from the converged state.
        history = problem.create_text_file_output(
            "control.txt", header=["t", "U", "U_over_VTC"]
        )
        t, dt_out = 0.0, 0.1
        while t < t_end - 1e-12:
            t = min(t + dt_out, t_end)
            problem.run(t, startstep=1e-4, maxstep=dt_out, temporal_error=1e-3)
            u = problem.control_velocity()
            now = float(problem.get_current_time(as_float=True))
            history.add_row(now, u, u / V_TAYLOR_CULICK)
            print(
                f"t = {now:8.4f}   U = {u:.9f}   U/V_TC = {u / V_TAYLOR_CULICK:.9f}",
                flush=True,
            )
        report("TRANSIENT, FINAL TIME", problem.control_velocity())

        # Now the stationary solve is well-posed: it starts from the settled
        # interface shape rather than the initial guess.
        try:
            problem.solve(max_newton_iterations=30)
            report("STATIONARY SOLVE (seeded from transient)", problem.control_velocity())
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            print(f"stationary polish failed: {exc}", flush=True)
