"""1D Trouton (slender-sheet) retraction, solved in pyoomph.

WHY A SECOND 1D SOLVER. ``validation/trouton1d.py`` integrates the same model
with scipy's BDF on a graded mass-coordinate grid. Its equations are right --
the mass-coordinate transform, the ``4 mu d_m(h^2 u_m)`` viscous term, the full
curvature and the Savva & Bush tip closure all check out by hand -- but the
discretisation does not scale. Grading puts the tip cell at ~1e-3 of the domain,
the viscous operator's stiffness rises as 1/dm^2, and scipy then rebuilds a
FINITE-DIFFERENCE Jacobian on 2n unknowns every few steps. n = 400 completes;
n = 1200 did not finish a t* = 30 run in four hours. A solver that cannot be
resolution-converged cannot answer the question it was written to answer,
whatever its equations say.

pyoomph removes the actual bottleneck rather than tuning around it: it
differentiates the residual symbolically, so the Jacobian is exact and assembled
analytically, and it brings the adaptive time stepping already in use for the 2D
case. Sharing the framework also sharpens the comparison -- if this and the 2D
FEM disagree, the difference is the dimensional reduction, not the integrator.

THE MODEL (Savva & Bush 2009 eqs 2.1-2.2), planar sheet of FULL thickness h:

    u_t + u u_x = (4 mu / (rho h)) (h u_x)_x + (gamma / rho) kappa_x
    h_t + (h u)_x = 0
    kappa = (1/2) h_xx / [1 + (1/4) h_x^2]^{3/2}

The 1/2 and 1/4 are because the interface sits at z = +-h/2 while h is the full
thickness; kappa is the curvature of the SURFACE, not of h.

LAGRANGIAN MASS COORDINATE. With m = int h dx' a material label, the tip sits at
m = 0 and the domain [0, M] is FIXED -- no moving mesh, no remeshing, and the
free boundary is a coordinate rather than something to track. With d/dx = h d/dm:

    h_t|_m = -h^2 u_m
    u_t|_m = 4 mu d_m(h^2 u_m) + h kappa_m

MIXED FORMULATION. The curvature carries h_xx, a second derivative, which C0
Lagrange elements cannot represent. Rather than reach for C1 elements, p = h_m
is carried as its own field, so every derivative in the residual is first order:

    h_x  = h p,        h_xx = h (p^2 + h p_m)

giving four scalar fields (h, u, p, kappa) on a 1D line mesh.

TIP CLOSURE. Savva & Bush close the tip not by resolving the cap but with their
eq. 2.28, a prescribed axial force at the free end. In our units that is

    h^2 u_m |_{m=0} = -gamma / (2 mu)

which is exactly the NATURAL boundary condition of the conservative viscous
operator -- it enters as a boundary term in the weak form, not as a constraint.
Verified against their non-dimensionalisation: with x = Oh_SB H x*, h = H h*,
u = u_c u*, their [h* u*_x*] = -1/4 maps to -1/(2 mu) only if Oh_SB = mu/sqrt(2)
rather than mu. Using our Oh gives a spurious factor of sqrt(2).
"""

from __future__ import annotations

import math

from pyoomph import DirichletBC, Equations, InitialCondition, Problem, TextFileOutput
from pyoomph.expressions import grad, partial_t, testfunction, var, weak
from pyoomph.meshes.simplemeshes import LineMesh

V_TC = math.sqrt(2.0)


class TroutonSheet(Equations):
    """Slender-sheet retraction in the Lagrangian mass coordinate."""

    def __init__(self, mu: float, space: str = "C2", with_capillarity: bool = True,
                 staggered: bool = True):
        super().__init__()
        self.mu = mu
        self.with_capillarity = with_capillarity
        self.staggered = staggered
        self.space = space
        if staggered and with_capillarity:
            # p = h_m is meaningless for a cell-constant h: grad(h) is zero
            # inside every element. A staggered capillary formulation needs the
            # curvature built from f = h^2 as Savva & Bush do (their eq. A1),
            # which is not implemented here. Fail loudly rather than silently
            # integrating a zero curvature.
            raise ValueError(
                "staggered=True does not support with_capillarity=True: a D0 "
                "thickness has zero elementwise gradient, so p = h_m is "
                "identically zero. Use staggered=False, or implement the "
                "f = h^2 curvature of Savva & Bush eq. (A1).")
        # gamma = 1 in code units; this is Savva & Bush eq. 2.28.
        self.tip_flux = -1.0 / (2.0 * mu)

    def define_fields(self):
        # STAGGERED IS THE DEFAULT, AND IT IS WHAT MAKES THIS SOLVER CONVERGE.
        # h cell-constant ("D0") against u on vertices ("C1") is the finite
        # element realisation of the staggered mesh Savva & Bush use (their
        # figure 22: f at nodes, u at midpoints), and they say why (Appendix,
        # p. 238): "Implementation with a non-staggered grid introduced spurious
        # oscillations in the curvature in the long time limit."
        #
        # The co-located C2/C2 arrangement is exactly the one they warn against,
        # and it is NOT merely less accurate -- it is non-convergent. Measured at
        # Oh_SB = 1, the co-located scheme's failure time is a power law in the
        # tip cell size, t*_fail = 5561 dm1^0.434 over five meshes spanning a
        # 64x range in dm1 and a 5.7x range in domain length, so a FINER mesh
        # fails SOONER and there is no mesh on which it converges. Staggered runs
        # to t* = 400 on a mesh where every co-located variant dies at 166-182,
        # at a quarter of the degrees of freedom.
        #
        # It agrees with the co-located answer where both exist: maximum relative
        # difference in u_tip of 7.1e-5 across all 183 common output rows, and
        # u*(t*=31.4) = 0.9806746 against 0.9806744, so it reproduces the
        # Richardson limit u*(30) -> 0.98073 recorded in docs/README.md.
        #
        # THE VELOCITY SPACE IS THE DISCRIMINATOR, NOT THE THICKNESS SPACE. A
        # variant sweep at N=1600 settles this: with u on C2, the failure time is
        # 174-182 whether h is C2, C1, D0, D1 or D2 (five spaces, 4800-12802
        # dof); with u on C1 it does not fail at all. The parasitic mode lives on
        # the MIDSIDE NODES OF THE QUADRATIC VELOCITY FIELD, and h merely
        # inherits it through h_t = -h^2 u_m. The h checkerboard visible in the
        # profiles is the print of the mode, not its source -- which is why every
        # earlier fix aimed at h, or at the p/kappa block, changed nothing.
        if self.staggered:
            self.define_scalar_field("h", "D0")
            self.define_scalar_field("u", "C1")
        else:
            self.define_scalar_field("h", self.space)
            self.define_scalar_field("u", self.space)
        # p AND kappa ARE ONLY UNKNOWNS WHEN THE CAPILLARY TERM USES THEM.
        # With with_capillarity=False, kappa never enters the momentum residual
        # and p is used only inside kappa_expr, so both are a one-way downstream
        # projection that cannot influence h or u -- yet they sat in the same
        # Newton system and set the convergence test. That is what killed every
        # production run: the residual went bit-identical across Newton steps
        # 7-10 and stayed unchanged while the controller halved dt forty times
        # to 9.3e-13. A residual invariant under dt -> 0 cannot be in a row
        # containing partial_t, so the stall was in the algebraic (p, kappa)
        # rows, not in the physics.
        #
        # Why those rows are unsolvable here: on the failure profiles kappa
        # reaches 8.6e3 to 4.9e5 as a sign-alternating node-to-node mode where
        # the physical curvature is O(1/h) ~ 0.016, and d(kappa_expr)/dp is
        # 6.3e5 to 2.4e6 against a kappa-row diagonal of O(dm) = 1.9e-4. A
        # 1e9-1e10 row/column scaling disparity inside an otherwise O(1) system
        # puts the Newton update below the accuracy floor of the linear solve,
        # which is why raising max_newton_iterations to 40 and switching on a
        # globally convergent line search both bought nothing -- line search
        # cannot rescue an update that is itself inaccurate.
        if self.with_capillarity:
            self.define_scalar_field("p", self.space)      # p = dh/dm
            self.define_scalar_field("kappa", self.space)

    def define_residuals(self):
        h, u = var("h"), var("u")
        ht, ut = testfunction("h"), testfunction("u")

        # h_m is only meaningful for a continuous h; see the capillarity guard.
        h_m = grad(h)[0] if not self.staggered else None
        u_m = grad(u)[0]

        # 1. mass:      h_t = -h^2 u_m
        self.add_residual(weak(partial_t(h) + h * h * u_m, ht))

        # 2. momentum:  u_t = 4 mu d_m(h^2 u_m) + h kappa_m
        #    The viscous term is integrated by parts, which is what turns the
        #    tip closure into a natural boundary term. The far end contributes
        #    nothing because u is pinned there, so its test function vanishes.
        self.add_residual(weak(partial_t(u), ut))
        self.add_residual(4.0 * self.mu * weak(h * h * u_m, grad(ut)[0]))

        if self.with_capillarity:
            p, kap = var("p"), var("kappa")
            pt, kt = testfunction("p"), testfunction("kappa")
            p_m, kap_m = grad(p)[0], grad(kap)[0]

            # Surface curvature via d/dx = h d/dm, with p standing in for h_m.
            h_x = h * p
            h_xx = h * (p * p + h * p_m)
            kappa_expr = 0.5 * h_xx / (1.0 + 0.25 * h_x * h_x) ** 1.5

            self.add_residual(-weak(h * kap_m, ut))
            # 3. auxiliary: p = h_m
            self.add_residual(weak(p - h_m, pt))
            # 4. curvature, projected onto the same space
            self.add_residual(weak(kap - kappa_expr, kt))


class TipFlux(Equations):
    """Savva & Bush eq. 2.28 applied at m = 0 as a natural condition.

    SIGN, derived rather than guessed. The momentum residual is

        R(v) = int u_t v  +  4 mu int h^2 u_m v_m  -  4 mu [h^2 u_m v]_0^M
               -  int h kappa_m v

    and the bracket expands as F(M)v(M) - F(0)v(0). The far end is Dirichlet so
    v(M) = 0 and only the tip survives, contributing

        -4 mu * ( -F(0) v(0) )  =  +4 mu F(0) v(0)
                                =  4 mu * (-1/(2 mu)) * v(0)  =  -2 v(0)

    so the term enters the residual as -2 v, NOT +2 v. It is exactly -2 for
    every Oh: the mu in the viscous prefactor cancels the mu in the closure.

    Getting this backwards pushes the tip outward instead of pulling it in. The
    sheet then thickens without bound at m = 0, and Newton fails to converge at
    ANY timestep -- the adaptive controller halves dt from the start step down
    through the 1e-12 floor and aborts, with a dt that is independent of the
    mesh, which is what gave the failure away.
    """

    def __init__(self, mu: float):
        super().__init__()
        self.value = 4.0 * mu * (-1.0 / (2.0 * mu))   # = -2, independent of mu

    def define_residuals(self):
        self.add_residual(self.value * weak(1, testfunction("u")))


class Trouton1DProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Oh_SB = 1.0
        self.mass = 120.0          # sheet mass per unit width, the domain [0, M]
        self.nelem = 400
        self.grade = 2.0           # tip clustering; 1 = uniform
        # DEFAULT OFF, and this is physics rather than convenience. Savva &
        # Bush close the tip with their eq. 2.28, which IS the capillary pull of
        # the massless cap. Adding a bulk kappa_x term on top double-counts the
        # capillarity, and the mixed-formulation kappa is a numerical second
        # derivative on a graded mesh, so it is large and noisy exactly where the
        # closure already acts. With it on, u*(t*=30) = 0.106 and the early-time
        # asymptote misses by RMS 1.34; with it off, 0.95180 and RMS 5.8e-4.
        # Keep the switch: it is how that was established, and it is the knob to
        # revisit if a properly regularised curvature is ever wanted.
        self.with_capillarity = False
        # See TroutonSheet.define_fields for the evidence. Co-located C2/C2 is
        # retained only to reproduce runs made before 2026-08-11.
        self.staggered = True
        # BDF1, NOT pyoomph's default BDF2. BDF2 fails at the third timestep,
        # every time, on this problem: the fields are healthy (h = 1.000005,
        # u_tip = 3e-3, nothing negative), Newton converges to 1e-10, and it
        # still collapses dt to the 1e-12 floor. BDF1 runs the same case to
        # t* = 1 and reproduces the exact tip growth h = 1 + t/(2 mu) to four
        # decimals. The suspected cause is BDF2's two-level history against the
        # sqrt(t) start-up singularity of an impulsively loaded tip, but the
        # switch is justified empirically, not by that explanation.
        self.default_timestepping_scheme = "BDF1"

    @property
    def mu(self) -> float:
        return self.Oh_SB * math.sqrt(2.0)

    @property
    def tau_vis(self) -> float:
        return self.mu / 2.0

    def define_problem(self):
        self.set_coordinate_system("cartesian")

        # LineMesh is UNIFORM. The tip cell would be M/N in mass -- 0.6 for the
        # default M = 120, N = 200 -- while the impulsive start puts a diffusive
        # boundary layer at m = 0 that is arbitrarily thin as t -> 0+. An
        # unresolved layer there does not merely lose accuracy: the adaptive
        # timestepper cannot meet its error tolerance on a feature the mesh
        # cannot represent, so dt collapses until it trips the 1e-12 floor and
        # the solve aborts. The nodes are therefore remapped onto a graded
        # distribution after construction, clustering at the tip.
        mesh = LineMesh(N=self.nelem, size=self.mass,
                        nodal_dimension=1, name="sheet")
        self += mesh

        eqs = TroutonSheet(mu=self.mu, with_capillarity=self.with_capillarity,
                           staggered=self.staggered)
        # Must match define_fields: pyoomph raises if an initial condition names
        # a field the element does not carry.
        if self.with_capillarity:
            eqs += InitialCondition(h=1, u=0, p=0, kappa=0)
        else:
            eqs += InitialCondition(h=1, u=0)
        eqs += TipFlux(mu=self.mu) @ "left"
        # Far end: undisturbed sheet. Valid only while the disturbance has not
        # arrived, which the driver checks by monitoring h and u there.
        # A "D0" thickness has no nodal value at the boundary to pin, so only u
        # is constrained there. That is also closer to Savva & Bush, whose far
        # field is Neumann on both thickness and velocity (p. 237); the
        # undisturbed state is maintained by the domain being long enough, which
        # far_field_drift checks explicitly.
        if self.staggered:
            eqs += DirichletBC(u=0) @ "right"
        else:
            eqs += DirichletBC(h=1, u=0) @ "right"
        eqs += TextFileOutput()

        self += eqs @ "sheet"

    def initialise(self, *args, **kwargs):
        """Build the problem, then grade the mesh.

        Grading cannot happen inside define_problem: the mesh does not exist
        until initialisation runs. Doing it afterwards is only safe because the
        initial condition is UNIFORM (h = 1, u = 0) -- moving nodes under a
        non-uniform field would silently reinterpolate it.
        """
        super().initialise(*args, **kwargs)
        self._grade_mesh_at_tip()

    def _grade_mesh_at_tip(self) -> None:
        """Remap nodes onto m = M (xi)^grade, clustering at the tip.

        Both the Eulerian and Lagrangian coordinates are set. In this
        formulation the mesh coordinate IS the material label, so the two must
        agree -- leaving x_lagr on the uniform distribution would silently give
        every element the wrong measure.
        """
        if self.grade == 1.0:
            return
        mesh = self.get_mesh("sheet")
        M = float(self.mass)
        for node in mesh.nodes():
            xi = node.x(0) / M
            m_new = M * xi ** self.grade
            node.set_x(0, m_new)
            node.set_x_lagr(0, m_new)

    def _tip_node_and_indices(self):
        mesh = self.get_mesh("sheet")
        # get_nodal_field_indices maps names to positions. has_interface_dof_id
        # is for INTERFACE dofs and returns -1 for a bulk field, which is not an
        # error the caller would notice -- index -1 silently reads the last
        # value on the node.
        idx = mesh.get_nodal_field_indices()
        return min(mesh.nodes(), key=lambda nd: nd.x(0)), idx

    def tip_velocity(self) -> float:
        """u at m = 0, the retracting tip."""
        node, idx = self._tip_node_and_indices()
        return float(node.value(idx["u"]))

    def tip_thickness(self) -> float:
        """h at the tip. Returns NaN in the staggered scheme.

        A "D0" thickness is elementwise-constant and carries no nodal value, so
        there is nothing to read at m = 0. NaN rather than a guess: the exact
        result h_tip = 1 + t*/4 is available analytically (Savva & Bush eq. 2.29,
        which this code reproduces exactly in the co-located scheme), so a caller
        that needs it should use that rather than have this method invent one.
        """
        if self.staggered:
            return float("nan")
        node, idx = self._tip_node_and_indices()
        return float(node.value(idx["h"]))

    def far_field_drift(self) -> tuple[float, float]:
        """|h - 1| and |u| at m = M.

        The far end is pinned, so this is a check that the DOMAIN is long
        enough, not that the boundary condition applied: a disturbance arriving
        at a clamped inlet corrupts the answer silently.
        """
        mesh = self.get_mesh("sheet")
        idx = mesh.get_nodal_field_indices()
        node = max(mesh.nodes(), key=lambda nd: nd.x(0))
        # In the staggered scheme h is elementwise-constant with no nodal value,
        # and the far field is Neumann rather than pinned, so only the velocity
        # drift is available. It is the more sensitive of the two anyway: the
        # disturbance arrives as motion before it measurably thins the sheet.
        dh = float("nan") if self.staggered else abs(float(node.value(idx["h"])) - 1.0)
        return dh, abs(float(node.value(idx["u"])))
