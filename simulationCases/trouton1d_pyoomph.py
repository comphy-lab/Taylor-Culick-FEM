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

    def __init__(self, mu: float, space: str = "C2", with_capillarity: bool = True):
        super().__init__()
        self.mu = mu
        self.with_capillarity = with_capillarity
        self.space = space
        # gamma = 1 in code units; this is Savva & Bush eq. 2.28.
        self.tip_flux = -1.0 / (2.0 * mu)

    def define_fields(self):
        self.define_scalar_field("h", self.space)
        self.define_scalar_field("u", self.space)
        self.define_scalar_field("p", self.space)      # p = dh/dm
        self.define_scalar_field("kappa", self.space)

    def define_residuals(self):
        h, u, p, kap = var("h"), var("u"), var("p"), var("kappa")
        ht, ut, pt, kt = (testfunction(n) for n in ("h", "u", "p", "kappa"))

        h_m = grad(h)[0]
        u_m = grad(u)[0]
        p_m = grad(p)[0]
        kap_m = grad(kap)[0]

        # Surface curvature via d/dx = h d/dm, with p standing in for h_m.
        h_x = h * p
        h_xx = h * (p * p + h * p_m)
        kappa_expr = 0.5 * h_xx / (1.0 + 0.25 * h_x * h_x) ** 1.5

        # 1. mass:      h_t = -h^2 u_m
        self.add_residual(weak(partial_t(h) + h * h * u_m, ht))

        # 2. momentum:  u_t = 4 mu d_m(h^2 u_m) + h kappa_m
        #    The viscous term is integrated by parts, which is what turns the
        #    tip closure into a natural boundary term. The far end contributes
        #    nothing because u is pinned there, so its test function vanishes.
        self.add_residual(weak(partial_t(u), ut))
        self.add_residual(4.0 * self.mu * weak(h * h * u_m, grad(ut)[0]))
        if self.with_capillarity:
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

        eqs = TroutonSheet(mu=self.mu, with_capillarity=self.with_capillarity)
        eqs += InitialCondition(h=1, u=0, p=0, kappa=0)
        eqs += TipFlux(mu=self.mu) @ "left"
        # Far end: undisturbed sheet. Valid only while the disturbance has not
        # arrived, which the driver checks by monitoring h and u there.
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
        return abs(float(node.value(idx["h"])) - 1.0), abs(float(node.value(idx["u"])))
