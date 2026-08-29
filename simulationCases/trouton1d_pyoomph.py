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

THE CHANGE OF VARIABLE f = h^2 (Savva & Bush p. 236, and their eq. A1).
------------------------------------------------------------------------
They do not solve for h. They solve for f = h^2, and they say why: near the tip
the profile is a parabola on its side, h ~ sqrt(r - r0(t)), so h_r is INFINITE
at r = r0. Every h-derivative in the curvature is then unrepresentable exactly
where the tip closure acts. f = h^2 is linear in (r - r0) there, so f_r is
finite and the curvature becomes a ratio of finite quantities.

DERIVATION, PLANAR, SURFACE AT z = +-h/2.

(a) Curvature. With h = f^{1/2},

        h_x  = f_x / (2 f^{1/2})
        h_xx = f_xx / (2 f^{1/2}) - f_x^2 / (4 f^{3/2})
             = (2 f f_xx - f_x^2) / (4 f^{3/2})

    Substituting into kappa = (1/2) h_xx / [1 + (1/4) h_x^2]^{3/2} and using
    1 + (1/4) h_x^2 = 1 + f_x^2/(16 f) = (16 f + f_x^2) / (16 f):

        kappa = (2 f f_xx - f_x^2) / (8 f^{3/2})  *  (16 f)^{3/2} / (16 f + f_x^2)^{3/2}
              = 8 (2 f f_xx - f_x^2) / (16 f + f_x^2)^{3/2}

    which is EXACTLY the first term of their eq. (A 1); their second term,
    f_r / [r (16 f + f_r^2)^{1/2}], is the azimuthal curvature of the
    axisymmetric sheet and has no planar counterpart. That the two agree term
    for term is the check that our 1/2 and 1/4 factor convention -- full
    thickness h, interface at +-h/2 -- is the same as theirs.

(b) Mass. h_t = -h^2 u_m and f = h^2 give f_t = 2 h h_t = -2 h^3 u_m, i.e.

        f_t = -2 f^{3/2} u_m

(c) Momentum. h^2 = f exactly, so the viscous operator is

        4 mu d_m(h^2 u_m) = 4 mu d_m(f u_m)

    which is LINEAR in f. That is much of the point: in the h-form the same
    term is quadratic, so its Jacobian carries an 8 mu h u_m d_m(.) block that
    the f-form simply does not have. The capillary term carries the remaining
    square root, u_t = ... + f^{1/2} kappa_m.

(d) Tip closure. h^2 u_m|_{m=0} = -1/(2 mu) becomes f u_m|_{m=0} = -1/(2 mu),
    the same number, and it is still the natural boundary term of the same
    conservative operator. TipFlux is therefore unchanged.

(e) Curvature in the mass coordinate. With d/dx = h d/dm = f^{1/2} d/dm,

        f_x  = f^{1/2} f_m
        f_xx = f^{1/2} d_m(f^{1/2} f_m) = f_m^2 / 2 + f f_mm

    so 2 f f_xx - f_x^2 = f f_m^2 + 2 f^2 f_mm - f f_m^2 = 2 f^2 f_mm and
    16 f + f_x^2 = f (16 + f_m^2). The f^{3/2} and the f^2 cancel down to

        kappa = 16 f^{1/2} f_mm / (16 + f_m^2)^{3/2}

    Note what happened: every negative power of f has gone. In the h-form the
    equivalent expression divides by h and by (1 + h_x^2/4)^{3/2} with
    h_x = h h_m unbounded; here the only nonlinearity left is a bounded
    rational function of f_m. This is why the f-form can carry capillarity at
    the tip at all, and it is what the with_capillarity guard used to point at.

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
from pyoomph.expressions import (grad, partial_t, square_root, testfunction,
                                 var, weak)
from pyoomph.meshes.simplemeshes import LineMesh

V_TC = math.sqrt(2.0)


class TroutonSheet(Equations):
    """Slender-sheet retraction in the Lagrangian mass coordinate."""

    def __init__(self, mu: float, space: str = "C2", with_capillarity: bool = True,
                 staggered: bool = True, variable: str = "h"):
        super().__init__()
        self.mu = mu
        self.with_capillarity = with_capillarity
        self.staggered = staggered
        self.space = space
        if variable not in ("f", "h"):
            raise ValueError(f"variable must be 'f' or 'h', not {variable!r}")
        # THE THICKNESS UNKNOWN IS ONE FIELD UNDER TWO NAMES. Everything
        # downstream (initial condition, Dirichlet value, diagnostics) keys off
        # this name, so the two formulations share one code path rather than
        # forking the Problem.
        self.variable = variable
        self.thickness_field = variable
        # The auxiliary gradient field is p = h_m in the h-form and q = f_m in
        # the f-form. Different symbols because they are different quantities:
        # substituting one for the other in the curvature is a silent factor of
        # 2 h.
        self.grad_field = "q" if variable == "f" else "p"
        if staggered and with_capillarity:
            # The gradient of a cell-constant thickness is zero inside every
            # element, so the auxiliary field is identically zero and the
            # curvature with it. This is a property of the D0 space, NOT of the
            # choice of variable -- the f-form fixes the tip singularity of the
            # curvature, not the absence of an elementwise gradient. Fail loudly
            # rather than silently integrating a zero curvature.
            raise ValueError(
                "staggered=True does not support with_capillarity=True: a D0 "
                "thickness has zero elementwise gradient, so its gradient "
                f"field {self.grad_field} = d/dm is identically zero. Use "
                "staggered=False; with variable='f' the co-located capillary "
                "formulation uses the Savva & Bush eq. (A1) curvature, which "
                "is regular at the tip.")
        # gamma = 1 in code units; this is Savva & Bush eq. 2.28. Identical in
        # both variables, since h^2 u_m = f u_m.
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
        # Richardson limit u*(30) -> 0.98073 recorded in
        # docs/planar-campaign/planar-campaign.tex sec:oned.
        #
        # THE VELOCITY SPACE IS THE DISCRIMINATOR, NOT THE THICKNESS SPACE. A
        # variant sweep at N=1600 settles this: with u on C2, the failure time is
        # 174-182 whether h is C2, C1, D0, D1 or D2 (five spaces, 4800-12802
        # dof); with u on C1 it does not fail at all. The parasitic mode lives on
        # the MIDSIDE NODES OF THE QUADRATIC VELOCITY FIELD, and h merely
        # inherits it through h_t = -h^2 u_m. The h checkerboard visible in the
        # profiles is the print of the mode, not its source -- which is why every
        # earlier fix aimed at h, or at the p/kappa block, changed nothing.
        #
        # THE STAGGERED/CO-LOCATED SWITCH IS ORTHOGONAL TO THE h/f SWITCH. The
        # spaces below are chosen by `staggered`; only the NAME of the thickness
        # unknown is chosen by `variable`. All four combinations are legal
        # (modulo the capillarity guard above).
        if self.staggered:
            self.define_scalar_field(self.thickness_field, "D0")
            self.define_scalar_field("u", "C1")
        else:
            self.define_scalar_field(self.thickness_field, self.space)
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
            # p = dh/dm, or q = df/dm
            self.define_scalar_field(self.grad_field, self.space)
            self.define_scalar_field("kappa", self.space)

    def define_residuals(self):
        if self.variable == "f":
            self._define_residuals_f()
        else:
            self._define_residuals_h()

    def _define_residuals_h(self):
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

    def _define_residuals_f(self):
        """Savva & Bush's f = h^2. See the derivation in the module docstring.

        The two residual sets are the SAME continuous problem; nothing here is
        a modelling change. What changes is where the nonlinearity sits: the
        viscous term becomes linear in the unknown thickness variable and the
        curvature loses every negative power of the thickness, at the cost of
        an f^{3/2} in the mass equation and an f^{1/2} in the capillary term.
        """
        f, u = var("f"), var("u")
        ft, ut = testfunction("f"), testfunction("u")

        f_m = grad(f)[0] if not self.staggered else None
        u_m = grad(u)[0]
        sqrt_f = square_root(f)

        # 1. mass:      f_t = -2 f^{3/2} u_m      [ = 2 h h_t = -2 h^3 u_m ]
        self.add_residual(weak(partial_t(f) + 2.0 * f * sqrt_f * u_m, ft))

        # 2. momentum:  u_t = 4 mu d_m(f u_m) + f^{1/2} kappa_m
        #    h^2 = f, so the viscous term is LINEAR in f -- its Jacobian block
        #    is 4 mu <v_m, u_m> in f and 4 mu <v_m, f> in u, with no
        #    8 mu h u_m term. Same integration by parts, so the tip closure is
        #    still the natural boundary term and TipFlux is unchanged.
        self.add_residual(weak(partial_t(u), ut))
        self.add_residual(4.0 * self.mu * weak(f * u_m, grad(ut)[0]))

        if self.with_capillarity:
            q, kap = var("q"), var("kappa")
            qt, kt = testfunction("q"), testfunction("kappa")
            q_m, kap_m = grad(q)[0], grad(kap)[0]

            # kappa = 16 f^{1/2} f_mm / (16 + f_m^2)^{3/2}, i.e. Savva & Bush
            # eq. (A 1) without their azimuthal term, pushed through
            # d/dx = f^{1/2} d/dm. Bounded as f -> 0 at the tip, which is the
            # entire reason for the change of variable.
            kappa_expr = 16.0 * sqrt_f * q_m / (16.0 + q * q) ** 1.5

            self.add_residual(-weak(sqrt_f * kap_m, ut))
            # 3. auxiliary: q = f_m
            self.add_residual(weak(q - f_m, qt))
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
        # THE THICKNESS UNKNOWN: "f" solves for f = h^2 as Savva & Bush do
        # (p. 236); "h" is the original formulation. Orthogonal to `staggered`:
        # all four combinations run.
        #
        # DEFAULT "h", AND THAT IS A MEASURED RESULT RATHER THAN INERTIA. The
        # f-form was implemented and verified at Oh_SB = 1 on 2026-08-11, and
        # the verification says it is the same continuous problem and a
        # slightly worse discretisation of it while with_capillarity is False:
        #
        #   * SAME EQUATIONS. Richardson-extrapolated to dt -> 0, u*(30) is
        #     0.9807280717 (f, staggered) against 0.9807280867 (h, staggered),
        #     a relative difference of 1.5e-8 -- and 1.5e-8 co-located as well.
        #     Two independent nonlinear discretisations agreeing to eight
        #     digits in the limit is the strongest correctness evidence
        #     available for the algebra above.
        #   * WORSE AT FINITE dt. At the reference nsteps = 6400 the f-form
        #     gives 0.9805177121 against 0.9805910803, off by 7.5e-5 relative,
        #     because its BDF1 error constant is 1.54x larger: the mass
        #     equation gained the f^{3/2} that the h-form does not carry.
        #     Same story on h_tip against the exact 1 + t*/4 -- 1.6e-4 relative
        #     against 1.1e-5 -- and both extrapolate to the same 1.1e-5.
        #   * NO EXTRA REACH. At N = 1600, M = 900 both staggered forms run the
        #     full t* = 400 cap. Co-located, the f-form dies at t* = 154 where
        #     the h-form dies at t* = 169 -- marginally SOONER. That is the
        #     expected result given the velocity space, not the thickness
        #     variable, is the discriminator, and it is direct evidence that
        #     f = h^2 is not a cure for the co-located mode.
        #   * NO CHEAPER. ~10.9 vs ~11.4 timesteps/s on the same mesh; the
        #     linear viscous term is paid back by the f^{3/2} mass term.
        #
        # So the change of variable earns nothing here and costs accuracy. It
        # is NOT dead code, though, and this default should flip the moment
        # with_capillarity is revisited: the reason Savva & Bush solve for f is
        # the curvature at the tip, and eq. (A1) is the only form of it that is
        # bounded there. Set variable="f" to get it.
        self.variable = "h"
        # BDF2. THIS WAS "BDF1" UNTIL 2026-08-11, AND THAT COST A FACTOR OF
        # ~250 IN ACCURACY PER STEP FOR NOTHING.
        #
        # The old comment recorded that BDF2 "fails at the third timestep, every
        # time, on this problem" -- dt collapsing to the 1e-12 floor while the
        # fields stayed healthy and Newton converged to 1e-10 -- and blamed
        # BDF2's two-level history against the sqrt(t) start-up singularity.
        # That failure is NO LONGER REPRODUCIBLE. BDF2 now runs to completion in
        # every configuration tried: staggered, co-located, and co-located with
        # with_capillarity=True. Which of the intervening changes fixed it is
        # NOT established -- note in particular that it works with p and kappa
        # present, so the auxiliary-block removal is not the explanation, and it
        # works co-located, so the staggering is not either. Recorded as
        # unexplained rather than given a tidy cause it has not earned.
        #
        # Measured, Oh_SB = 1, t* = 30, N = 400, staggered:
        #     nsteps    1600            3200            6400
        #     u*(30)    0.9807275968    0.9807280191    0.9807281257
        #     successive ratio 3.96  ->  observed order 1.99
        # against first order (ratio 2.00) for BDF1. The BDF2 value at 6400 sits
        # 3.9e-8 from the independently obtained BDF1 dt->0 Richardson limit
        # 0.9807280867, so a SINGLE BDF2 run is now the converged answer and the
        # two-level Richardson extrapolation every previous result needed is no
        # longer required.
        #
        # If BDF2 ever does misbehave at start-up again, do NOT shrink startstep.
        # For stiff problems that is a documented anti-remedy: variable-step BDF2
        # started from a first-order step is asymptotically equivalent to the
        # trapezoidal rule and loses L-stability, so the error GROWS as the first
        # step shrinks (Nishikawa, J. Comput. Phys. 392 (2019) 456). The sourced
        # fixes are a self-starting L-stable second-order start (SDIRK2/TR-BDF2),
        # a graded first-step schedule, or Rannacher damping -- or hold BDF1 on
        # the stiff momentum term alone via partial_t(u, scheme="BDF1").
        #
        # Note also that nobody in this literature uses BDF at all: Eggers &
        # Dupont, Savva & Bush and Bertozzi & Pugh all use a two-level theta
        # scheme (theta ~ 0.55), and Ahsan et al. use a modified Crank-Nicolson.
        # Those are self-starting and have no multistep history to corrupt; they
        # remain the contingency if this ever regresses.
        self.default_timestepping_scheme = "BDF2"

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
                           staggered=self.staggered, variable=self.variable)
        # Must match define_fields: pyoomph raises if an initial condition names
        # a field the element does not carry. The undisturbed sheet is h = 1,
        # hence f = h^2 = 1 -- the same number, which is why the initial
        # condition needs no branch on the variable, only on its name.
        # Read the field names off the equations BEFORE the first `+=`: that
        # operator returns a combined EquationTree, not the TroutonSheet, so
        # `eqs.thickness_field` stops resolving after it.
        thickness, gradfield = eqs.thickness_field, eqs.grad_field
        ic = {thickness: 1, "u": 0}
        if self.with_capillarity:
            ic[gradfield] = 0
            ic["kappa"] = 0
        eqs += InitialCondition(**ic)
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
            eqs += DirichletBC(**{thickness: 1, "u": 0}) @ "right"
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
        """h at the tip, as sqrt(f) when the unknown is f. NaN if unavailable.

        A "D0" thickness is elementwise-constant and carries no nodal value, so
        in the staggered scheme there is nothing to read at m = 0. NaN rather
        than a guess: the exact result h_tip = 1 + t*/4 is available
        analytically (Savva & Bush eq. 2.29, which this code reproduces exactly
        in the co-located scheme), so a caller that needs it should use that
        rather than have this method invent one.

        In the f-form the stored unknown is f = h^2, so this takes the square
        root and returns a THICKNESS in both formulations -- the column means
        the same thing in every output file. Clamped at zero first: f is only
        non-negative as a matter of physics, not as a constraint the discrete
        problem enforces, and a transiently negative tip value should surface as
        NaN rather than as a domain error inside math.sqrt.
        """
        if self.staggered:
            return float("nan")
        node, idx = self._tip_node_and_indices()
        val = float(node.value(idx[self.variable]))
        if self.variable == "h":
            return val
        return math.sqrt(val) if val >= 0.0 else float("nan")

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
        # Reported as a THICKNESS drift in both formulations: the undisturbed
        # state is h = 1 and f = 1, but away from it |f - 1| = |h^2 - 1| is
        # about twice |h - 1|, so comparing a raw f residual against the same
        # drift tolerance would be a factor-of-two tightening by accident.
        if self.staggered:
            dh = float("nan")
        else:
            val = float(node.value(idx[self.variable]))
            h_far = val if self.variable == "h" else (
                math.sqrt(val) if val >= 0.0 else float("nan"))
            dh = abs(h_far - 1.0)
        return dh, abs(float(node.value(idx["u"])))
