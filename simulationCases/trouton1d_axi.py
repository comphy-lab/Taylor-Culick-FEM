"""Axisymmetric hole opening in the per-radian mass coordinate (axi TC1D-FEM).

This is the axisymmetric sibling of ``trouton1d_pyoomph.py``. The grid stays
one-dimensional and axisymmetry enters through the metric: an ``r``-weighted
mass measure and the hoop stress. Read ``docs/axi-design.md`` first; the
equations below are its "Mass-coordinate form" section, and the reference is
Savva & Bush, JFM 626 (2009), section 3 and Appendix.

MODEL. Radial coordinate ``r``, hole edge ``R(t)``, film on ``r > R``, full
thickness ``h``, depth-averaged radial velocity ``u``. Plane-stress viscous
membrane stresses (``sigma_zz = 0``):

    sigma_rr   = 2 mu (2 u_r + u/r)
    sigma_thth = 2 mu (u_r + 2 u/r)

    h_t + (1/r) (r u h)_r = 0
    h (u_t + u u_r) = d(h sigma_rr)/dr + h (sigma_rr - sigma_thth) / r

Both new terms relative to planar -- the metric weight and the hoop stress --
vanish as ``h0/R -> 0``, so the planar operator coded in
``trouton1d_pyoomph.py`` is the exact ``R0 -> infinity`` limit. That is not a
remark, it is the primary regression test: ``validation/trouton1d_axi_checks.py
planar-limit``.

THE MASS COORDINATE, AND WHY THE METRIC BECOMES AN UNKNOWN. With the per-radian
mass label ``dm = r h dr`` the tip sits at ``m = 0`` and the domain ``[0, M]``
is fixed, exactly as planar. But ``d/dr = r h d/dm`` now carries ``r``, so the
radial map has to be carried along with the solution. It is carried as

    w = r^2 / 2   on C1,     w_m = 1/h,     r = sqrt(2 w)

and NOT as ``r`` itself. Two reasons, and they are the whole reason this
formulation is exact rather than merely consistent:

  * ``w_m = 1/h`` is linear in the carried unknown, where ``r_m = 1/(r h)`` is
    not. With ``w`` on C1 and ``h`` on D0 both sides of the constraint are
    cell-constant, so the discrete residual is a per-cell statement, and the
    boundary anchor below closes it by back-substitution from the far end. The
    discrete radial map is then EXACTLY consistent with the discrete mass
    distribution, cell by cell, to round-off -- verified by the ``metric``
    check rather than assumed.
  * ``r^2 = 2w`` exactly, so every even power of ``r`` in the residuals is a
    polynomial in the unknown and only the odd powers cost a square root.

Evolving ``r_t = u`` pointwise instead is consistent only in the continuum; it
would let the discrete radial map drift away from the discrete mass it is
supposed to describe.

TIP CLOSURE. The massless-cap force balance is unchanged from planar at leading
order, ``N_rr(R) = -2 gamma_total = -2``; Savva & Bush's axisymmetric jump
condition (3.20) is the same ``[u_r] = -1/4`` as their planar (2.22). What the
axisymmetric problem adds is the measure: the natural boundary term of the flux
slot carries the tip radius, so the per-radian pull is ``2 R`` and the boundary
value is the planar ``-2`` MULTIPLIED BY THE UNKNOWN ``R``. That makes the tip
term nonlinear in a global degree of freedom, which pyoomph's symbolic Jacobian
handles without special treatment. ``R`` itself is defined by the tip
kinematics, ``dR/dt = u|_{m=0}``, as a point contribution, and the metric is
anchored to it by ``w|_{m=0} = R^2/2``.

That anchor must be an ``EnforcedBC`` and never a ``DirichletBC``: its value
depends on an unknown. This is the recorded repository trap, and here it would
fail silently rather than loudly -- a Dirichlet value read once from the
current ``R`` gives a plausible run whose metric lags the rim by one Newton
step.

THE ONE RECIPROCAL. The momentum weak form is written so that the only negative
power of the radius anywhere in it is a single guarded ``u/r^2``, implemented
as ``u/(2w)``:

    int u_t v dm
    + int [ 4 mu r^2 h^2 u_m + 2 mu h u ] v_m dm      [flux slot]
    + int [ 2 mu h u_m + 4 mu u / r^2 ] v dm          [hoop remainder]
    - 2 R v(0) = 0                                    [tip pull]

Coding the hoop term literally as ``sigma_thth / r`` instead introduces a
spurious ``1/r`` which cancels analytically but not numerically, and it is what
a future refactor is most likely to reintroduce. The ``dissipation`` check
exists to catch exactly that: setting ``v = u`` in the weak form above
collapses the viscous terms to ``int 4 mu [u_r^2 + u_r (u/r) + (u/r)^2] dm``
with ``u_r = r h u_m``, i.e. exactly the depth-averaged dissipation
``2 mu h [u_r^2 + (u/r)^2 + (u_r + u/r)^2]`` integrated over the per-radian
area ``dm/h``. That form is positive definite (eigenvalues 1/2 and 3/2), so a
sign error in the hoop term makes it go negative immediately.

CAPILLARITY IS NOT PORTED. As planar, the tip closure already carries the
capillary driving and a bulk ``kappa_r`` would double-count it. Beyond that, a
D0 thickness has zero elementwise gradient, so any curvature built from it is
identically zero; and if capillarity is ever wanted here it must arrive
together with the change of variable ``f = h^2``, whose azimuthal curvature
(Savva & Bush eq. A1) is the only form bounded at the tip. ``with_capillarity``
therefore raises rather than pretending.

STAGGERING AND THE LEGACY PATH. ``h`` on D0 against ``u`` on C1, no midside
velocity degrees of freedom. The co-located C2/C2 arrangement is retained in
the planar module only to reproduce runs made before 2026-08-11; it is
non-convergent (the parasitic mode lives on the midside nodes of the quadratic
velocity field) and is deliberately NOT available here.

``R0 = 0`` IS EXCLUDED, on four independent grounds recorded in the design
note: the early-time solution ``R(t) = R0 exp(t/(2 tau_vis))`` is linear in
``R0``, the mass map degenerates, the edge force and rim mass both vanish, and
the azimuthal curvature diverges below the Taylor & Michael (1973) closing
threshold at ``R ~ h0``. Production runs use ``R0 >= 5`` (Savva & Bush's own
smallest is 5H).
"""

from __future__ import annotations

import math

from pyoomph import (
    DirichletBC,
    Equations,
    InitialCondition,
    Problem,
    TextFileOutput,
    WeakContribution,
)
from pyoomph.expressions import grad, partial_t, square_root, testfunction, var, weak
from pyoomph.meshes.bcs import EnforcedBC
from pyoomph.meshes.simplemeshes import LineMesh

V_TC = math.sqrt(2.0)


def auto_mass(r0: float, t_end: float, margin: float = 120.0) -> float:
    """Per-radian mass of an undisturbed film out to a sized radial margin.

    THE PLANAR RULE DOES NOT TRANSFER. ``trouton1d_sweep.auto_mass`` returns a
    LENGTH, because in a planar sheet of unit thickness mass and length are the
    same number. Here the domain is ``M = int_R^{r_max} r h dr``, which for the
    undisturbed film ``h = 1`` is ``(r_max^2 - R0^2)/2`` -- quadratic in the
    radial extent, so reusing the planar expression would size the domain by a
    factor ``~r_max`` too small.

    The radial extent itself is sized as planar: the rim accumulates the film
    it sweeps, so the disturbed region grows with the tip travel ``V_TC t``,
    and ``2.5x`` that (floored at the planar minimum of 120 thicknesses) is the
    validated margin. ``far_field_drift`` checks the result rather than
    trusting it.
    """
    r_max = r0 + max(margin, 2.5 * V_TC * t_end)
    return 0.5 * (r_max * r_max - r0 * r0)


class AxiTroutonSheet(Equations):
    """Axisymmetric slender-sheet retraction in the per-radian mass coordinate."""

    def __init__(self, mu: float, with_capillarity: bool = False):
        super().__init__()
        if with_capillarity:
            raise ValueError(
                "with_capillarity=True is not implemented in the axisymmetric "
                "mass-coordinate solver. A D0 thickness has zero elementwise "
                "gradient, so a curvature built from it is identically zero; "
                "and the axisymmetric curvature is only bounded at the tip in "
                "the f = h^2 variable (Savva & Bush eq. A1, both principal "
                "parts finite, azimuthal part -> 1/r). See the 'Governing "
                "equations' and 'Mass-coordinate form' sections of "
                "docs/axi-design.md: capillarity here requires switching to "
                "f = h^2 at the same time, which is a different discretisation "
                "rather than a flag.")
        self.mu = mu
        self.with_capillarity = False

    def define_fields(self):
        # Staggered only. See the module docstring: the velocity space is the
        # discriminator, and C2 velocities carry a non-convergent parasitic
        # mode on their midside nodes.
        self.define_scalar_field("h", "D0")
        self.define_scalar_field("u", "C1")
        # The radial metric, carried as w = r^2/2 so that its constraint is
        # linear and cell-constant against the D0 thickness.
        self.define_scalar_field("w", "C1")

    def define_residuals(self):
        mu = self.mu
        h, u, w = var("h"), var("u"), var("w")
        ht, ut, wt = testfunction("h"), testfunction("u"), testfunction("w")
        u_m, w_m = grad(u)[0], grad(w)[0]

        # r^2 = 2 w EXACTLY. Every even power of the radius is therefore a
        # polynomial in the unknown; only the odd powers pay for a square root,
        # and the sole reciprocal in the momentum residual is the guarded
        # u/(2w) below.
        r2 = 2.0 * w
        r = square_root(r2)

        # 1. mass:  h_t = -h (r h u_m + u/r)
        #    The tip-side factor u/r is the azimuthal stretching that has no
        #    planar counterpart; it is what thins the rim to 2/3 of the planar
        #    rate at late time.
        self.add_residual(weak(partial_t(h) + r * h * h * u_m + h * u / r, ht))

        # 2. momentum. The flux slot is integrated by parts, which is what
        #    turns the tip closure into a natural boundary term; the far end is
        #    Dirichlet so its test function vanishes there.
        self.add_residual(weak(partial_t(u), ut))
        self.add_residual(weak(4.0 * mu * r2 * h * h * u_m + 2.0 * mu * h * u,
                               grad(ut)[0]))
        # Hoop remainder, expanded. Do NOT restructure this into sigma_thth/r.
        self.add_residual(weak(2.0 * mu * h * u_m + 4.0 * mu * u / r2, ut))

        # 3. metric constraint. Both sides are cell-constant (w linear on the
        #    element, h cell-constant), so with the tip anchor this is solved
        #    exactly per cell by back-substitution from m = M.
        self.add_residual(weak(w_m - 1.0 / h, wt))


class AxiTipFlux(Equations):
    """The planar tip closure carried by the axisymmetric measure.

    SIGN AND MAGNITUDE, inherited term for term from ``TipFlux`` in
    ``trouton1d_pyoomph.py`` and then re-weighted. The planar momentum residual
    integrates the viscous flux by parts and the surviving tip bracket
    contributes ``4 mu * (h^2 u_m)|_0 * v(0) = 4 mu * (-1/(2 mu)) * v(0)``,
    i.e. ``-2 v(0)``, independent of ``mu`` because the viscous prefactor
    cancels the closure. Getting that sign backwards pushes the tip outward,
    the sheet thickens without bound at ``m = 0``, and Newton then fails at
    every timestep down to the 1e-12 floor with a dt independent of the mesh.

    Axisymmetry changes only the weight. The flux slot here is
    ``4 mu r^2 h^2 u_m``, so the same bracket carries ``r_tip^2 h^2 u_m``, and
    with the tip closure written on the physical stress ``N_rr(R) = -2`` the
    boundary value becomes ``-2 R``: the per-radian version of a line force
    ``2 gamma`` acting along an edge of length ``R`` per radian. It is
    NONLINEAR in the global unknown ``R``.

    OPTIONAL AZIMUTHAL CORRECTION (off by default). The toroidal rim surface
    opposes opening with ``gamma pi h_tip / (2R)`` per unit edge length, so
    ``N_rr(R) = -2 + pi h_tip/(2R)`` and the boundary value gains
    ``+ pi h_tip / 2``. This is the O(h/R) effect neglected in Savva & Bush's
    early-time analysis, which requires ``H/R0 << 1``. At ``R0 ~ h0`` it is an
    O(1) fraction of the edge force, so the flag exists to MEASURE its size
    against the DNS, not to assert it. ``h_tip`` is the first cell's value, the
    thickness field being cell-constant.
    """

    def __init__(self, R, azimuthal_correction: bool = False):
        super().__init__()
        self.R = R
        self.azimuthal_correction = azimuthal_correction

    def define_residuals(self):
        value = -2.0 * self.R
        if self.azimuthal_correction:
            # domain=".." reaches into the parent bulk element. A D0 field has
            # no value ON the boundary -- ``var("h")`` here raises "Cannot
            # expand the field 'h'" -- but the interface element knows its bulk
            # element, whose thickness is the constant first-cell value. So
            # this IS the tip cell's h, one cell wide, not an extrapolation.
            value = value + math.pi * var("h", domain="..") / 2.0
        self.add_residual(value * weak(1, testfunction("u")))


class Trouton1DAxiProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Oh_SB = 1.0
        #: Initial hole radius in units of the full thickness h0. See the
        #: module docstring for why R0 = 0 is excluded and R0 >= 5 is the
        #: production floor.
        self.r0 = 5.0
        #: Per-radian mass of the domain [0, M]. None means auto_mass at the
        #: default margin, which is a STANDING film sizing and knows nothing
        #: about t_end; the driver overrides it with the run's own t_end.
        self.mass: float | None = None
        self.nelem = 400
        self.grade = 2.0           # tip clustering; 1 = uniform
        #: Not ported; see AxiTroutonSheet.__init__.
        self.with_capillarity = False
        #: The first geometric correction to the tip closure, default off.
        self.tip_azimuthal_correction = False
        #: Absolute Newton tolerance when the roundoff floor permits it, and
        #: the safety factor on the measured floor eps*(R0^2+2M)/dt; see
        #: actions_before_newton_solve.
        self.newton_tolerance_base = 1e-8
        self.newton_tolerance_safety = 10.0
        # BDF2, as planar. If it ever misbehaves at start-up do NOT shrink
        # startstep: variable-step BDF2 started from a first-order step is
        # asymptotically the trapezoidal rule and loses L-stability, so the
        # error GROWS as the first step shrinks (Nishikawa, J. Comput. Phys.
        # 392 (2019) 456). The sourced remedies are a self-starting L-stable
        # second-order start, a graded first-step schedule, Rannacher damping,
        # or holding BDF1 on the stiff momentum term alone.
        self.default_timestepping_scheme = "BDF2"

    @property
    def mu(self) -> float:
        return self.Oh_SB * math.sqrt(2.0)

    @property
    def tau_vis(self) -> float:
        return self.mu / 2.0

    @property
    def r0_star(self) -> float:
        """Savva & Bush's small-hole parameter R0* = R0 / (Oh_SB h0).

        Their high-Oh exponential opening ``R = R0 exp(t/(2 tau_vis))`` holds
        only for ``R0* << 1``; ``R0* >> 1`` follows the planar ``sqrt(t*/pi)``
        start instead. Use it as a regime label when choosing a test, not as a
        prediction.
        """
        return self.r0 / self.Oh_SB

    @property
    def domain_mass(self) -> float:
        return auto_mass(self.r0, 0.0) if self.mass is None else float(self.mass)

    def actions_before_newton_solve(self):
        """Re-tune the absolute Newton tolerance to this step's roundoff floor.

        Same trap and same cure as ``axisymmetric_lab.py`` (see its TRAP
        section): ``newton_solver_tolerance`` is an absolute max-norm bound on
        a residual whose flux slot carries ``4 mu r^2 h^2 u_m`` with
        ``r^2`` up to ``r_max^2 = R0^2 + 2M``, and whose mass rows carry
        ``1/dt``, so the direct-solve floor is ``~ eps*(R0^2 + 2M)/dt``.
        Measured on the t* = 100 comparison legs at r0 = 1.1129, M ~ 3.1e4:
        both runs died mid-campaign in a Newton two-cycle at 9.0e-4 --
        exactly ``eps*6.3e4/dt`` at the cascade's dt of ~1.5e-8 -- after the
        controller halved an output-landing sliver step into the raised
        floor. The checks never saw this because their shorter horizons keep
        M (hence the floor) three orders smaller.
        """
        super().actions_before_newton_solve()
        base = self.newton_tolerance_base
        try:
            dt = float(self.time_pt().dt(0))
        except Exception:  # noqa: BLE001 -- no timestepper yet: stationary solve
            dt = 0.0
        if not dt > 0.0:
            self.newton_solver_tolerance = base
            return
        r_max_sq = self.r0 * self.r0 + 2.0 * self.domain_mass
        floor = 2.220446049250313e-16 * r_max_sq / dt
        self.newton_solver_tolerance = max(base, self.newton_tolerance_safety * floor)

    def define_problem(self):
        # CARTESIAN, DELIBERATELY. The mesh coordinate is the mass label m, not
        # a radius, so pyoomph's axisymmetric coordinate system would apply a
        # 2*pi*m weight to the measure -- a weight in the wrong variable. The
        # axisymmetric metric is carried EXPLICITLY by w and the r-factors in
        # the residuals above. This is the whole design choice of the file.
        self.set_coordinate_system("cartesian")

        M = self.domain_mass
        # LineMesh is uniform; the nodes are remapped after initialisation to
        # cluster at the tip, where the impulsive start puts a diffusive layer
        # that is arbitrarily thin as t -> 0+. An unresolved layer there does
        # not merely lose accuracy: the timestepper cannot meet its tolerance
        # on a feature the mesh cannot represent, so dt collapses to the floor
        # and the solve aborts.
        self += LineMesh(N=self.nelem, size=M, nodal_dimension=1, name="sheet")

        # The rim radius: one global unknown, fixed by the tip kinematics.
        R, Rtest = self.add_global_dof("R", initial_condition=self.r0)

        eqs = AxiTroutonSheet(mu=self.mu,
                              with_capillarity=self.with_capillarity)

        # Undisturbed film of unit thickness at rest, with the EXACT radial map
        # for h = 1: w = (R0^2 + 2m)/2, i.e. r^2 = R0^2 + 2m. That is the
        # canonical-rupture analogue of the planar h = 1, u = 0 start, and it
        # satisfies w_m = 1/h identically, so the run begins on the metric
        # constraint rather than relaxing onto it.
        eqs += InitialCondition(
            h=1, u=0, w=(self.r0 * self.r0 + 2.0 * var("coordinate_x")) / 2.0)

        eqs += AxiTipFlux(
            R=R, azimuthal_correction=self.tip_azimuthal_correction) @ "left"

        # dR/dt = u(0). A point contribution: in this cartesian 1D problem the
        # "left" boundary mesh is a single point and the weak form is plain
        # evaluation, so nothing vanishes the way a point constraint on an
        # axisymmetric axis would.
        eqs += WeakContribution(partial_t(R) - var("u"), Rtest) @ "left"

        # The metric anchor. EnforcedBC and never DirichletBC: the prescribed
        # value depends on the unknown R.
        eqs += EnforcedBC(w=var("w") - R * R / 2.0) @ "left"

        # Far end: undisturbed film. A D0 thickness has no nodal value to pin,
        # and w must stay free -- its own constraint plus the tip anchor
        # determine it, and pinning it would over-determine the metric.
        # Validity is a domain-length statement, which far_field_drift checks.
        eqs += DirichletBC(u=0) @ "right"

        eqs += TextFileOutput()
        self += eqs @ "sheet"

    def initialise(self, *args, **kwargs):
        """Build, grade the mesh, then RE-APPLY the initial condition.

        The re-application is not optional here, and this is where the axi
        solver departs from the planar one. Grading cannot happen inside
        ``define_problem`` because the mesh does not exist until initialisation
        runs; planar can then get away with grading afterwards only because its
        initial condition is uniform, so moving nodes under it changes nothing.
        Here ``w`` is a FUNCTION OF m, so nodes moved after the fact would
        carry values sampled at their old positions -- an initial metric
        inconsistent with the initial mass distribution, which is precisely the
        error the formulation exists to avoid.
        """
        super().initialise(*args, **kwargs)
        self._grade_mesh_at_tip()
        self.set_initial_condition()

    def _grade_mesh_at_tip(self) -> None:
        """Remap nodes onto m = M (xi)^grade, clustering at the tip.

        Both the Eulerian and Lagrangian coordinates are set: the mesh
        coordinate IS the material label, so leaving x_lagr on the uniform
        distribution would silently give every element the wrong measure.
        """
        if self.grade == 1.0:
            return
        mesh = self.get_mesh("sheet")
        M = self.domain_mass
        for node in mesh.nodes():
            xi = node.x(0) / M
            m_new = M * xi ** self.grade
            node.set_x(0, m_new)
            node.set_x_lagr(0, m_new)

    # ---- diagnostics --------------------------------------------------------

    def _sorted_nodes(self):
        mesh = self.get_mesh("sheet")
        # get_nodal_field_indices maps names to positions. has_interface_dof_id
        # is for INTERFACE dofs and returns -1 for a bulk field, which index
        # -1 then silently reads as the last value on the node.
        return sorted(mesh.nodes(), key=lambda nd: nd.x(0)), \
            mesh.get_nodal_field_indices()

    def nodal_profile(self) -> tuple[list[float], list[float], list[float]]:
        """``(m, u, w)`` at the nodes, ordered from the tip outwards."""
        nodes, idx = self._sorted_nodes()
        iu, iw = idx["u"], idx["w"]
        return ([float(nd.x(0)) for nd in nodes],
                [float(nd.value(iu)) for nd in nodes],
                [float(nd.value(iw)) for nd in nodes])

    def cell_profile(self) -> list[tuple[float, float, float]]:
        """``(m_left, m_right, h)`` per cell, ordered from the tip outwards.

        A D0 field lives in the element's internal data rather than on nodes,
        so it is read through the discontinuous field index, not through
        ``get_nodal_field_indices``.
        """
        mesh = self.get_mesh("sheet")
        di = mesh.get_code_gen().get_code().get_discontinuous_field_index("h")
        if di < 0:
            raise RuntimeError("No discontinuous field 'h' on the sheet mesh")
        cells = []
        for e in mesh.elements():
            xs = [e.node_pt(i).x(0) for i in range(e.nnode())]
            cells.append((float(min(xs)), float(max(xs)),
                          float(e.internal_data_pt(di).value(0))))
        cells.sort(key=lambda c: c[0])
        return cells

    def tip_velocity(self) -> float:
        """``u`` at m = 0, the retracting tip."""
        nodes, idx = self._sorted_nodes()
        return float(nodes[0].value(idx["u"]))

    def rim_radius(self) -> float:
        """The global unknown ``R(t)``."""
        return float(self.get_ode("globals").get_value("R", as_float=True))

    def tip_thickness(self) -> float:
        """``h`` in the first cell.

        The thickness is cell-constant, so there is no nodal value at m = 0 and
        this is a one-cell-wide average rather than a point value. It converges
        to the tip thickness with the mesh; the exact statement to compare it
        against is the ODE ``dh_tip/dt = 1/(2 mu) - h_tip u_tip / (2 R)``,
        integrated alongside the run by the driver.
        """
        return self.cell_profile()[0][2]

    def far_field_drift(self) -> tuple[float, float]:
        """``|h - 1|`` in the last cell and ``|u|`` at m = M.

        The far end is a clamped undisturbed film, valid only while the
        disturbance has not reached it, so this checks that the DOMAIN is long
        enough rather than that the boundary condition applied. Unlike planar,
        the thickness drift IS available here: it is the last cell's value, not
        a nodal one. The velocity remains the more sensitive of the two -- the
        disturbance arrives as motion before it measurably thins the film.
        """
        nodes, idx = self._sorted_nodes()
        h_far = self.cell_profile()[-1][2]
        return abs(h_far - 1.0), abs(float(nodes[-1].value(idx["u"])))

    def metric_defect(self) -> float:
        """max over cells of ``|w_m - 1/h|``.

        Zero to round-off by construction: the constraint residual is
        cell-constant and the tip anchor closes the triangular system. If this
        drifts, the discretisation of ``w`` is wrong -- it is the sharpest
        internal check the formulation offers.
        """
        _, _, w = self.nodal_profile()
        worst = 0.0
        for j, (ml, mr, h) in enumerate(self.cell_profile()):
            w_m = (w[j + 1] - w[j]) / (mr - ml)
            worst = max(worst, abs(w_m - 1.0 / h))
        return worst

    def metric_closure(self) -> float:
        """``r(M)^2 - R^2 - 2 int_0^M dm/h``, which is identically zero.

        Mass is exactly conserved in this coordinate by construction -- ``M``
        does not move -- so the meaningful global identity is the metric one:
        integrating ``w_m = 1/h`` from the anchored tip out to the far end must
        reproduce the far-end radius. It tests the anchor, the constraint and
        the cell measures together.
        """
        _, _, w = self.nodal_profile()
        integral = sum((mr - ml) / h for ml, mr, h in self.cell_profile())
        return 2.0 * w[-1] - self.rim_radius() ** 2 - 2.0 * integral

    def dissipation(self) -> tuple[float, float]:
        """``(min cell density, total per-radian dissipation)``.

        The density is the depth-averaged
        ``2 mu h [u_r^2 + (u/r)^2 + (u_r + u/r)^2]`` with ``u_r = r h u_m``,
        evaluated per cell from the solved fields; the integral is over the
        per-radian area ``dm/h``. Setting the momentum test function to ``u``
        collapses the coded weak form to exactly this integral, so a negative
        value is not a rounding artefact but a sign error in the hoop term.
        """
        _, u, w = self.nodal_profile()
        worst = math.inf
        total = 0.0
        for j, (ml, mr, h) in enumerate(self.cell_profile()):
            dm = mr - ml
            u_m = (u[j + 1] - u[j]) / dm
            r = math.sqrt(w[j] + w[j + 1])          # sqrt(2 * w_centre)
            u_r = r * h * u_m
            s = 0.5 * (u[j] + u[j + 1]) / r
            density = 2.0 * self.mu * h * (u_r * u_r + s * s + (u_r + s) ** 2)
            worst = min(worst, density)
            total += density * dm / h
        return worst, total
