"""TC1D-ALE-AXI: axisymmetric hole-opening on a genuinely moving physical mesh.

Direct axisymmetric extension of ``trouton1d_ale.py`` (TC1D-ALE, the planar
physical-coordinate ALE solver -- read that module's docstring first, it is
not repeated here). See ``docs/axi-design.md``, section "ALE physical-
coordinate form (axi TC1D-ALE)", for the governing-equation derivation this
file implements; equation numbers below are Savva & Bush, JFM 626 (2009).

THE MODEL.
----------
Radial coordinate ``r``, hole edge (per-radian, membrane) ``R(t)``, film on
``r > R``. Depth-averaged radial velocity ``u(r,t)``, full thickness
``h(r,t)``, ``rho = sigma = h0 = 1``, coded viscosity ``mu = sqrt(2) Oh``:

    h_t + (1/r)(r h u)_r = 0                                    (= S&B 3.15)
    h(u_t + u u_r) = (r N_rr)_r/r - N_thth/r
    N_rr = 2 mu h (2 u_r + u/r),   N_thth = 2 mu h (u_r + 2 u/r)

Two pieces are new relative to planar, both vanishing as ``R -> infinity``
(the planar solver is the exact large-``R`` limit, checked below): the
metric term ``h u/r`` in the mass equation, and the hoop stress ``N_thth/r``
in momentum. Capillarity is off by default for the same reason as planar:
the tip closure already carries the capillary pull of the massless cap, so a
bulk curvature term would double-count it.

WHY THE MEASURE, NOT THE RESIDUALS, DOES MOST OF THE WORK.
------------------------------------------------------------
``Trouton1DALEAxiProblem.define_problem`` calls
``self.set_coordinate_system("axisymmetric")``. pyoomph's
``AxisymmetricCoordinateSystem.integral_dx`` (``.venv/.../pyoomph/expressions/
coordsys.py:409-422``) returns ``2*pi*coordinate_x*dx`` for *every* element
dimension a ``weak()`` term is assembled over on this domain -- including the
zero-dimensional point term a boundary ``Equations`` class contributes at a
mesh endpoint. Consequently:

  * ``weak(f, v)`` over the bulk line already means ``int f v 2 pi r dr``; no
    residual below multiplies by ``r`` for that reason.
  * ``TipClosure``'s point contribution at "left" is automatically weighted
    by ``2*pi*R(t)``, so the constant it carries is the SAME ``-2`` as
    planar's ``TipClosure`` / the mass-coordinate solver's ``TipFlux`` --
    nothing in this file has to know ``R(t)`` to get the edge force right.

The one place this has to be checked rather than trusted is the momentum
by-parts split, because both an integration-by-parts boundary term and an
explicit hoop term are in play. Multiplying the strong momentum equation by
a test function ``v`` and the measure ``r dr`` and integrating
``(r N_rr)_r`` by parts,

    int h(u_t+u u_r) v r dr
      = int (r N_rr)_r v dr - int N_thth v dr        [strong form, times r,
                                                        (N_rr-N_thth)/r * r
                                                        = N_rr - N_thth]
      = [N_rr v r] - int N_rr (v r)_r dr - int N_thth v dr
      = [N_rr v r] - int N_rr v_r r dr - int N_rr v dr - int N_thth v dr
      = [N_rr v r] - int N_rr v_r r dr - int N_thth v dr        (N_rr - N_thth
                                                                  cancel the
                                                                  +int N_rr v dr
                                                                  against the
                                                                  hoop term:
                                                                  see below)

so, moving the boundary term to the other side and dropping the interior
``int N_rr v dr`` against the same term hidden inside ``N_thth``'s weak form
(the algebra is carried out in full in the commit that introduced this file;
what matters operationally is the result):

    int h(u_t+u u_r) v r dr + int N_rr v_r r dr + int N_thth v dr
      - [N_rr v r]_bdry = 0.

Written as pyoomph ``weak()`` calls under the axisymmetric measure (which
supplies the missing ``r`` in each term above automatically):

    weak(N_rr, grad(v)[0])   ==  int N_rr v_r * (2 pi r) dr      [flux slot]
    weak(N_thth/r, v)        ==  int (N_thth/r) v * (2 pi r) dr
                              ==  int N_thth v * (2 pi) dr        [hoop term]

i.e. the ``1/r`` written explicitly in the hoop-term code below is not a
numerical convenience, it is exactly what cancels the extra ``r`` the
measure would otherwise leave on that term -- the flux slot needs the
measure's ``r`` (that IS the ``(r N_rr)_r`` divergence), the hoop term must
not have it (there is no ``r`` multiplying ``N_thth`` in the strong
equation), and writing ``N_thth/r`` under the ``r``-weighted measure is how
a single ``weak()`` call gets both terms right without a guarded reciprocal
anywhere except the physical ``u/r`` inside ``N_rr``/``N_thth`` themselves
(finite since ``r >= R0 > 0`` always -- ``R0 = 0`` is excluded on four
independent grounds in ``docs/axi-design.md``).

TIP CLOSURE.
------------
Leading order is unchanged from planar: S&B's axisymmetric jump condition
(3.20) is the same ``[u_r] = -1/4`` as planar (2.22), so the massless-cap
force balance is still ``N_rr(R) = -2`` for every ``Oh`` -- see
``TipClosure`` below, and trouton1d_ale.py's module docstring for the
``mu``-cancellation derivation, which is identical here. The optional
``tip_azimuthal_correction`` flag adds the first geometric correction (the
toroidal rim opposing opening, O(h/R), docs/axi-design.md "Tip closure"):
``N_rr(R) = -2 + pi h_tip / (2 R)``. It exists to MEASURE that term's size
against a DNS, not to assert it belongs in the leading-order model, and it
is cheap here specifically because ``h`` is C1 (a genuine nodal tip value
exists), unlike the mass-coordinate solver where ``h`` is D0.

MESH MOTION.
------------
``LaplaceSmoothedMesh`` is unchanged, with its OWN default cartesian
coordinate system (``coordsys=cartesian`` in its constructor,
``pyoomph/equations/ALE.py:159`` -- deliberately not overridden here). In 1D
its ``laplace(x - X) = 0`` is affine in the Lagrangian coordinate exactly as
in planar, so with the domain offset to ``[R0, R0+l0]`` (see below) and a
kinematic tip and a pinned far end it gives

    x(X) = R(t) + (R0 + l0 - R(t)) * (X - R0) / l0,

precisely S&B's own mapping (A2), applied to the OFFSET domain rather than
``[0, l0]``. This is the design note's recorded caveat, not a hidden
assumption: the affine map compresses resolution uniformly across the whole
domain as ``R`` grows, so it loses near-tip resolution by orders of
magnitude if ``R`` grows by orders of magnitude (the exponential small-hole
regime, ``R0* << 1``). An area- (or arclength-) preserving stretch is the
recorded upgrade path if reach becomes limited; it is not implemented here.

THE OFFSET DOMAIN, AND WHY IT IS DONE THE WAY GRADING IS DONE.
----------------------------------------------------------------
The planar mesh spans ``[0, l0]``; this one must span ``[R0, R0+l0]``. The
grading trap recorded in trouton1d_ale.py (trap 1: ``x_lagr`` must be set
alongside ``x`` after ``initialise()``, because the Laplace mesh equation is
written in ``x_lagr`` and leaving it ungraded silently changes the mesh-
motion law from "affine stretch of the graded mesh" to "affine stretch of a
uniform mesh interpolated onto graded data") applies IDENTICALLY to the
offset: ``LineMesh``'s own ``minimum=`` constructor argument was
deliberately NOT used here, so that the offset goes through the same
post-``initialise()``, set-``x``-and-``x_lagr`` code path as grading, in one
combined pass (``_offset_and_grade_mesh_at_tip``). Doing the offset any
other way (e.g. ``minimum=r0`` at mesh-construction time) would likely also
be correct, since ``x`` and ``x_lagr`` coincide at construction regardless
-- but it would hide the fact that this is the same trap, which is the point
of writing it this way.

DISCRETISATION.
----------------
``h`` and ``u`` on C1 (no D0 staggering here, no midside dofs anywhere) --
identical reasoning to planar: the ALE mass equation carries an advective
``(u-w) h_r``-type term (folded into ``partial_t(h, ALE="auto")`` plus the
Eulerian flux terms below), so a cell-constant ``h`` would need an upwind
flux to mean anything, and C1/C1 co-location is safe on a linear
``LineMesh`` because the non-convergent co-located mode lives on midside
nodes, which this mesh does not have.

TRAP HIT WHILE WRITING THIS: pyoomph's default ``newton_solver_tolerance``
(1e-8, an ABSOLUTE residual tolerance, ``pyoomph/generic/problem.py:564``) is
unreachable here at exactly the settings that work perfectly in planar --
``Oh=1, R0=5``, the same tiny relative first step
(``startstep = t_end*1e-6``) planar's own sweep driver uses. Newton does not
diverge; it lands on a genuine fixed point (bit-identical residual, or a
2-cycle between two bit-identical values, for 50+ iterations) at a residual
of a few ``1e-8``, and errors out only because that is just above tolerance.
Bisection (``debug_jacobian_by_fd_epsilon``, and reducing the residual to the
literal planar terms with nothing but the coordinate system switched to
axisymmetric) shows this is NOT a formulation bug: the FD Jacobian agrees
with the analytic one to roundoff, and the plateau reproduces with the
axi-only terms removed entirely -- switching on the axisymmetric measure
alone is sufficient. It is a CONDITIONING effect exactly like the one the
mass-coordinate solver's design note warns about for its ``r^2 h^2 u_m``
flux slot ("Conditioning watch", ``docs/axi-design.md``): the axisymmetric
measure multiplies bulk AND point residual/Jacobian entries by ``2*pi*r``,
so absolute magnitudes -- and the double-precision roundoff riding on them --
scale with ``R0``. At ``R0=200`` the SAME plateau appears even at an
ordinary (not tiny) first step. Measured fix, verified across
``R0 in {5, 200, 1000}`` and ``startstep/t_end in {1e-6, ..., 1e-3}``:
passing ``newton_solver_tolerance=1e-6`` to every ``solve()``/``run()`` call
(still nine orders of magnitude tighter than the ``V_TC=sqrt(2)`` scale of
the physics) restores clean 2-3-iteration quadratic Newton convergence in
every case tried. ``validation/trouton1d_ale_axi_sweep.py`` therefore passes
it by default; do not remove it without re-running the R0 sweep above.
"""

from __future__ import annotations

import math
import os

# Must precede the first pyoomph (hence MPI) import; see planar_comoving.py
# for why hwloc's OpenGL backend hangs MPI_Init on a host with a running
# compositor.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

from pyoomph import (DirichletBC, Equations, InitialCondition,  # noqa: E402
                     Problem, TextFileOutput)
from pyoomph.equations.ALE import LaplaceSmoothedMesh  # noqa: E402
from pyoomph.expressions import (grad, mesh_velocity, partial_t, pi,  # noqa: E402
                                 testfunction, var, weak)
from pyoomph.meshes.bcs import EnforcedBC  # noqa: E402
from pyoomph.meshes.simplemeshes import LineMesh  # noqa: E402

V_TC = math.sqrt(2.0)


class TroutonSheetALEAxi(Equations):
    """Axisymmetric long-wave sheet equations, physical (radial) coordinate.

    Same structure as ``TroutonSheetALE`` (trouton1d_ale.py): the residuals
    are written in their plain Eulerian form, and ``partial_t(..., ALE=...)``
    supplies the mesh-velocity correction once mesh-motion equations exist on
    the domain. Relative to planar, the mass residual gains the metric term
    ``h*u/r``; the momentum residual's flux slot gains the ``u/r`` part of
    ``N_rr``, and momentum gains an entirely new, non-integrated-by-parts
    hoop term ``N_thth/r``. See the module docstring for the by-parts
    algebra that fixes the exact form of both new momentum pieces.

    ``r`` is read as ``var("coordinate_x")``, the domain's own nodal
    coordinate under the axisymmetric measure -- not a derived quantity, and
    not the mesh's Lagrangian coordinate (that is ``coordinate_x`` too, in
    the sense that the Eulerian and Lagrangian coordinates coincide
    instantaneously; ``partial_t(..., ALE=...)`` is what tells the two apart
    in time).
    """

    def __init__(self, mu: float, ale: bool = True):
        super().__init__()
        self.mu = mu
        self.ale = ale

    def define_fields(self):
        # C1/C1, no D0 staggering, no midside dofs: see the module docstring.
        self.define_scalar_field("h", "C1")
        self.define_scalar_field("u", "C1")

    def define_residuals(self):
        h, u = var("h"), var("u")
        ht, ut = testfunction("h"), testfunction("u")
        h_x, u_x = grad(h)[0], grad(u)[0]
        r = var("coordinate_x")
        ale = "auto" if self.ale else False

        # 1. mass: h_t|_x + (1/r)(r h u)_r = h_t + h u_x + u h_x + h u/r = 0.
        #    The only new term relative to planar is h*u/r; the ALE
        #    mesh-velocity correction is exactly partial_t(h, ALE="auto"),
        #    as in planar.
        self.add_residual(
            weak(partial_t(h, ALE=ale) + h * u_x + u * h_x + h * u / r, ht))

        # 2. momentum, multiplied through by h as in planar. u*u_x is the
        #    material-derivative advection, untouched by axisymmetry --
        #    partial_t already removed the mesh motion.
        self.add_residual(weak(h * (partial_t(u, ALE=ale) + u * u_x), ut))

        # flux slot: N_rr = 2 mu h (2 u_x + u/r). Its natural boundary term
        # is exactly what TipClosure supplies below, with the r_tip factor
        # coming from the axisymmetric measure rather than being written
        # here. Replaces planar's `4*mu*h*u_x`.
        self.add_residual(
            weak(2.0 * self.mu * h * (2.0 * u_x + u / r), grad(ut)[0]))

        # hoop source: N_thth/r = 2 mu h (u_x + 2 u/r) / r. Entirely new
        # relative to planar; NOT a boundary term (see the module-docstring
        # derivation for why this exact form, under the r-weighted measure,
        # is the correct discretisation of the hoop stress divergence).
        self.add_residual(
            weak(2.0 * self.mu * h * (u_x + 2.0 * u / r) / r, ut))


class TipClosure(Equations):
    """Free-edge stress condition, natural boundary term of the flux slot.

    Written as a fresh class rather than imported from ``trouton1d_ale``
    only because it needs the axi-only ``tip_azimuthal_correction`` flag;
    the leading-order constant is identical to planar's ``TipClosure`` and
    to the mass-coordinate solver's ``TipFlux`` (``h^2 u_m = h u_x``, so all
    three formulations must and do agree). See trouton1d_ale.py's module
    docstring for the ``4*mu*(-1/(2*mu)) = -2`` derivation, unchanged here.

    UNDER THE AXISYMMETRIC MEASURE this residual's point contribution at
    "left" is automatically weighted by ``2*pi*coordinate_x``
    (``AxisymmetricCoordinateSystem.integral_dx`` supplies that factor for
    every element dimension including zero, i.e. point terms --
    ``.venv/.../pyoomph/expressions/coordsys.py:409-422``), so the natural
    boundary term this class contributes is ``2*pi*R(t)*value*v(tip)``:
    with ``value = -2`` that is the per-radian pull ``2*gamma*R`` of S&B's
    leading-order closure, obtained without this class ever touching
    ``R(t)``.

    ``tip_azimuthal_correction`` (default False): adds the O(h/R) toroidal-
    rim term from ``docs/axi-design.md`` -- ``N_rr(R) = -2 + pi*h_tip/(2*R)``
    -- evaluated at the tip node, where ``h`` (C1 here) has a genuine nodal
    value. It exists to measure that correction's size against a DNS, not
    to assert it belongs in the leading-order closure.
    """

    def __init__(self, mu: float, tip_azimuthal_correction: bool = False):
        super().__init__()
        self.mu = mu
        self.tip_azimuthal_correction = tip_azimuthal_correction

    def define_residuals(self):
        # = -2, same constant as planar TipClosure / mass-coordinate
        # TipFlux; kept in this mu-cancelling form so the derivation stays
        # visible in the code, not just the docstring.
        value = 4.0 * self.mu * (-1.0 / (2.0 * self.mu))
        if self.tip_azimuthal_correction:
            value = value + pi * var("h") / (2.0 * var("coordinate_x"))
        self.add_residual(value * weak(1, testfunction("u")))


class Trouton1DALEAxiProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Oh = 1.0
        self.r0 = 5.0               # initial hole radius R0, units of h0
        self.l0 = 500.0             # radial extent of film beyond r0
        self.nelem = 800
        self.grade = 2.0            # r = r0 + l0 (j/N)^grade; 1 = uniform
        self.ale = True             # False only to measure the correction's size
        self.tip_azimuthal_correction = False
        self.default_timestepping_scheme = "BDF2"

    @property
    def mu(self) -> float:
        return self.Oh * math.sqrt(2.0)

    @property
    def tau_vis(self) -> float:
        return self.mu / 2.0

    @property
    def r0_star(self) -> float:
        """Savva-Bush small-hole parameter R0* = R0/(Oh h0), h0 = 1.

        R0* << 1 is the regime where R(t) = R0 exp(t*/2) (their 3.30); R0* >>
        1 is the planar-limit regime (docs/axi-design.md, "Internal exact
        checks" #6).
        """
        return self.r0 / self.Oh

    def define_problem(self):
        # THE ONE LINE THAT MAKES THIS SOLVER AXISYMMETRIC. Every weak()
        # term on this domain -- bulk and point alike -- gains the 2*pi*r
        # measure from here; see the module docstring.
        self.set_coordinate_system("axisymmetric")

        self += LineMesh(N=self.nelem, size=self.l0, nodal_dimension=1,
                         name="sheet")

        eqs = TroutonSheetALEAxi(mu=self.mu, ale=self.ale)

        # THE MESH-MOTION EQUATION, with its OWN cartesian coordinate system
        # (its constructor default, not overridden). Laplace smoothing does
        # not need to know about the physical measure; it produces the
        # affine map of the OFFSET Lagrangian coordinate regardless, exactly
        # as derived in the module docstring.
        eqs += LaplaceSmoothedMesh()

        eqs += InitialCondition(h=1, u=0)
        eqs += TipClosure(
            mu=self.mu,
            tip_azimuthal_correction=self.tip_azimuthal_correction) @ "left"

        # Kinematic condition dR/dt = u(R, t). EnforcedBC, not DirichletBC:
        # the prescribed mesh position depends on the unknown velocity.
        eqs += EnforcedBC(mesh_x=mesh_velocity()[0] - var("u")) @ "left"

        # Far end: undisturbed sheet at rest, mesh pinned at r0 + l0. Both
        # needed -- u=0 alone leaves the Laplace mesh problem with a null
        # space (see trap 2 in trouton1d_ale.py's module docstring).
        eqs += DirichletBC(u=0, mesh_x=True) @ "right"

        eqs += TextFileOutput()
        self += eqs @ "sheet"

    def initialise(self, *args, **kwargs):
        super().initialise(*args, **kwargs)
        self._offset_and_grade_mesh_at_tip()

    def _offset_and_grade_mesh_at_tip(self) -> None:
        """Offset [0, l0] to [r0, r0+l0] and grade it, in BOTH x and x_lagr.

        Combined into one pass deliberately: see the module docstring's
        "THE OFFSET DOMAIN" section for why the offset must go through the
        identical set_x/set_x_lagr code path as the grading trap in
        trouton1d_ale.py:407-422, rather than via ``LineMesh(minimum=...)``.
        The offset applies even when ``grade == 1`` (uniform mesh); only the
        grading exponent is optional.

        Safe only because the initial condition is uniform (h=1, u=0), so
        there is no field data on the old node positions that would need
        re-interpolating.
        """
        mesh = self.get_mesh("sheet")
        L = float(self.l0)
        R0 = float(self.r0)
        grade = float(self.grade)
        for node in mesh.nodes():
            xi = node.x(0) / L
            if grade != 1.0:
                xi = xi ** grade
            xn = R0 + L * xi
            node.set_x(0, xn)
            node.set_x_lagr(0, xn)

    # ---- diagnostics --------------------------------------------------------

    def _sorted_nodes(self):
        return sorted(self.get_mesh("sheet").nodes(), key=lambda nd: nd.x(0))

    def _idx(self):
        return self.get_mesh("sheet").get_nodal_field_indices()

    def edge_position(self) -> float:
        """R(t), the hole radius -- planar's ``x_c`` renamed to its axi role.

        Read directly off the mesh, exactly as planar's ``edge_position``:
        R is a mesh degree of freedom here (via the kinematic EnforcedBC),
        not a time integral of the tip velocity.
        """
        return float(self._sorted_nodes()[0].x(0))

    def tip_velocity(self) -> float:
        return float(self._sorted_nodes()[0].value(self._idx()["u"]))

    def tip_thickness(self) -> float:
        return float(self._sorted_nodes()[0].value(self._idx()["h"]))

    def mesh_affinity_residual(self) -> float:
        """max |r_j - [R + span*(X_j - r0)/l0]| / span, span = (r0+l0) - R.

        Axisymmetric analogue of planar's ``mesh_affinity_residual``: the
        affine map now anchors the moving end at ``R(t)`` (instead of
        ``x_c``) and the pinned end at ``r0 + l0`` (instead of ``l0``); see
        the module docstring for the derivation with the offset carried
        through. A non-zero value means the mesh-motion law is not the one
        documented there.
        """
        nodes = self._sorted_nodes()
        R = nodes[0].x(0)
        far = self.r0 + self.l0
        span = far - R
        if span <= 0.0:
            return float("nan")
        L = float(self.l0)
        R0 = float(self.r0)
        return max(abs(nd.x(0) - (R + span * (nd.x_lagr(0) - R0) / L))
                   for nd in nodes) / span

    def thickness_roughness(self) -> float:
        """Checkerboard indicator for h, identical to planar's.

        Unaffected by axisymmetry: it measures each node's departure from
        the linear interpolant of its two neighbours, as a fraction of the
        profile's range. See ``trouton1d_ale.py``'s
        ``thickness_roughness`` for the full rationale (in particular why a
        naive second difference will not do on a graded mesh).
        """
        nodes = self._sorted_nodes()
        ih = self._idx()["h"]
        xs = [nd.x(0) for nd in nodes]
        hs = [nd.value(ih) for nd in nodes]
        rng = max(hs) - min(hs)
        if rng <= 0.0:
            return 0.0
        worst = 0.0
        for j in range(1, len(hs) - 1):
            span = xs[j + 1] - xs[j - 1]
            if span <= 0.0:
                continue
            lin = hs[j - 1] + (hs[j + 1] - hs[j - 1]) * (xs[j] - xs[j - 1]) / span
            worst = max(worst, abs(hs[j] - lin))
        return worst / rng

    def far_field_drift(self) -> tuple[float, float]:
        """|h - 1| and |u| in the outer tenth of the domain.

        Probed interior to the boundary, as in planar: u is pinned at the
        far end, so its boundary value says nothing about whether l0 is
        adequate.
        """
        nodes = self._sorted_nodes()
        idx = self._idx()
        j = int(0.9 * (len(nodes) - 1))
        dh = max(abs(nd.value(idx["h"]) - 1.0) for nd in nodes[j:])
        du = max(abs(nd.value(idx["u"])) for nd in nodes[j:])
        return dh, du

    def tip_cell_size(self) -> float:
        nodes = self._sorted_nodes()
        return float(nodes[1].x(0) - nodes[0].x(0))

    def total_mass(self) -> float:
        """2*pi*int h r dr by the trapezoidal rule; conserved exactly by
        the model (docs/axi-design.md, "Internal exact checks" #4).
        """
        nodes = self._sorted_nodes()
        ih = self._idx()["h"]
        m = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            ha, hb = a.value(ih), b.value(ih)
            ra, rb = a.x(0), b.x(0)
            m += 0.5 * (ha * ra + hb * rb) * (rb - ra)
        return 2.0 * math.pi * m
