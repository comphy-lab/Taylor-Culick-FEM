"""TC1D-ALE: 1D Trouton sheet retraction on a genuinely moving mesh.

WHY A FOURTH 1D SOLVER, WHEN THREE ALREADY AGREE.
-------------------------------------------------
The answer to this problem is not in doubt. TC1D-FEM (``trouton1d_pyoomph.py``,
pyoomph FEM in the Lagrangian mass coordinate), ABDS-CN
(``validation/ahsan_thinfilm.py``, a port of Ahsan et al.'s finite differences in
the physical coordinate with an algebraic stretching map) and TC1D-MASS
(``validation/tc1d_mass.py``, the nonlinear diffusion ``H_T = H^2 H_mm`` on a
fixed mass domain) agree on ``u*(t*=30) = 0.98073`` and on the late-time
coefficient ``3/2`` to six digits. A fourth solver adds nothing to that.

What it adds is a test of the MACHINERY. The 2D DNS in this repository
(``simulationCases/planar_comoving.py``) is an ALE calculation: the free surface
is sharp, the mesh deforms with it, and every inertial term is written as
``partial_t(..., ALE="auto")`` against a ``BaseMovingMeshEquations`` mesh-motion
field. That moving-mesh path has no independent verification -- it is exercised
only on the 2D problem whose answer it is supposed to produce. This solver puts
the same pyoomph machinery, on the same physical coordinate, on a problem whose
answer is known to five digits from three directions. If the ALE path is wrong,
this is where it shows up cheaply.

It is therefore the methodological bridge: TC1D-FEM (1D, fixed mass domain, no
mesh motion) -> TC1D-ALE (1D, moving physical mesh) -> 2D co-moving DNS
(deforming physical mesh in two dimensions).

THE MODEL.
----------
Planar sheet of FULL thickness ``h``, physical coordinate ``x``, with
``rho = gamma = h0 = 1`` and ``mu = Oh_SB sqrt(2)``:

    h_t + (h u)_x = 0
    u_t + u u_x   = (4 mu / h) (h u_x)_x

Capillarity is off, matching all three reference solvers: Savva & Bush close the
tip with a prescribed axial force which IS the capillary pull of the massless
cap, so a bulk ``kappa_x`` would double-count it (see the long note in
``trouton1d_pyoomph.py``).

The liquid occupies ``[x_c(t), l0]``. The edge is free and retracts into the
sheet, so ``x_c`` INCREASES and the liquid domain SHRINKS -- the mass that leaves
the domain does not leave the sheet, it piles into the rim, whose thickness grows
as ``1 + t*/4``.

THE EDGE CLOSURE, DERIVED RATHER THAN COPIED.
---------------------------------------------
For a massless cap the longitudinal force balance is ``N + 2 gamma = 0`` with
``N = 4 mu h u_x`` the Trouton tension of a planar sheet of full thickness ``h``.
The factor 2 on gamma is the sheet's TWO surfaces. Hence

    4 mu h u_x|_{x_c} = -2 gamma   <=>   h u_x|_{x_c} = -1/(2 mu).

Ahsan et al.'s (2.18a) ``H V_X = -1`` is written on the HALF thickness, so it
maps to the same thing: with ``X = x/l_I``, ``l_I = 4 Oh_SB``, ``V = u/sqrt(2)``
and ``H = h``, ``H V_X = 2 mu h u_x``, and ``= -1`` gives ``h u_x = -1/(2 mu)``.
Both routes agree, and both agree with what TC1D-FEM imposes in the mass
coordinate, ``h^2 u_m|_0 = -1/(2 mu)``, because ``d/dm = (1/h) d/dx``.

It enters as a NATURAL boundary term, exactly as in TC1D-FEM. Multiplying the
momentum equation by ``h``, testing with ``v`` and integrating the viscous term
by parts,

    R(v) = int h (u_t + u u_x) v dx + 4 mu int h u_x v_x dx - 4 mu [h u_x v]

and the bracket is ``h u_x v|_{l0} - h u_x v|_{x_c}``. The far end is Dirichlet
so ``v(l0) = 0``, leaving ``+4 mu h u_x|_{x_c} v(x_c) = -2 v(x_c)``. The constant
is ``-2`` for every ``Oh``: the ``mu`` of the viscous prefactor cancels the ``mu``
of the closure. It is the SAME NUMBER as ``TipFlux`` in the mass-coordinate
solver, which is a non-trivial consistency check of the two formulations rather
than a coincidence.

WHAT ALE MEANS HERE.
--------------------
Mesh nodes move with a velocity ``w`` that is neither the material velocity nor
zero, so a derivative taken at a fixed MESH point is not the Eulerian one:

    d/dt|_mesh = d/dt|_x + w d/dx.

pyoomph's ``partial_t(f, ALE="auto")`` returns ``d f/dt|_x``, i.e. it subtracts
``w f_x`` from the nodal derivative -- but ONLY once mesh-motion equations exist
on the domain, which is the trap recorded in the repository ``AGENTS.md``. With
that in hand the residuals are written in their plain Eulerian form and the
correction is pyoomph's business, which is precisely the code path the 2D case
depends on.

THE MESH-MOTION CHOICE: AFFINE STRETCH, AS AN EQUATION.
------------------------------------------------------
At the free edge the mesh MUST be material, ``w = u``, or the boundary stops
being the interface. Everywhere else ``w`` is free. The choice here is the affine
stretch of ``[x_c(t), l0]``,

    w(x) = u(x_c) (l0 - x) / (l0 - x_c),

and it is obtained as the solution of an EQUATION solved simultaneously with the
physics, not by renoding between steps. ``LaplaceSmoothedMesh`` imposes
``d^2(x - X)/dX^2 = 0`` in the Lagrangian coordinate ``X``, whose solution with
``x(0) = x_c(t)`` and ``x(l0) = l0`` is exactly ``x = x_c + (l0 - x_c) X/l0``,
i.e. the affine map, and differentiating it in time gives the ``w`` above. Two
consequences worth stating:

  * it preserves the GRADING of the initial mesh, because affinity is in ``X``,
    not in the node index. The mesh is graded to ``x = l0 (j/N)^p``, so the tip
    cell stays the smallest cell and stays at the tip;
  * the mesh really does move non-materially. Away from the edge ``u`` decays to
    zero over ``O(l_I)`` while ``w`` decays over ``O(l0 - x_c)``, so ``u - w`` is
    ``O(u_tip)`` across most of the domain. This is not a Lagrangian scheme
    wearing a different label; the advective ALE terms are the same size as the
    physical ones. That is what makes it a test.

The edge condition ``w = u`` is the kinematic condition ``dx_c/dt = u(x_c,t)``.
It is imposed with ``EnforcedBC``, not ``DirichletBC``: the position it prescribes
depends on an unknown, which is the first pyoomph trap in the repository
``AGENTS.md``, and ``EnforcedBC`` imposes it through a Lagrange multiplier.

DISCRETISATION.
---------------
``h``, ``u`` and the mesh position are all C1 on a ``LineMesh``, whose elements
are ``add_line_1d_C1`` -- linear, with NO midside nodes anywhere in the mesh. The
non-convergent co-located mode documented at length in ``trouton1d_pyoomph.py``
lives on the midside nodes of a quadratic velocity field; a C1/C1 pair on a
linear mesh cannot carry it, which is why co-location is safe here and was not
there. The workspace rule "no midside velocity DOFs" is satisfied by
construction. Time integration is BDF2, as in TC1D-FEM.

The thickness cannot be staggered onto ``D0`` here the way it is in the mass
coordinate. There, ``h_t|_m = -h^2 u_m`` contains no ``h`` derivative at all, so a
cell-constant ``h`` is a complete discretisation. Here the ALE mass equation
carries ``(u - w) h_x`` with ``u - w`` of order ``u_tip``, so a cell-constant
``h`` would need an upwind numerical flux to say anything at all. Continuous
Galerkin is used instead, which for a pure advection equation is a real risk,
so the result is checked for the checkerboard that would signal it was the wrong
call (``thickness_roughness`` below). Measured at Oh_SB = 1, t* = 30, the
indicator is 0.117, 0.0333, 8.68e-3, 2.19e-3, 5.49e-4 and 1.37e-4 at N = 100 to
3200, i.e. it falls as dx^2 with no floor. That is truncation error, not a mode:
a checkerboard would be mesh-independent and would drive the indicator towards
one. No stabilisation is used, and none is warranted.

TRAPS HIT WHILE WRITING THIS, recorded because none of them announce themselves.
------------------------------------------------------------------------------
  1. ``InitialCondition`` on a moving-mesh domain does not set the mesh. The
     grading must be applied to BOTH ``x`` and ``x_lagr`` after initialisation,
     as in TC1D-FEM -- and here ``x_lagr`` is not bookkeeping: it is the ``X``
     the Laplace mesh equation is written in, so leaving it uniform silently
     changes the mesh-motion law from "affine stretch of the graded mesh" to
     "affine stretch of a uniform mesh interpolated onto graded data".
  2. The far end must pin the mesh AND the velocity. Pinning only ``u`` leaves
     the Laplace mesh problem with a Neumann condition at both ends, hence a
     rigid-body null space in the mesh position; the Newton solve then fails on
     a singular Jacobian with nothing wrong in the physics.
  3. The physical coordinate loses resolution where the mass coordinate gains
     it. The rim occupies ``Delta m ~ 1`` of mass, so its physical extent is
     ``~ 1/H = 1/(1+T)`` and it NARROWS as the run proceeds, while the affine
     stretch only compresses the mesh by ``(l0 - x_c)/l0``. Resolution at the tip
     must therefore be bought up front through the grading exponent, and the
     converged answer at late ``T`` costs more here than in TC1D-FEM. This is a
     real property of the coordinate, not of the implementation, and it is the
     reason Ahsan et al. use an algebraic stretching map in the same coordinate.

VERIFICATION, Oh_SB = 1. Reproduce with ``validation/trouton1d_ale_sweep.py``
and read with ``validation/trouton1d_ale_report.py``.
------------------------------------------------------------------------------
1. CONVERGENCE ORDERS, measured with the two discretisations varied
   independently. Temporal, at N = 800, nsteps 400 -> 12800: successive
   differences in u*(30) fall by 3.78, 3.92, 3.96, 3.97, i.e. observed order
   1.92 rising to 1.99. That is BDF2 behaving as BDF2. Spatial, at
   nsteps = 12800, N = 100 -> 3200: the ratio is 4.000 at every level, observed
   order 2.000, which is the C1 element behaving as a C1 element. The moving
   mesh costs no order in either.

2. u*(t* = 30). Richardson limits 0.9807279456 (dt -> 0) and 0.9807278002
   (dx -> 0), agreeing to 1.5e-7. Both give 0.98073, the value the other three
   solvers give. Against TC1D-FEM's own dt -> 0 limit of 0.9807280867 the
   relative difference is 2.9e-7, i.e. the two agree to their joint
   discretisation floor.

3. THE LATE-TIME COEFFICIENT R = (1+T)^2 (1 - dXc/dT), which is the sharpest
   test available because R multiplies a small difference by (1+T)^2: at
   T = 100 an error of 1e-7 in u* is 1e-3 in R, twice the tolerance the
   coefficient is quoted to. Neither discretisation may be assumed converged, so
   both are laddered:

       T = 50    N = 400, 800, 1600 at fixed dt   1.497765, 1.498390, 1.498546
       T = 100   same                             1.502868, 1.500486, 1.499891

   Both ladders halve their differences by 4.00, confirming second order at late
   T as well, and extrapolate to R(50) = 1.49860 and R(100) = 1.49969 against
   the three-solver values 1.498623 and 1.499660 -- 2.5e-5 and 3.3e-5 apart.
   Continued to T = 200 on an independent pair of meshes, R extrapolates to
   1.49860, 1.49964 and 1.50000 at T = 50, 100 and 200, against 1.498623,
   1.499660 and 1.499919.

   The two error sources are cleanly separated, which is what makes those
   extrapolations trustworthy. At fixed N = 1600, doubling nsteps moves R by
   1.8e-5 at T = 100 and 6.2e-5 at T = 200; at fixed nsteps, halving dx moves it
   by 2.7e-4 and 4.4e-3. The residual is spatial by one to two orders of
   magnitude, so extrapolating in the mesh is extrapolating in the term that
   dominates. The measurement is therefore

       lim_{T->inf} (1+T)^2 (1 - dXc/dT) = 1.5000 +/- 0.0005,

   the same as the other three solvers, with unity excluded by three orders of
   magnitude.

4. THE DISPLACEMENT LAW, which this solver tests more directly than any of the
   others because x_c is a degree of freedom rather than a time integral of the
   tip velocity. Against Xc = T - 1 + (3/2)/(1+T), at N = 1600 the residual is
   2.4e-6 at T = 50 and 9.7e-6 at T = 100 on an Xc of 49 and 99, i.e. 1e-7
   relative, and it falls as dx^2 (4.9e-5, 1.9e-5, 9.7e-6 at N = 400, 800,
   1600 at T = 100). Equivalently T - Xc = 0.985139 against the predicted
   0.985149 at T = 100, and 0.992523 against 0.992537 at T = 200. The O(1) term
   of that law is -1 for any coefficient, so this is an independent check of the
   grouping rather than a second look at the same number.

5. THE TIP-THICKNESS LAW h(x_c) = 1 + t*/4, which the solver is never told:
   relative error 2.1e-6 at t* = 30, N = 800, falling as dx^2 to 1.3e-7 at
   N = 3200, and 2.8e-6 at t* = 400 where h_tip = 101.

6. THE MOVING MESH IS THE INTENDED ONE. Node positions depart from the affine
   stretch by at most 6e-15 of the domain length over every run, so
   ``LaplaceSmoothedMesh`` in 1D is the affine map to machine precision, as
   claimed above rather than assumed. Mass is conserved to 3e-10 at the finest
   timestep, and the far-field probe stays within 1e-6 of the undisturbed state.

7. THE ALE CORRECTION IS LOAD-BEARING, which matters because a solver that
   happened to be right with the correction switched off would be testing
   nothing. Rerunning t* = 30 with ``--no-ale`` -- the same moving mesh, the
   same everything, with ``partial_t`` taken at the node instead of at the
   fixed point -- gives a relative mass drift of 15.6 and a tip thickness 16.6
   times the exact value. The correction is not a refinement of this answer; it
   is the difference between this answer and nonsense.

8. AGAINST TC1D-FEM OVER THE WHOLE CURVE, not only at the checkpoints: run to
   t* = 400 side by side, max |du*| = 4.9e-5 and RMS 1.2e-5 over 399 samples,
   the maximum sitting at the earliest times where the sqrt(t) start-up
   dominates. The difference does not shrink when TC1D-ALE is refined (3.7e-5 at
   N = 800 against 4.9e-5 at N = 1600), so it is set by the reference run's own
   discretisation rather than by this one, and it is the same size as the 7.1e-5
   at which TC1D-FEM and ABDS-CN meet their joint floor.

   That comparison is nevertheless worth less than the three above, and should
   be read as such: TC1D-FEM shares this solver's framework, so agreement rules
   out an error in the ALE formulation but cannot rule out one in pyoomph. The
   independent evidence is items 2-5, against codes that share nothing.

9. DOMAIN TRUNCATION. l0 = 100, 150 and 250 give u*(30) within 1.9e-7 of one
   another. That spread is not a pure truncation effect: the graded mesh ties
   the tip cell to l0/N^2, so changing l0 at fixed N also changes the
   resolution, and the variation is consistent with the measured spatial error
   alone.
"""

from __future__ import annotations

import math
import os

# Must precede the first pyoomph (hence MPI) import; see planar_comoving.py for
# why hwloc's OpenGL backend hangs MPI_Init on a host with a running compositor.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

from pyoomph import (DirichletBC, Equations, InitialCondition,  # noqa: E402
                     Problem, TextFileOutput)
from pyoomph.equations.ALE import LaplaceSmoothedMesh  # noqa: E402
from pyoomph.expressions import (grad, mesh_velocity, partial_t,  # noqa: E402
                                 testfunction, var, weak)
from pyoomph.meshes.bcs import EnforcedBC  # noqa: E402
from pyoomph.meshes.simplemeshes import LineMesh  # noqa: E402

V_TC = math.sqrt(2.0)


class TroutonSheetALE(Equations):
    """Long-wave sheet equations in the physical coordinate, ALE form.

    The residuals are written in their plain Eulerian form; the mesh-motion
    correction is supplied by ``partial_t(..., ALE="auto")`` and is therefore
    only present when a ``BaseMovingMeshEquations`` has been added to the same
    domain. ``ale=False`` drops the correction deliberately, which turns the
    solver into a wrong one on a moving mesh -- it exists so that the size of
    the correction can be measured rather than assumed.
    """

    def __init__(self, mu: float, ale: bool = True):
        super().__init__()
        self.mu = mu
        self.ale = ale

    def define_fields(self):
        # C1 FOR BOTH, AND THE SPACE IS NOT A KNOB.
        #
        # Not C2: the workspace rule for long-time 1D production is "no midside
        # velocity DOFs, C2/C2 is legacy-only", and the non-convergent
        # co-located mode measured in trouton1d_pyoomph.py lives precisely on
        # the midside nodes of a quadratic velocity field. A LineMesh is built
        # from add_line_1d_C1, so at C1 the mesh has no midside nodes at all and
        # the mode has nowhere to live.
        #
        # Not D0 for h either, which is what the mass coordinate uses. There
        # h_t|_m = -h^2 u_m contains no h derivative, so a cell-constant h is a
        # complete discretisation; here the ALE mass equation carries
        # (u - w) h_x with u - w of order u_tip, and a cell-constant h would
        # need an upwind numerical flux to mean anything.
        self.define_scalar_field("h", "C1")
        self.define_scalar_field("u", "C1")

    def define_residuals(self):
        h, u = var("h"), var("u")
        ht, ut = testfunction("h"), testfunction("u")
        h_x, u_x = grad(h)[0], grad(u)[0]
        ale = "auto" if self.ale else False

        # 1. mass: h_t|_x + (h u)_x = 0, expanded because h is C1 and both
        #    products are then differentiable termwise. No integration by parts,
        #    so this equation contributes no boundary term and needs no
        #    condition at either end.
        self.add_residual(weak(partial_t(h, ALE=ale) + h * u_x + u * h_x, ht))

        # 2. momentum, multiplied through by h so that the viscous operator is
        #    the conservative (h u_x)_x whose natural boundary term is the edge
        #    closure. The u u_x term is the material-derivative advection and is
        #    NOT the ALE correction -- partial_t already removed the mesh motion,
        #    so what remains here is physics.
        self.add_residual(weak(h * (partial_t(u, ALE=ale) + u * u_x), ut))
        self.add_residual(4.0 * self.mu * weak(h * u_x, grad(ut)[0]))


class TipClosure(Equations):
    """The free-edge stress condition as a natural boundary term.

    See the module docstring for the derivation of the constant. It is exactly
    ``4 mu * (-1/(2 mu)) = -2`` for every Oh, and it is the same constant as
    ``TipFlux`` in the mass-coordinate solver -- the two formulations must agree
    on it because ``h^2 u_m = h u_x``.
    """

    def __init__(self, mu: float):
        super().__init__()
        self.value = 4.0 * mu * (-1.0 / (2.0 * mu))

    def define_residuals(self):
        self.add_residual(self.value * weak(1, testfunction("u")))


class Trouton1DALEProblem(Problem):
    def __init__(self):
        super().__init__()
        self.Oh_SB = 1.0
        self.l0 = 500.0            # truncation length of the semi-infinite sheet
        self.nelem = 800
        self.grade = 2.0           # x = l0 (j/N)^grade; 1 = uniform
        self.ale = True            # False only to measure the correction's size
        self.default_timestepping_scheme = "BDF2"

    @property
    def mu(self) -> float:
        return self.Oh_SB * math.sqrt(2.0)

    @property
    def tau_vis(self) -> float:
        return self.mu / 2.0

    @property
    def l_I(self) -> float:
        """Ahsan et al.'s inertial-viscous length, l_I = 4 h0 Oh_paper = 4 Oh_SB.

        The conversion to their variables is X = x/l_I, T = t*/4, V = u*; see
        ``validation/ahsan_thinfilm.py`` for the full mapping and its
        independent confirmation from h_tip = 1 + t*/4.
        """
        return 4.0 * self.Oh_SB

    def define_problem(self):
        self.set_coordinate_system("cartesian")

        self += LineMesh(N=self.nelem, size=self.l0, nodal_dimension=1,
                         name="sheet")

        eqs = TroutonSheetALE(mu=self.mu, ale=self.ale)

        # THE MESH-MOTION EQUATION. Without a BaseMovingMeshEquations on this
        # domain the coordinates are not degrees of freedom, partial_t's ALE
        # correction is silently absent, and the run produces a plausible wrong
        # answer on a mesh that never moves.
        eqs += LaplaceSmoothedMesh()

        eqs += InitialCondition(h=1, u=0)
        eqs += TipClosure(mu=self.mu) @ "left"

        # Kinematic condition dx_c/dt = u(x_c). EnforcedBC and not DirichletBC:
        # the prescribed position depends on an unknown.
        eqs += EnforcedBC(mesh_x=mesh_velocity()[0] - var("u")) @ "left"

        # Far end: undisturbed sheet at rest, and the mesh pinned. Both are
        # needed -- u=0 alone leaves the mesh problem with a null space.
        # h is left free: the mass equation is its own equation there, and the
        # domain is long enough that h stays at 1, which far_field_drift checks.
        eqs += DirichletBC(u=0, mesh_x=True) @ "right"

        eqs += TextFileOutput()
        self += eqs @ "sheet"

    def initialise(self, *args, **kwargs):
        super().initialise(*args, **kwargs)
        self._grade_mesh_at_tip()

    def _grade_mesh_at_tip(self) -> None:
        """Cluster nodes at the free edge, in BOTH x and x_lagr.

        x_lagr is the coordinate the Laplace mesh equation is written in, so the
        two must be set together; see trap 1 in the module docstring. Safe only
        because the initial condition is uniform (h = 1, u = 0).
        """
        if self.grade == 1.0:
            return
        mesh = self.get_mesh("sheet")
        L = float(self.l0)
        for node in mesh.nodes():
            xi = node.x(0) / L
            xn = L * xi ** self.grade
            node.set_x(0, xn)
            node.set_x_lagr(0, xn)

    # ---- diagnostics --------------------------------------------------------

    def _sorted_nodes(self):
        return sorted(self.get_mesh("sheet").nodes(), key=lambda nd: nd.x(0))

    def _idx(self):
        return self.get_mesh("sheet").get_nodal_field_indices()

    def edge_position(self) -> float:
        """x_c, read directly off the mesh.

        This is the point of the solver as a check on the displacement law: the
        other three codes obtain x_c by integrating the tip velocity in time,
        so their X_c inherits the time integrator's error. Here it is a degree
        of freedom.
        """
        return float(self._sorted_nodes()[0].x(0))

    def tip_velocity(self) -> float:
        return float(self._sorted_nodes()[0].value(self._idx()["u"]))

    def tip_thickness(self) -> float:
        return float(self._sorted_nodes()[0].value(self._idx()["h"]))

    def mesh_affinity_residual(self) -> float:
        """max |x_j - [x_c + (l0 - x_c) X_j/l0]| / (l0 - x_c).

        The affine stretch is an ASSERTION about what LaplaceSmoothedMesh does in
        1D, so it is checked rather than believed. A non-zero value means the
        mesh-motion law is not the one documented above.
        """
        nodes = self._sorted_nodes()
        xc = nodes[0].x(0)
        L = float(self.l0)
        span = L - xc
        if span <= 0.0:
            return float("nan")
        return max(abs(nd.x(0) - (xc + span * nd.x_lagr(0) / L))
                   for nd in nodes) / span

    def thickness_roughness(self) -> float:
        """Checkerboard indicator for h, as a fraction of the profile's range.

        The failure mode a co-located thickness would show is a sign-alternating
        node-to-node mode, so what is measured is each node's departure from the
        linear interpolant of its two NEIGHBOURS (which the node itself does not
        enter). For a smooth resolved profile that departure is the local
        truncation error, O(dx^2 h''), and vanishes with the mesh; for a
        checkerboard of amplitude a it is a, independent of the mesh.

        A naive second difference will NOT do here. On the graded mesh the cell
        size doubles and trebles over the first few cells, so even an exactly
        linear h has a large second difference -- that measures the grading, not
        the solution, and it reads ~0.6 on a profile with no oscillation at all.
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

        Probed INTERIOR to the boundary: u is pinned at x = l0, so its boundary
        value says nothing about whether the truncation length is adequate.
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
        """int h dx by the trapezoidal rule; conserved exactly by the model."""
        nodes = self._sorted_nodes()
        ih = self._idx()["h"]
        m = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            m += 0.5 * (a.value(ih) + b.value(ih)) * (b.x(0) - a.x(0))
        return m
