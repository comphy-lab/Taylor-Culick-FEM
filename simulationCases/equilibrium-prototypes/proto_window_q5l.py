# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Window prototype Q5L: plug window on a long reference grid, acceptance version l.

Window formulation. The window keeps the sheet columns of the reference grid from an inner column i_c (reference
radius X_c) to the outer cut X_m; the cap and the material just behind it, which the rim crushes into the nose, are
removed from the computation, and their effect is kept as follows.

- Bulk: u = b_a(X; K) e_r + b_n + v, with b_a = sqrt(X^2 + K) - X the analytic area-preserving map (exact gradient),
  K = R^2 - r0^2 a global parameter, b_n a pinned nodal correction and v the unknown; F, cof F and J evaluated
  directly; the volume penalty of the seed case; no cap stiffening (there is no cap).
- Outer cut X_m: the depth-averaged membrane traction of the universal decaying branch, plus the continuation of
  the free surface.
- Inner cut (the column X = X_c): a plug. Its deformed shape is frozen at the start: the unknown part of the
  displacement is constrained to v = (Delta, 0) on the column by Lagrange multipliers lambda, and the radial
  translation Delta is one global unknown whose equation is the plug's radial balance
      -int lambda_r dA + 2 pi (G I_c r + c_nose) = 0,
  with I_c = int dX dZ / X over the removed region (Gauss quadrature over the removed grid elements) the hoop moment
  of the removed material. The removed free surface lies inside the plug, so its pull on the kept corner is
  internal, and the plug's vertical load is a reaction. The nose constant c_nose (per radian) collects what the
  hoop term omits (the removed surface's hoop and the nose pressure); it is a global parameter calibrated at the
  start by a secant so that the converged window puts the inner column's midplane node where the start state has it.
- The only approximation of the plug is its frozen shape: exact at the start, and in error by the change of the
  nose shape as Ec moves. The error falls with chi_min = X_c / sqrt(R); a family of cuts measures and extrapolates it.
- Re-plugging: START may be a converged window state with an older cut and INNER a column further out; without
  C_NOSE the nose constant starts from 0 and the calibration finds it. C_NOSE (per radian) sets the constant and
  skips the calibration; F_NOSE (the nose's radial force, 2 pi units) sets its first value from a measured force.
- Reference grid: GRIDS (default the seed grid). Nodes beyond the start state's reference extent X_s are initialised
  from its last column, with the deviation from the area-preserving map decaying as X_s/X (radial) and (X_s/X)^2
  (vertical).
- Each saved point records R as r at the inner column's midplane node, and the rim height H.

Continuation in log Ec with steps of at most STEP (default 0.001) and step halving on failure, a secant predictor
in R and in the total nodal displacement before every step (b_n reset to the predictor, v = 0), and an own Newton
loop: SuperLU with three refinement sweeps and backtracking. Residuals are grouped in reference bands of X; the
near region X < FAR_CHI sqrt(R) (default 10) holds the rim and the plug. The round-off floor of a band is estimated
from one random 1e-13 relative perturbation of the dofs and the nodal correction (scaled to machine epsilon).

Acceptance (version Q5l). Newton measures progress, stalls and backtracking on the near residual (X < FAR_CHI sqrt(R)
plus the pressure and multiplier dofs); the far film needs only to lie within FLOOR_FACTOR times its round-off
floor, estimated from 1e-13 relative perturbations of the dofs, the nodal correction and K. Re-splits inside Newton
and in the polish loop are residual-neutral: b_n += v, v = 0, plug shift 0, K unchanged, so the discrete state is
unchanged exactly. A stall below STALL_CAP with the far film at its floor is provisional, and the polish
(reproduction) loop of Q5i decides acceptance. RESPLITS, NEWTON_TOL and HALVINGS as Q5j.

Usage: proto_window_q5l.py <out> <X_m> <target Ec> [...]
env (required): CASE (case file of the seed equilibrium segment), SEED (folder of the retained seed segment, holding
its grids.npz and state.npz).
env (optional): START (saved point folder: a seed-grid or window state; default SEED), GRIDS, INNER (even sheet
column index of the inner cut, default 140), STEP, C_NOSE, F_NOSE, CAL_TOL (default 1e-7), FAR_CHI, TAU_TOL (unused
by the plug), FLOOR_FACTOR, DTOL_Z, DTOL_X, POLISH_MAX, HALVINGS, NEWTON_TOL, STALL_CAP, CAL_TOL_REL, RESPLITS.
Normally driven by march_plug2.py, which sets START and INNER for each segment.
"""

import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

FEM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FEM / "src"))
from pyoomph.generic.jit_cache import set_enabled  # noqa: E402

set_enabled(False)
import numpy as np  # noqa: E402
import scipy.sparse as sparse  # noqa: E402
from pyoomph import DirichletBC  # noqa: E402
from pyoomph.expressions import (exp, grad, log, matproduct, matrix, partial_t, square_root,  # noqa: E402
                                 testfunction, transpose, var, vector)
from pyoomph.expressions.interpol import InterpolateSpline1d  # noqa: E402
from pyoomph.generic.codegen import InterfaceEquations  # noqa: E402
from pyoomph.equations.generic import GlobalLagrangeMultiplier  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

import taylor_culick_fem.depth_averaged_equilibrium as da  # noqa: E402
import taylor_culick_fem.elastocapillary_equilibrium as eq  # noqa: E402
from taylor_culick_fem.equilibrium_branch import SegmentSettings  # noqa: E402
from taylor_culick_fem.equilibrium_geometry import FieldMap  # noqa: E402

CASE = Path(os.environ["CASE"])
SEG = Path(os.environ["SEED"])
FULL = {0.37: (150.3883804984, 1.2936673), 0.35: (205.58378168509037, 1.3179252140280093),
        0.3: (536.6705989797, 1.3896701224106813), 0.25: (2044.166526, 1.48239662)}
BACKGROUND_K = []
PENALTY = []
EPS = 2.220446049250313e-16
BANDS = [(0, 1.5), (1.5, 5), (5, 100), (100, 1000), (1000, 4000), (4000, 2e4), (2e4, 1e5), (1e5, 1e9)]
FAR_CHI = float(os.environ.get("FAR_CHI", "10"))
STALL_CAP = float(os.environ.get("STALL_CAP", "1e-3"))


def polish(p, st, xs, tol, rng, r0):
    """Re-split and re-solve until two successive states agree. Returns (ok, polishes, history)."""
    polishes, history = [], []
    for _ in range(int(os.environ.get("POLISH_MAX", "3"))):
        tx, ty = st.totals()
        rr = st.radius(tx)
        resplit(st)
        ok2, history2, info2 = newton(p, st, xs, tol, rng)
        px, py = st.totals()
        dz = float(np.abs(py - ty).max())
        dx = px - tx
        near_nodes = st.x < FAR_CHI * math.sqrt(max(rr, 1.0))
        dxr = float(np.abs(dx - np.median(dx))[near_nodes].max())
        polishes.append({"history": history2[:10], "dz": dz, "dx_nonrigid": dxr, "dx_rigid": float(np.median(dx))})
        history = history + history2
        sane = bool(history2) and all(math.isfinite(h) for h in history2) and history2[-1] <= 10 * min(history2)
        if not sane:
            return False, polishes, history
        if (ok2 and info2 is None) or (dz <= DTOL_Z and dxr <= DTOL_X):
            return True, polishes, history
    return False, polishes, history
FLOOR_FACTOR = float(os.environ.get("FLOOR_FACTOR", "10"))
DTOL_Z = float(os.environ.get("DTOL_Z", "1e-8"))
DTOL_X = float(os.environ.get("DTOL_X", "1e-6"))
STEP = float(os.environ.get("STEP", "0.001"))
PTS = ((0, 0), (-0.77, -0.77), (0.77, -0.77), (-0.77, 0.77), (0.77, 0.77))


def universal_tension_table():
    prof = da.log_profile(0.05)
    sol = prof.solution if hasattr(prof, "solution") else next(v for v in vars(prof).values() if hasattr(v, "sol"))
    seed = float(sol.t[0])
    ys = np.concatenate([np.linspace(0.0, seed, 5, endpoint=False), np.linspace(seed, prof.log_radius_ratio, 40000)])
    small = ys < seed
    tension = np.empty_like(ys)
    tension[small] = -2 * ys[small] + ys[small] ** 2 - 7 * ys[small] ** 3 / 12
    tension[~small] = sol.sol(ys[~small])[0]
    return np.column_stack([ys, tension])


TENSION = InterpolateSpline1d(universal_tension_table(), k=5)


def analytic_map(x, k):
    return np.sqrt(x * x + k) - x


def _physical_displacement(background=False):
    """Always the full displacement: unknown + pinned nodal correction + analytic map."""
    x = var("coordinate_x")
    return var("displacement") + var("displacement_background") + vector(square_root(x ** 2 + BACKGROUND_K[0]) - x, 0)


eq.physical_displacement = _physical_displacement


class DirectNeoHookean(eq.NeoHookeanDisplacementEquations):
    """Same weak form as the parent, with F, cof F and J evaluated directly (no small-gradient expansion)."""

    def define_residuals(self):
        u = eq.physical_displacement(True)
        pressure = var("pressure")
        g = grad(u)
        a, bb, c, d = 1 + g[0, 0], g[0, 1], g[1, 0], 1 + g[1, 1]
        h = 1 + u[0] / var("coordinate_x")
        det_m = a * d - bb * c
        deformation = matrix([[a, bb, 0], [c, d, 0], [0, 0, h]])
        cofactor = matrix([[d * h, -c * h, 0], [-bb * h, a * h, 0], [0, 0, det_m]])
        volume = h * det_m
        stress = self.shear_modulus * (deformation - cofactor) - pressure * cofactor
        if self.volume_penalty:
            stress = stress + self.shear_modulus * PENALTY[0] * (volume - 1) * cofactor
        if self.stiffening is not None:
            first_invariant = a ** 2 + bb ** 2 + c ** 2 + d ** 2 + h ** 2
            cubic = matproduct(deformation, matproduct(transpose(deformation), deformation))
            stress = stress + self.shear_modulus * self.stiffening * (first_invariant * deformation - cubic - 2 * cofactor)
        self.add_weak(stress, grad(testfunction("displacement")))
        self.add_weak(volume - 1, testfunction("pressure"))
        if self.damping:
            self.add_weak(self.damping * partial_t(u), testfunction("displacement"))
        self.set_Dirichlet_condition("displacement_background_x", True)
        self.set_Dirichlet_condition("displacement_background_y", True)


class MembraneCutTraction(InterfaceEquations):
    required_parent_type = eq.NeoHookeanDisplacementEquations

    def __init__(self, shear_modulus):
        super().__init__()
        self.shear_modulus = shear_modulus

    def define_residuals(self):
        u = eq.physical_displacement(True)
        b = 1 + u[0] / var("coordinate_x")
        self.add_weak(vector(-self.shear_modulus * b * TENSION(log(b)), 0), testfunction("displacement"))


class SheetWindowMesh(eq.ReferenceSheetMesh):
    """The sheet block only, with boundaries symmetry, interface, far and inner (its first column)."""

    def define_geometry(self):
        domain = self.new_domain(self.domain_name)
        registry = {}

        def node(p):
            key = (round(float(p[0]), 11), round(float(p[1]), 11))
            if key not in registry:
                registry[key] = self.add_node_unique(float(p[0]), float(p[1]))
            return registry[key]

        grid = self.grids["sheet"]
        ids = [[node(grid[i, j]) for j in range(grid.shape[1])] for i in range(grid.shape[0])]
        for i in range(0, grid.shape[0] - 1, 2):
            for j in range(0, grid.shape[1] - 1, 2):
                domain.add_quad_2d_C2(*(ids[i + di][j + dj] for dj in range(3) for di in range(3)))

        def edge(name, nodes):
            for k in range(0, len(nodes) - 1, 2):
                facet = nodes[k:k + 3]
                self.add_facet_to_boundary(name, facet, [facet[0], facet[-1]])

        edge("symmetry", [row[0] for row in ids])
        edge("interface", [row[-1] for row in ids])
        edge("far", ids[-1])
        edge("inner", ids[0])


INNER = {}


NOSE_C = []  # global parameter: the calibrated nose constant (per radian)
TAU = []  # global parameters (tau_r, tau_z): deformed surface tangent at the corner, into the removed nose


class InnerNoseTraction(InterfaceEquations):
    """Bulk action of the removed nose: (G I_c + sigma tau_r) r / (S_col X) per boundary measure."""

    required_parent_type = eq.NeoHookeanDisplacementEquations

    def __init__(self, shear_modulus):
        super().__init__()
        self.shear_modulus = shear_modulus

    def define_residuals(self):
        u = eq.physical_displacement(True)
        x = var("coordinate_x")
        r = x + u[0]
        self.add_weak(vector((self.shear_modulus * INNER["I_c"] + INNER["sigma"] * TAU[0]) * r / (INNER["S_col"] * x), 0),
                      testfunction("displacement"))


class CornerPull(InterfaceEquations):
    """Continuation of the free surface into the removed nose: sigma (r/X) tau at the corner (tau lagged)."""

    def __init__(self, surface_tension=1):
        super().__init__()
        self.surface_tension = surface_tension

    def define_residuals(self):
        u = eq.physical_displacement(True)
        hoop = 1 + u[0] / var("coordinate_x")
        self.add_weak(-self.surface_tension * hoop * vector(TAU[0], TAU[1]), testfunction("displacement"))


class PlugColumn(InterfaceEquations):
    """Inner column as a plug: v = (Delta, 0) via multipliers; Delta balances the reactions and the nose hoop."""

    required_parent_type = eq.NeoHookeanDisplacementEquations

    def __init__(self, shear_modulus):
        super().__init__()
        self.shear_modulus = shear_modulus

    def define_fields(self):
        self.define_scalar_field("plug_lx", "C2")
        self.define_scalar_field("plug_ly", "C2")

    def define_residuals(self):
        v = var("displacement")
        lx, ly = var("plug_lx"), var("plug_ly")
        delta = var("plug_shift", domain="globals")
        u = eq.physical_displacement(True)
        x = var("coordinate_x")
        r = x + u[0]
        self.add_weak(v[0] - delta, testfunction("plug_lx"))
        self.add_weak(v[1], testfunction("plug_ly"))
        self.add_weak(vector(lx, ly), testfunction("displacement"))
        self.add_weak(-lx + (self.shear_modulus * INNER["I_c"] * r + NOSE_C[0]) / (INNER["S_col"] * x),
                      testfunction("plug_shift", domain="globals"))


def removed_surface_length(fmap, grids, i_c, n_arc=801, n_sub=5):
    """Deformed length of the removed free surface: the cap arc (centre (1.5, 0), radius 1/2) and the sheet's top
    row from column 0 to i_c, mapped through the start state's field map."""
    th = np.linspace(0, np.pi / 2, n_arc)
    arc = np.column_stack([1.5 - (0.5 - 1e-10) * np.cos(th), (0.5 - 1e-10) * np.sin(th)])
    top = grids["sheet"][:, -1]
    pts = [a + t * (b - a) - [0, 1e-10] for a, b in zip(top[:i_c], top[1:i_c + 1]) for t in np.linspace(0, 1, n_sub + 1)[:-1]]
    pts.append(top[i_c] - [0, 1e-10])
    total = 0.0
    for line in (arc, np.array(pts)):
        d = np.array([p + fmap.forward(np.asarray(p, float)) for p in line])
        total += float(np.sum(np.hypot(*np.diff(d, axis=0).T)))
    return total


def removed_hoop_moment(grids, i_c):
    """I_c = int dX dZ / X over the removed region: core, lower, upper and sheet columns 0..i_c (3x3 Gauss)."""
    gp = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    gw = np.array([5 / 9, 8 / 9, 5 / 9])

    def n1(s):
        return np.array([s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2]), np.array([s - 0.5, -2 * s, s + 0.5])

    total = 0.0
    blocks = [grids["core"], grids["lower"], grids["upper"], grids["sheet"][: i_c + 1]]
    for a in blocks:
        for i in range(0, a.shape[0] - 1, 2):
            for j in range(0, a.shape[1] - 1, 2):
                P = a[i:i + 3, j:j + 3]
                for s, ws in zip(gp, gw):
                    Ns, dNs = n1(s)
                    for t, wt in zip(gp, gw):
                        Nt, dNt = n1(t)
                        x = np.einsum("a,b,abk->k", Ns, Nt, P)
                        xs = np.einsum("a,b,abk->k", dNs, Nt, P)
                        xt = np.einsum("a,b,abk->k", Ns, dNt, P)
                        total += ws * wt * abs(xs[0] * xt[1] - xs[1] * xt[0]) / x[0]
    return total


class BlobProblem(eq.ElastocapillaryEquilibriumProblem):
    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(SheetWindowMesh(self.grids, self.r0, self.triangles, quadratic=True))
        BACKGROUND_K.clear()
        BACKGROUND_K.append(self.define_global_parameter(background_K=0.0))
        PENALTY.clear()
        PENALTY.append(self.define_global_parameter(volume_gamma=self.volume_penalty))
        modulus = exp(self.define_global_parameter(log_Ec=0.0))
        beta = self.define_global_parameter(cap_beta=self.cap_stiffening)
        step = (var("coordinate_x") - self.cap_stiffening_boundary()) / self.cap_stiffening_width
        eqs = DirectNeoHookean(modulus, pressure_space=self.pressure_space, damping=var("coordinate_x") ** -2,
                               stiffening=None, volume_penalty=self.volume_penalty,
                               displacement_background=True)
        eqs += eq.DeformedSurfaceEnergy(self.surface_tension, True) @ "interface"
        eqs += DirichletBC(displacement_y=0) @ "symmetry"
        eqs += MembraneCutTraction(modulus) @ "far"
        eqs += eq.SurfaceContinuation(self.surface_tension, True) @ "interface/far"
        NOSE_C.clear()
        NOSE_C.append(self.define_global_parameter(nose_c=0.0))
        eqs += PlugColumn(modulus) @ "inner"
        eqs += DirichletBC(plug_ly=0) @ "inner/symmetry"
        self.add_equations(GlobalLagrangeMultiplier(plug_shift=0) @ "globals")
        self.add_equations(eqs @ "solid")


def make_problem(settings, grids, folder, log_ec, beta, k):
    p = BlobProblem()
    p.ignore_command_line = True
    p.set_output_directory(str(folder))
    p.r0, p.surface_tension, p.grids, p.quadratic_grids = settings.r0, settings.surface_tension, grids, True
    p.length = float(grids["sheet"][-1, 0, 0])
    p.cap_stiffening, p.volume_penalty = settings.cap_beta, settings.volume_penalty
    p.displacement_background = True
    p.newton_solver_tolerance = settings.newton_tolerance
    p.set_linear_solver("superlu")
    p.set_eigensolver("scipy")
    p.initialise()
    p.get_global_parameter("log_Ec").value = log_ec
    p.get_global_parameter("cap_beta").value = beta
    p.get_global_parameter("background_K").value = k
    return p


class State:
    """Nodal bookkeeping: U = v + b_n + b_a(X; K) on every node of the solid mesh."""

    def __init__(self, p):
        self.p = p
        mesh = p.get_mesh("solid")
        idx = mesh.get_nodal_field_indices()
        self.nodes = list(mesh.nodes())
        self.ix, self.iy = idx["displacement_x"], idx["displacement_y"]
        self.bx, self.by = idx["displacement_background_x"], idx["displacement_background_y"]
        self.x = np.array([nd.x(0) for nd in self.nodes])
        self.z = np.array([nd.x(1) for nd in self.nodes])
        self.k_param = p.get_global_parameter("background_K")
        sx = {(round(nd.x(0), 12), round(nd.x(1), 12)) for nd in p.get_mesh("solid/interface").nodes()}
        sx |= {(round(nd.x(0), 12), round(nd.x(1), 12)) for nd in p.get_mesh("solid/inner").nodes()}
        self.surface = np.array([(round(a, 12), round(b, 12)) in sx for a, b in zip(self.x, self.z)])  # free surface + inner cut
        top = {(round(nd.x(0), 12), round(nd.x(1), 12)) for nd in p.get_mesh("solid/interface").nodes()}
        on_top = [i for i, (a, b) in enumerate(zip(self.x, self.z)) if (round(a, 12), round(b, 12)) in top]
        self.corner3 = sorted(on_top, key=lambda i: self.x[i])[:3]  # corner, mid-side and end node of the first surface element
        inner = {(round(nd.x(0), 12), round(nd.x(1), 12)) for nd in p.get_mesh("solid/inner").nodes()}
        on_inner = [i for i, (a, b) in enumerate(zip(self.x, self.z)) if (round(a, 12), round(b, 12)) in inner]
        self.inner_mid = min(on_inner, key=lambda i: self.z[i])  # midplane node of the inner column

    def totals(self):
        k = self.k_param.value
        ux = np.array([nd.value(self.ix) + nd.value(self.bx) for nd in self.nodes]) + analytic_map(self.x, k)
        uy = np.array([nd.value(self.iy) + nd.value(self.by) for nd in self.nodes])
        return ux, uy

    def radius(self, ux=None):
        """r of the inner column's midplane node."""
        ux = self.totals()[0] if ux is None else ux
        return float(self.x[self.inner_mid] + ux[self.inner_mid])

    def corner_tangent(self):
        """Unit deformed tangent of the free surface at the corner, pointing into the removed nose (quadratic)."""
        ux, uy = self.totals()
        p0, p1, p2 = (np.array([self.x[i] + ux[i], self.z[i] + uy[i]]) for i in self.corner3)
        d = -(-3 * p0 + 4 * p1 - p2)
        return d / np.hypot(*d)

    def load(self, ux, uy, k):
        """b_n = U - b_a(X; K), v = 0, plug shift 0."""
        self.p.get_ode("globals").set_value(plug_shift=0.0)
        self.k_param.value = k
        bn = ux - analytic_map(self.x, k)
        for nd, b1, b2, z in zip(self.nodes, bn, uy, self.z):
            nd.set_value(self.bx, float(b1))
            nd.set_value(self.by, float(b2) if z > 1e-14 else 0.0)
            nd.set_value(self.ix, 0.0)
            nd.set_value(self.iy, 0.0)

    def v_bands(self):
        v = np.hypot([nd.value(self.ix) for nd in self.nodes], [nd.value(self.iy) for nd in self.nodes])
        bn = np.abs([nd.value(self.bx) for nd in self.nodes])
        return ({f"{lo:g}-{hi:g}": float(v[(self.x >= lo) & (self.x < hi)].max()) for lo, hi in BANDS
                 if np.any((self.x >= lo) & (self.x < hi))},
                {f"{lo:g}-{hi:g}": float(bn[(self.x >= lo) & (self.x < hi)].max()) for lo, hi in BANDS
                 if np.any((self.x >= lo) & (self.x < hi))})

    def field_map(self):
        fmap = FieldMap.from_problem(self.p)  # v + b_n
        ue = fmap.ue.copy()
        ue[..., 0] += analytic_map(fmap.Xe[..., 0], self.k_param.value)
        return FieldMap(fmap.Xe, ue, fmap.ij)


def equation_positions(p):
    mesh = p.get_mesh("solid")
    idx = mesh.get_nodal_field_indices()
    xs = np.full(p.ndof(), np.nan)
    for node in mesh.nodes():
        for name in ("displacement_x", "displacement_y"):
            e = node.eqn_number(idx[name])
            if e >= 0:
                xs[e] = node.x(0)
    return xs


def residual_bands(res, xs):
    return {f"{lo:g}-{hi:g}": float(np.abs(res[(xs >= lo) & (xs < hi)]).max())
            for lo, hi in BANDS if np.any((xs >= lo) & (xs < hi))}


def floor_test(p, st, xs, tol, rng):
    """Band-wise test against the round-off floor, perturbing the dofs and the nodal correction by 1e-13."""
    res = np.asarray(p.get_residuals()).copy()
    dofs = np.asarray(p.get_current_dofs()[0]).copy()
    bn = [(nd.value(st.bx), nd.value(st.by)) for nd in st.nodes]
    k_saved = st.k_param.value
    st.k_param.value = k_saved * (1 + 1e-13 * rng.standard_normal())
    p.set_current_dofs(dofs * (1 + 1e-13 * rng.standard_normal(len(dofs))))
    xi = 1 + 1e-13 * rng.standard_normal((len(bn), 2))
    for nd, (b1, b2), f in zip(st.nodes, bn, xi):
        nd.set_value(st.bx, b1 * f[0])
        nd.set_value(st.by, b2 * f[1])
    noise = np.abs(np.asarray(p.get_residuals()) - res)
    for nd, (b1, b2) in zip(st.nodes, bn):
        nd.set_value(st.bx, b1)
        nd.set_value(st.by, b2)
    st.k_param.value = k_saved
    p.set_current_dofs(dofs)
    bands, ok = {}, True
    for lo, hi in BANDS:
        m = (xs >= lo) & (xs < hi)
        if not m.any():
            continue
        r, f = float(np.abs(res[m]).max()), float(noise[m].max()) * EPS / 1e-13
        bands[f"{lo:g}-{hi:g}"] = [r, f]
        ok &= r <= max(tol, FLOOR_FACTOR * f)
    return ok, bands


def top_residuals(p, n=8):
    r = np.asarray(p.get_residuals())
    mesh = p.get_mesh("solid")
    idx = mesh.get_nodal_field_indices()
    where = {}
    for node in mesh.nodes():
        for name in ("displacement_x", "displacement_y"):
            e = node.eqn_number(idx[name])
            if e >= 0:
                where[int(e)] = (name, float(node.x(0)), float(node.x(1)))
    order = np.argsort(-np.abs(r))[:n]
    return [{"res": float(r[e]), **({"field": where[e][0], "X": where[e][1], "Z": where[e][2]}
                                    if int(e) in where else {"field": "pressure/other"})} for e in order]


R0_REF = [1.0]  # settings.r0, set in main


def resplit(st):
    """b_n += v, v = 0 at every node, plug shift 0 (the column's v_x was the shift); K unchanged (residual-neutral)."""
    st.p.get_ode("globals").set_value(plug_shift=0.0)
    for nd in st.nodes:
        nd.set_value(st.bx, nd.value(st.bx) + nd.value(st.ix))
        nd.set_value(st.by, nd.value(st.by) + nd.value(st.iy))
        nd.set_value(st.ix, 0.0)
        nd.set_value(st.iy, 0.0)


def far_ok(p, st, xs, r, far, tol, rng):
    """The far film is at its round-off floor: max|r| there <= max(tol, FLOOR_FACTOR x floor), floor from 1e-13 relative
    perturbations of the dofs, the nodal correction and K (scaled to eps)."""
    if not far.any():
        return True
    dofs = np.asarray(p.get_current_dofs()[0]).copy()
    bn = [(nd.value(st.bx), nd.value(st.by)) for nd in st.nodes]
    k_saved = st.k_param.value
    st.k_param.value = k_saved * (1 + 1e-13 * rng.standard_normal())
    p.set_current_dofs(dofs * (1 + 1e-13 * rng.standard_normal(len(dofs))))
    xi = 1 + 1e-13 * rng.standard_normal((len(bn), 2))
    for nd, (b1, b2), fct in zip(st.nodes, bn, xi):
        nd.set_value(st.bx, b1 * fct[0])
        nd.set_value(st.by, b2 * fct[1])
    noise = np.abs(np.asarray(p.get_residuals()) - r)
    for nd, (b1, b2) in zip(st.nodes, bn):
        nd.set_value(st.bx, b1)
        nd.set_value(st.by, b2)
    st.k_param.value = k_saved
    p.set_current_dofs(dofs)
    floor = float(noise[far].max()) * EPS / 1e-13
    return float(np.abs(r[far]).max()) <= max(tol, FLOOR_FACTOR * floor)


def newton(p, st, xs, tol, rng, max_iter=12):
    """Newton on the near residual (X < FAR_CHI sqrt(R): rim, plug, pressure, multipliers); the far film only has to
    sit at its round-off floor. Neutral re-splits on a stall; a stall below STALL_CAP with the far film at its floor
    is provisional (the polish loop decides). Returns (ok, near-residual history, info)."""
    far = xs >= FAR_CHI * math.sqrt(max(st.radius(), 1.0))
    near = ~far
    hist, hist_all = [], []
    resplits, since = int(os.environ.get("RESPLITS", "2")), 0
    for k in range(max_iter + 2 * resplits):
        r, jac = p.assemble_jacobian(with_residual=True)
        r = np.asarray(r)
        res_all = float(np.abs(r).max())
        res = float(np.abs(r[near]).max())
        hist.append(res)
        hist_all.append(res_all)
        if not (math.isfinite(res_all) and math.isfinite(res)):
            return False, hist, {"hist_all": hist_all}
        if res_all <= tol:
            return True, hist, None
        since += 1
        stalled = since >= 4 and res > 0.5 * min(hist[-3:-1])
        if res <= tol or stalled:
            if res < STALL_CAP and far_ok(p, st, xs, r, far, tol, rng):
                return True, hist, {"stall": res, "near_res": res, "res_all": res_all, "hist_all": hist_all}
            if stalled and resplits > 0:
                resplits -= 1
                since = 0
                resplit(st)
                continue
            if stalled and k >= 7:
                return False, hist, {"top": top_residuals(p), "hist_all": hist_all}
        jac = sparse.csc_matrix(jac)
        lu = splu(jac)
        d = lu.solve(-r)
        for _ in range(3):
            d = d + lu.solve(-r - jac @ d)
        x = np.asarray(p.get_current_dofs()[0]).copy()
        alpha, improved = 1.0, False
        for _ in range(6):
            p.set_current_dofs(x + alpha * d)
            rn = np.asarray(p.get_residuals())
            if k == 0 or (np.all(np.isfinite(rn)) and float(np.abs(rn[near]).max()) < 2 * res):
                improved = True
                break
            alpha /= 2
        if not improved:  # a round-off direction: keep the iterate; re-split, else provisional if small
            p.set_current_dofs(x)
            if resplits > 0:
                resplits -= 1
                since = 0
                resplit(st)
                continue
            if res < STALL_CAP and far_ok(p, st, xs, r, far, tol, rng):
                return True, hist, {"stall": res, "near_res": res, "backtrack_failed": True, "hist_all": hist_all}
            return False, hist, {"top": top_residuals(p), "backtrack_failed": True, "hist_all": hist_all}
    r = np.asarray(p.get_residuals())
    res = float(np.abs(r[near]).max())
    hist.append(res)
    if res < STALL_CAP and far_ok(p, st, xs, r, far, tol, rng):
        return True, hist, {"stall": res, "near_res": res, "max_iter": True, "hist_all": hist_all}
    return False, hist, {"top": top_residuals(p), "hist_all": hist_all}


def solve(p, st, xs, tol, rng, max_iter=12, passes=12):
    """Newton with the corner tangent lagged: re-solve until tau changes by less than TAU_TOL."""
    tau_r, tau_z = p.get_global_parameter("inner_tau_r"), p.get_global_parameter("inner_tau_z")
    history, info, changes = [], None, []
    for _ in range(passes):
        ok, hist, info = newton(p, st, xs, tol, rng, max_iter)
        history += hist
        if not ok:
            return False, history, {"newton": info, "tau_changes": changes}
        t = st.corner_tangent()
        change = float(np.hypot(t[0] - tau_r.value, t[1] - tau_z.value))
        changes.append(change)
        tau_r.value, tau_z.value = float(t[0]), float(t[1])
        if change < TAU_TOL:
            ok, hist, info = newton(p, st, xs, tol, rng, max_iter)  # converge with the final tau
            history += hist
            return ok, history, ({"floor": info, "tau_changes": changes} if info else None)
    return False, history, {"tau_not_converged": changes}


TAU_TOL = float(os.environ.get("TAU_TOL", "1e-10"))


def j_by_band(fm):
    centres = fm.Xe.mean(axis=1)
    out = {}
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 13), (13, 40)):
        worst = 0.0
        for e in np.where((centres[:, 0] >= lo) & (centres[:, 0] < hi))[0]:
            for s, t in PTS:
                N, Ns, Nt = fm._shape(e, s, t)
                dX = np.array([Ns @ fm.Xe[e], Nt @ fm.Xe[e]]).T
                dy = np.array([Ns @ fm.ye[e], Nt @ fm.ye[e]]).T
                X, y = N @ fm.Xe[e], N @ fm.ye[e]
                worst = max(worst, abs(np.linalg.det(dy @ np.linalg.inv(dX)) * y[0] / X[0] - 1.0))
        out[f"{lo:g}-{hi:g}"] = worst
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    xm, targets = float(sys.argv[2]), sorted((float(x) for x in sys.argv[3:]), reverse=True)
    settings = SegmentSettings.from_case(json.loads(CASE.read_text()))
    R0_REF[0] = settings.r0
    if os.environ.get("NEWTON_TOL"):
        settings = dataclasses.replace(settings, newton_tolerance=float(os.environ["NEWTON_TOL"]))
    grids = dict(np.load(Path(os.environ.get("GRIDS", str(SEG / "grids.npz")))))
    sheet = grids["sheet"]
    last = int(np.searchsorted(sheet[:, 0, 0], xm))
    last -= last % 2
    grids["sheet"] = sheet[: last + 1].copy()
    i_c = int(os.environ.get("INNER", "140"))
    if i_c % 2:
        raise SystemExit("INNER must be an even column index")
    INNER["I_c"] = removed_hoop_moment(grids, i_c)
    col = grids["sheet"][i_c]
    INNER["S_col"] = float(np.sum(np.hypot(np.diff(col[:, 0]), np.diff(col[:, 1]))))
    INNER["sigma"] = settings.surface_tension
    _start = np.load(Path(os.environ.get("START", str(SEG))) / "state.npz")
    if os.environ.get("C_NOSE"):
        INNER["c_nose"] = float(os.environ["C_NOSE"])
    else:
        fm0 = FieldMap(_start["Xe"], _start["ue"], _start["ij"])
        w = np.zeros(len(col))  # quadratic (Simpson) weights along the column, by reference length
        for j in range(0, len(col) - 1, 2):
            seg = float(np.sum(np.hypot(*np.diff(col[j:j + 3], axis=0).T)))
            w[j:j + 3] += seg * np.array([1, 4, 1]) / 6
        rcol = np.array([q[0] + fm0.forward(np.asarray(q, float))[0] for q in col])
        INNER["rbar_start"] = float(w @ rcol / w.sum())
        if os.environ.get("F_NOSE"):
            INNER["F_true"] = float(os.environ["F_NOSE"])
            INNER["c_nose"] = INNER["F_true"] / (2 * math.pi) - math.exp(float(_start["log_ec"])) * INNER["I_c"] * INNER["rbar_start"]
        else:
            INNER["c_nose"] = 0.0
    INNER["X_c_mid"], INNER["X_c_top"], INNER["index"] = float(col[0, 0]), float(col[-1, 0]), i_c
    grids = {"sheet": grids["sheet"][i_c:].copy()}
    x_m = float(grids["sheet"][-1, 0, 0])
    start_folder = Path(os.environ.get("START", str(SEG)))
    saved = np.load(start_folder / "state.npz")
    fmap = FieldMap(saved["Xe"], saved["ue"], saved["ij"])
    r0_hole = float(np.min(saved["Xe"][..., 0] + saved["ue"][..., 0]))
    beta0 = float(saved["cap_beta"])
    p = make_problem(settings, grids, out / "pyoomph", float(saved["log_ec"]), beta0, r0_hole ** 2 - settings.r0 ** 2)
    st = State(p)
    x_s = float(np.max(saved["Xe"][..., 0]))
    k0 = r0_hole ** 2 - settings.r0 ** 2
    u0 = np.zeros((len(st.x), 2))
    for n, (x, z) in enumerate(zip(st.x, st.z)):
        if x <= x_s - 1e-9:
            u0[n] = fmap.forward(np.array([x, z]))
        else:  # beyond the start state's extent: decay the deviation from the area-preserving map
            edge = fmap.forward(np.array([x_s - 1e-9, z]))
            u0[n, 0] = analytic_map(x, k0) + (edge[0] - analytic_map(x_s, k0)) * x_s / x
            u0[n, 1] = edge[1] * (x_s / x) ** 2
    record_extension = {"X_start_extent": x_s, "nodes_extrapolated": int(np.sum(st.x > x_s - 1e-9))}
    st.load(u0[:, 0], u0[:, 1], r0_hole ** 2 - settings.r0 ** 2)
    xs, rng = equation_positions(p), np.random.default_rng(20261026)
    record = {"extension": record_extension, "grids": os.environ.get("GRIDS", str(SEG / "grids.npz")), "X_m": x_m, "inner": dict(INNER), "ndof": p.ndof(), "beta0": beta0, "R0": r0_hole, "start": str(start_folder),
              "points": [], "steps": []}
    cpar = p.get_global_parameter("nose_c")
    cpar.value = INNER["c_nose"]
    r_target = float(st.x[st.inner_mid] + u0[st.inner_mid, 0])
    saved_start = np.asarray(p.get_current_dofs()[0]).copy()
    record["calibration"] = []
    cal_state_ok, cal_hist = False, None
    if not os.environ.get("C_NOSE"):
        tries = []
        for it in range(8):
            p.set_current_dofs(saved_start)
            st.load(u0[:, 0], u0[:, 1], r0_hole ** 2 - settings.r0 ** 2)
            ok, hist, info = newton(p, st, xs, settings.newton_tolerance, rng, 20)
            dr = st.radius() - r_target
            tries.append((cpar.value, dr))
            record["calibration"].append({"c": cpar.value, "dr": dr, "ok": ok, "newton": len(hist)})
            print("calibration", it, "c %.12f dr %+.3e" % (cpar.value, dr), flush=True)
            cal_tol = max(float(os.environ.get("CAL_TOL", "1e-7")), float(os.environ.get("CAL_TOL_REL", "3e-14")) * r_target)
            if not ok or abs(dr) < cal_tol:
                cal_state_ok, cal_hist = ok and abs(dr) < cal_tol, hist
                break
            if len(tries) == 1:
                cpar.value = tries[-1][0] + 1e-3
            else:
                (c0, d0), (c1, d1) = tries[-2], tries[-1]
                cpar.value = c1 - d1 * (c1 - c0) / (d1 - d0)
        INNER["c_nose"] = cpar.value
        record["inner"]["c_nose"] = cpar.value
    if cal_state_ok:  # the last calibration solve is the converged start
        ok, record["start_history"], record["start_info"] = True, cal_hist, {"kept_calibration_state": True}
    else:
        p.set_current_dofs(saved_start)
        st.load(u0[:, 0], u0[:, 1], r0_hole ** 2 - settings.r0 ** 2)
        ok, record["start_history"], record["start_info"] = newton(p, st, xs, settings.newton_tolerance, rng, 20)
        if not ok:
            tx0, ty0 = st.totals()
            st.load(tx0, ty0, st.radius(tx0) ** 2 - settings.r0 ** 2)
            ok, hist2, info2 = newton(p, st, xs, settings.newton_tolerance, rng, 20)
            record["start_history"] = record["start_history"] + hist2
            record["start_info"] = {"first": record["start_info"], "second": info2}
    if ok and record["start_info"]:  # provisional start: the polish loop decides
        ok, record["start_polishes"], _h = polish(p, st, xs, settings.newton_tolerance, rng, settings.r0)
    record["start_dr_vs_start_state"] = st.radius() - r_target
    if not ok:
        (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
        raise RuntimeError("start state did not converge")
    param, beta = p.get_global_parameter("log_Ec"), p.get_global_parameter("cap_beta")
    t0 = time.time()
    ux, uy = st.totals()
    radius = st.radius(ux)
    record["start_R"] = radius
    prior = None  # (ux, uy, R, step) of the state before the current one
    for target in targets:
        while abs(param.value - math.log(target)) > 1e-13:
            beta.value = beta0 * (r0_hole / radius) ** 2
            n_steps = max(1, math.ceil(abs(math.log(target) - param.value) / STEP))
            start = param.value
            step = (math.log(target) - start) / n_steps
            saved_dofs = np.asarray(p.get_current_dofs()[0]).copy()
            for halving in range(int(os.environ.get("HALVINGS", "7"))):
                p.set_current_dofs(saved_dofs)
                q = step / prior[3] if prior else 0.0
                r_pred = radius + q * (radius - prior[2]) if prior else radius
                st.load(ux + q * (ux - prior[0]) if prior else ux, uy + q * (uy - prior[1]) if prior else uy,
                        r_pred ** 2 - settings.r0 ** 2)
                param.value = float(start + step)
                ok, history, info = newton(p, st, xs, settings.newton_tolerance, rng)
                if ok and info:  # provisional: re-split and re-solve until two successive states agree
                    first = info
                    ok, polishes, history2 = polish(p, st, xs, settings.newton_tolerance, rng, settings.r0)
                    history = history + history2
                    info = {"floor_first": first, "polishes": polishes}
                if ok:
                    break
                param.value = start
                step /= 2
                record.setdefault("retries", []).append({"Ec": math.exp(start), "halving": halving,
                                                         "history": history[:14], "info": info})
            else:
                record["failure"] = record["retries"][-1]
                p.set_current_dofs(saved_dofs)
                st.load(ux, uy, radius ** 2 - settings.r0 ** 2)
                fm_last = st.field_map()
                (out / "last-converged").mkdir()
                np.savez(out / "last-converged" / "state.npz", Xe=fm_last.Xe, ue=fm_last.ue, ij=fm_last.ij,
                         log_ec=start, cap_beta=beta.value)
                np.savez(out / "last-converged" / "grids.npz", **grids)
                (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
                print("FAILED", json.dumps(record["failure"])[:1500], flush=True)
                return
            v_bands, bn_bands = st.v_bands()
            step_rec = {"Ec": math.exp(param.value), "beta": beta.value, "K": st.k_param.value,
                        "newton": len(history), "start_residual": history[0], "max_residual": history[-1],
                        "residual_by_X": residual_bands(np.asarray(p.get_residuals()), xs),
                        "v_by_band": v_bands, "bn_by_band": bn_bands}
            if info:
                step_rec["floor_accepted"] = info
            prior = (ux, uy, radius, param.value - start)
            ux, uy = st.totals()
            radius = st.radius(ux)
            step_rec["R"] = radius
            record["steps"].append(step_rec)
        fm = st.field_map()
        point = {"Ec": target, "R": radius, "H": p.rim_height(), "beta": beta.value, "J_by_band": j_by_band(fm),
                 "plug_shift_last": float(p.get_ode("globals").get_value("plug_shift", as_float=True)),
                 "residual_by_X": record["steps"][-1]["residual_by_X"] if record["steps"] else None,
                 "floor_accepted_steps": sum("floor_accepted" in s for s in record["steps"]),
                 "seconds": time.time() - t0}
        ref = FULL.get(round(target, 4))
        if ref:
            point["dR_rel_vs_full"] = radius / ref[0] - 1
            point["dH_vs_full"] = point["H"] - ref[1]
        record["points"].append(point)
        folder = out / f"Ec{target:g}"
        folder.mkdir()
        np.savez(folder / "state.npz", Xe=fm.Xe, ue=fm.ue, ij=fm.ij, log_ec=param.value, cap_beta=beta.value)
        np.savez(folder / "grids.npz", **grids)
        (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({k: v for k, v in point.items() if k not in ("residual_by_X", "J_by_band")}),
              json.dumps({b: round(v, 4) for b, v in point["J_by_band"].items()}),
              json.dumps({b: float(f"{v:.2g}") for b, v in (point["residual_by_X"] or {}).items()}), flush=True)
    (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
