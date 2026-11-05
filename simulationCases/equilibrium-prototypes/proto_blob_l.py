# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Truncated-domain prototype L: analytic area-preserving map + pinned nodal correction + v.

Domain: the retained grid of a full-domain seed equilibrium segment, truncated at the reference cut X_m. At the cut
the sheet carries the depth-averaged membrane traction of the universal decaying (Lame) branch of the
depth-averaged model, plus the continuation of the free surface; the midplane is a symmetry line. Stress: F, cof F
and J evaluated directly (no small-gradient expansion), with the volume penalty of the seed case. The Mooney-Rivlin
cap stiffening of the seed case is scaled as beta0 (R0/R)^2, with beta0 and R0 taken from the start state, so its
relative strength stays at its start value as the hole opens.

A nodal background of size ~R at the rim puts round-off of order eps R |grad N| into F, which a perturbation of the
dofs does not detect. L therefore writes
    u = b_a(X; K) e_r + b_n + v,   b_a = sqrt(X^2 + K) - X,
with K a global parameter (the analytic map has an exact gradient), b_n the pinned nodal correction
(displacement_background) and v the unknown. Before every step (a secant in log Ec once two converged states exist):
    R_pred = R_n + q (R_n - R_(n-1)),  K = R_pred^2 - r0^2,  U_pred = U_n + q (U_n - U_(n-1)),
    b_n = U_pred - b_a(X; K) (nodes),  v = 0,
with U the total nodal displacement and q the ratio of the step sizes. Newton is an own loop: SuperLU with three
refinement sweeps and backtracking, converged at max|r| <= the case tolerance or, on a stall, accepted by the
band-wise floor test, whose 1e-13 relative perturbation covers the dofs and the nodal correction together.
Log-Ec steps of at most 0.002 with halving.
Usage: proto_blob_l.py <out> <X_m> <target Ec> [...]
env (required): CASE (case file of the seed equilibrium segment), SEED (folder of the retained seed segment, holding
its grids.npz and state.npz).
env (optional): START (saved point folder on the seed grid, default SEED; beta0 and R0 are taken from it).
"""

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
BANDS = [(0, 1.5), (1.5, 5), (5, 100), (100, 1000), (1000, 4000), (4000, 1e9)]
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


class BlobProblem(eq.ElastocapillaryEquilibriumProblem):
    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(eq.ReferenceSheetMesh(self.grids, self.r0, self.triangles, quadratic=True))
        BACKGROUND_K.clear()
        BACKGROUND_K.append(self.define_global_parameter(background_K=0.0))
        PENALTY.clear()
        PENALTY.append(self.define_global_parameter(volume_gamma=self.volume_penalty))
        modulus = exp(self.define_global_parameter(log_Ec=0.0))
        beta = self.define_global_parameter(cap_beta=self.cap_stiffening)
        step = (var("coordinate_x") - self.cap_stiffening_boundary()) / self.cap_stiffening_width
        eqs = DirectNeoHookean(modulus, pressure_space=self.pressure_space, damping=var("coordinate_x") ** -2,
                               stiffening=beta / (1 + exp(step)), volume_penalty=self.volume_penalty,
                               displacement_background=True)
        eqs += eq.DeformedSurfaceEnergy(self.surface_tension, True) @ "interface"
        eqs += DirichletBC(displacement_y=0) @ "symmetry"
        eqs += MembraneCutTraction(modulus) @ "far"
        eqs += eq.SurfaceContinuation(self.surface_tension, True) @ "interface/far"
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
        self.surface = np.array([(round(a, 12), round(b, 12)) in sx for a, b in zip(self.x, self.z)])

    def totals(self):
        k = self.k_param.value
        ux = np.array([nd.value(self.ix) + nd.value(self.bx) for nd in self.nodes]) + analytic_map(self.x, k)
        uy = np.array([nd.value(self.iy) + nd.value(self.by) for nd in self.nodes])
        return ux, uy

    def radius(self, ux=None):
        ux = self.totals()[0] if ux is None else ux
        return float(np.min((self.x + ux)[self.surface]))

    def load(self, ux, uy, k):
        """b_n = U - b_a(X; K), v = 0."""
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
    p.set_current_dofs(dofs * (1 + 1e-13 * rng.standard_normal(len(dofs))))
    xi = 1 + 1e-13 * rng.standard_normal((len(bn), 2))
    for nd, (b1, b2), f in zip(st.nodes, bn, xi):
        nd.set_value(st.bx, b1 * f[0])
        nd.set_value(st.by, b2 * f[1])
    noise = np.abs(np.asarray(p.get_residuals()) - res)
    for nd, (b1, b2) in zip(st.nodes, bn):
        nd.set_value(st.bx, b1)
        nd.set_value(st.by, b2)
    p.set_current_dofs(dofs)
    bands, ok = {}, True
    for lo, hi in BANDS:
        m = (xs >= lo) & (xs < hi)
        if not m.any():
            continue
        r, f = float(np.abs(res[m]).max()), float(noise[m].max()) * EPS / 1e-13
        bands[f"{lo:g}-{hi:g}"] = [r, f]
        ok &= r <= max(tol, f)
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


def newton(p, st, xs, tol, rng, max_iter=12):
    """SuperLU + 3 refinement sweeps, backtracking, floor test on a stall. Returns (ok, history, info)."""
    hist = []
    for k in range(max_iter):
        r, jac = p.assemble_jacobian(with_residual=True)
        r = np.asarray(r)
        res = float(np.abs(r).max())
        hist.append(res)
        if not math.isfinite(res):
            return False, hist, None
        if res <= tol:
            return True, hist, None
        if k >= 3 and res > 0.5 * min(hist[-3:-1]):
            ok, bands = floor_test(p, st, xs, tol, rng)
            if ok and res < 1e-3:
                return True, hist, {"bands": bands}
            if k >= 7:
                return False, hist, {"bands": bands, "top": top_residuals(p)}
        jac = sparse.csc_matrix(jac)
        lu = splu(jac)
        d = lu.solve(-r)
        for _ in range(3):
            d = d + lu.solve(-r - jac @ d)
        x = np.asarray(p.get_current_dofs()[0]).copy()
        alpha = 1.0
        for _ in range(6):
            p.set_current_dofs(x + alpha * d)
            if k == 0 or float(np.abs(np.asarray(p.get_residuals())).max()) < 2 * res:
                break
            alpha /= 2
    r = np.asarray(p.get_residuals())
    hist.append(float(np.abs(r).max()))
    return hist[-1] <= tol, hist, (None if hist[-1] <= tol else {"top": top_residuals(p)})


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
    grids = dict(np.load(SEG / "grids.npz"))
    sheet = grids["sheet"]
    last = int(np.searchsorted(sheet[:, 0, 0], xm))
    last -= last % 2
    grids["sheet"] = sheet[: last + 1].copy()
    x_m = float(grids["sheet"][-1, 0, 0])
    start_folder = Path(os.environ.get("START", str(SEG)))
    saved = np.load(start_folder / "state.npz")
    fmap = FieldMap(saved["Xe"], saved["ue"], saved["ij"])
    r0_hole = float(np.min(saved["Xe"][..., 0] + saved["ue"][..., 0]))
    beta0 = float(saved["cap_beta"])
    p = make_problem(settings, grids, out / "pyoomph", float(saved["log_ec"]), beta0, r0_hole ** 2 - settings.r0 ** 2)
    st = State(p)
    u0 = np.array([fmap.forward(np.array([x, z])) for x, z in zip(st.x, st.z)])
    st.load(u0[:, 0], u0[:, 1], r0_hole ** 2 - settings.r0 ** 2)
    xs, rng = equation_positions(p), np.random.default_rng(20261026)
    record = {"X_m": x_m, "ndof": p.ndof(), "beta0": beta0, "R0": r0_hole, "start": str(start_folder),
              "points": [], "steps": []}
    ok, record["start_history"], record["start_info"] = newton(p, st, xs, settings.newton_tolerance, rng, 20)
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
            n_steps = max(1, math.ceil(abs(math.log(target) - param.value) / 0.002))
            start = param.value
            step = (math.log(target) - start) / n_steps
            saved_dofs = np.asarray(p.get_current_dofs()[0]).copy()
            for halving in range(5):
                p.set_current_dofs(saved_dofs)
                q = step / prior[3] if prior else 0.0
                r_pred = radius + q * (radius - prior[2]) if prior else radius
                st.load(ux + q * (ux - prior[0]) if prior else ux, uy + q * (uy - prior[1]) if prior else uy,
                        r_pred ** 2 - settings.r0 ** 2)
                param.value = float(start + step)
                ok, history, info = newton(p, st, xs, settings.newton_tolerance, rng)
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
