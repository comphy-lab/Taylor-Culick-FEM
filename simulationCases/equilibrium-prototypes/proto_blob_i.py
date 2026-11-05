# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Truncated-domain prototype I: prototype H with the previous solution as the predictor background.

Domain: the retained grid of a full-domain seed equilibrium segment, truncated at the reference cut X_m. At the cut
the sheet carries the depth-averaged membrane traction of the universal decaying (Lame) branch of the
depth-averaged model, plus the continuation of the free surface; the midplane is a symmetry line. Stress: F, cof F
and J evaluated directly (no small-gradient expansion), with the volume penalty of the seed case. The Mooney-Rivlin
cap stiffening of the seed case is scaled as beta0 (R0/R)^2, with beta0 and R0 taken from the start state, so its
relative strength stays at its start value as the hole opens.

As prototype H, the displacement is split as u = v + b with a pinned nodal background b. Before every step the
background is set to the predictor
    b = U_n + (b_m(X; c_pred) - b_m(X; c_n)) e_r,   v = 0,
with U_n the last converged total nodal displacement (both components), b_m the membrane map of the universal
branch and c_pred its shift extrapolated linearly in log Ec from the last two converged steps. The rim's shape
changes slowly with Ec while its position moves, so v at convergence carries only the change of the rim's shape,
which lowers both the start residual of each step and the round-off floor in the rim. A failed attempt restores
the dofs and rebuilds the predictor for the halved step.

Continuation and acceptance as prototype H: log-Ec steps of at most 0.002 with halving, and band-wise acceptance
at the measured round-off floor.
Usage: proto_blob_i.py <out> <X_m> <target Ec> [...]
env (required): CASE (case file of the seed equilibrium segment), SEED (folder of the retained seed segment, holding
its grids.npz and state.npz).
env (optional): START (saved point folder on the seed grid, default SEED; beta0 and R0 are taken from it),
BETA_MODE (scaled, the default, or fixed cap stiffening).
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
from pyoomph import DirichletBC  # noqa: E402
from pyoomph.expressions import (exp, grad, log, matproduct, matrix, partial_t,  # noqa: E402
                                 testfunction, transpose, var, vector)
from pyoomph.expressions.interpol import InterpolateSpline1d  # noqa: E402
from pyoomph.generic.codegen import InterfaceEquations  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

import taylor_culick_fem.depth_averaged_equilibrium as da  # noqa: E402
import taylor_culick_fem.elastocapillary_equilibrium as eq  # noqa: E402
from taylor_culick_fem.equilibrium_branch import SegmentSettings  # noqa: E402
from taylor_culick_fem.equilibrium_geometry import FieldMap  # noqa: E402

CASE = Path(os.environ["CASE"])
SEG = Path(os.environ["SEED"])
FULL = {0.37: (150.3883804984, 1.2936673), 0.35: (205.58378168509037, 1.3179252140280093),
        0.3: (536.6705989797, 1.3896701224106813), 0.25: (2044.166526, 1.48239662)}
PENALTY = []
BETA_MODE = os.environ.get("BETA_MODE", "scaled")
EPS = 2.220446049250313e-16
BANDS = [(0, 1.5), (1.5, 5), (5, 100), (100, 1000), (1000, 4000), (4000, 1e9)]
PTS = ((0, 0), (-0.77, -0.77), (0.77, -0.77), (-0.77, 0.77), (0.77, 0.77))


def universal_branch():
    """(y, T, s) on the decaying Lame branch; s = ln X - c, normalised as -ln(y)/2 - 7y/16 + o(y)."""
    prof = da.log_profile(0.05)
    sol = prof.solution if hasattr(prof, "solution") else next(v for v in vars(prof).values() if hasattr(v, "sol"))
    seed = float(sol.t[0])
    y_far = np.geomspace(1e-14, seed, 400, endpoint=False)
    y_ode = np.concatenate([np.geomspace(seed, 1e-2, 4000), np.linspace(1e-2, float(prof.log_radius_ratio), 40000)[1:]])
    t_ode = sol.sol(y_ode)[0]
    k_ode = np.array([math.exp(da._log_column_stretch(y, t) - 2 * y) for y, t in zip(y_ode, t_ode)])
    s_ode = -0.5 * math.log(seed) - 7 * seed / 16 + np.concatenate([[0.0], cumulative_trapezoid(1 / (k_ode - 1), y_ode)])
    s_far = -0.5 * np.log(y_far) - 7 * y_far / 16
    t_far = -2 * y_far + y_far ** 2 - 7 * y_far ** 3 / 12
    return (np.concatenate([y_far, y_ode]), np.concatenate([t_far, t_ode]), np.concatenate([s_far, s_ode]))


Y_TAB, T_TAB, S_TAB = universal_branch()
TENSION = InterpolateSpline1d(np.column_stack([Y_TAB[Y_TAB >= 0], T_TAB[Y_TAB >= 0]]), k=5)
S_OF_Y = PchipInterpolator(Y_TAB, S_TAB)
Y_OF_S = PchipInterpolator(S_TAB[::-1], Y_TAB[::-1])


def membrane_background(x, c):
    """Radial displacement of the membrane map with shift c at reference radii x (numpy)."""
    s = np.clip(np.log(x) - c, S_TAB[-1], S_TAB[0])
    return x * np.expm1(Y_OF_S(s))


def fit_shift(x, r, x_lo, x_hi):
    """c = median of ln X - s(ln(r/X)) over x_lo <= X <= x_hi (midplane samples)."""
    m = (x >= x_lo) & (x <= x_hi)
    y = np.clip(np.log(r[m] / x[m]), Y_TAB[0], Y_TAB[-1])
    return float(np.median(np.log(x[m]) - S_OF_Y(y)))


class DirectNeoHookean(eq.NeoHookeanDisplacementEquations):
    """Same weak form as the parent, with F, cof F and J evaluated directly (no small-gradient expansion)."""

    def define_residuals(self):
        u = eq.physical_displacement(self.displacement_background)
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
        if self.displacement_background:
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


def make_problem(settings, grids, folder, log_ec, beta):
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
    return p


class Background:
    """Pinned nodal membrane background and its shift c; rebasing keeps u = v + b fixed on every node."""

    def __init__(self, p, x_m):
        mesh = p.get_mesh("solid")
        idx = mesh.get_nodal_field_indices()
        self.nodes = list(mesh.nodes())
        self.ix, self.iy = idx["displacement_x"], idx["displacement_y"]
        self.bx, self.by = idx["displacement_background_x"], idx["displacement_background_y"]
        self.x = np.array([nd.x(0) for nd in self.nodes])
        self.z = np.array([nd.x(1) for nd in self.nodes])
        self.mid = np.abs(self.z) < 1e-12
        self.x_m = x_m
        self.c = None

    def total_radial(self):
        return np.array([nd.value(self.ix) + nd.value(self.bx) for nd in self.nodes])

    def totals(self):
        return (np.array([nd.value(self.ix) + nd.value(self.bx) for nd in self.nodes]),
                np.array([nd.value(self.iy) + nd.value(self.by) for nd in self.nodes]))

    def v_bands(self):
        v = np.hypot([nd.value(self.ix) for nd in self.nodes], [nd.value(self.iy) for nd in self.nodes])
        return {f"{lo:g}-{hi:g}": float(np.abs(v[(self.x >= lo) & (self.x < hi)]).max())
                for lo, hi in BANDS if np.any((self.x >= lo) & (self.x < hi))}

    def fit(self, ux):
        return fit_shift(self.x[self.mid], self.x[self.mid] + ux[self.mid], 50.0, 0.5 * self.x_m)

    def set_predictor(self, ux, uy, c_now, c_pred):
        """b = U + (b_m(c_pred) - b_m(c_now)) e_r and v = 0 on every node."""
        bx = ux + membrane_background(self.x, c_pred) - membrane_background(self.x, c_now)
        for nd, b1, b2, z in zip(self.nodes, bx, uy, self.z):
            nd.set_value(self.bx, float(b1))
            nd.set_value(self.by, float(b2) if z > 1e-14 else 0.0)
            nd.set_value(self.ix, 0.0)
            nd.set_value(self.iy, 0.0)

    def set_from_physical(self, p, fmap):
        """Load a physical displacement map: fit c on it, then v = u - b."""
        u = np.array([fmap.forward(np.array([x, z])) for x, z in zip(self.x, self.z)])
        self.c = fit_shift(self.x[self.mid], self.x[self.mid] + u[self.mid, 0], 50.0, 0.5 * self.x_m)
        b = membrane_background(self.x, self.c)
        for nd, ux, uz, bx, z in zip(self.nodes, u[:, 0], u[:, 1], b, self.z):
            nd.set_value(self.bx, float(bx))
            nd.set_value(self.by, 0.0)
            nd.set_value(self.ix, float(ux - bx))
            nd.set_value(self.iy, float(uz) if z > 1e-14 else 0.0)

    def rebase(self):
        """Refit c on the current state and move the split between v and b; returns |v_r| max by band."""
        ur = self.total_radial()
        c_new = fit_shift(self.x[self.mid], self.x[self.mid] + ur[self.mid], 50.0, 0.5 * self.x_m)
        b_new = membrane_background(self.x, c_new)
        for nd, total, bx in zip(self.nodes, ur, b_new):
            nd.set_value(self.bx, float(bx))
            nd.set_value(self.ix, float(total - bx))
        self.c = c_new
        v = ur - b_new
        return {f"{lo:g}-{hi:g}": float(np.abs(v[(self.x >= lo) & (self.x < hi)]).max())
                for lo, hi in BANDS if np.any((self.x >= lo) & (self.x < hi))}


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


def residual_bands(p, xs, res=None):
    res = np.asarray(p.get_residuals()) if res is None else res
    return {f"{lo:g}-{hi:g}": float(np.abs(res[(xs >= lo) & (xs < hi)]).max())
            for lo, hi in BANDS if np.any((xs >= lo) & (xs < hi))}


def floor_test(p, xs, tol, rng):
    """Band-wise test of the current iterate against the measured round-off floor."""
    res = np.asarray(p.get_residuals()).copy()
    dofs = np.asarray(p.get_current_dofs()[0]).copy()
    p.set_current_dofs(dofs * (1 + 1e-13 * rng.standard_normal(len(dofs))))
    noise = np.abs(np.asarray(p.get_residuals()) - res)
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
    p = make_problem(settings, grids, out / "pyoomph", float(saved["log_ec"]), beta0)
    bg = Background(p, x_m)
    bg.set_from_physical(p, fmap)
    xs, rng = equation_positions(p), np.random.default_rng(20261025)
    record = {"X_m": x_m, "ndof": p.ndof(), "beta_mode": BETA_MODE, "beta0": beta0, "R0": r0_hole,
              "start": str(start_folder), "c_start": bg.c, "points": [], "steps": []}
    try:
        p.solve(max_newton_iterations=settings.max_newton)
    except RuntimeError:
        ok, record["start_floor_accepted"] = floor_test(p, xs, settings.newton_tolerance, rng)
        if not ok:
            (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
            raise
    record["start_R"] = p.hole_radius()
    record["start_v_by_band"] = bg.v_bands()
    param, beta = p.get_global_parameter("log_Ec"), p.get_global_parameter("cap_beta")
    t0 = time.time()
    radius = p.hole_radius()
    ux, uy = bg.totals()
    c_now = bg.fit(ux)
    previous = None  # (log Ec, c) of the step before
    for target in targets:
        while abs(param.value - math.log(target)) > 1e-13:
            if BETA_MODE == "scaled":
                beta.value = beta0 * (r0_hole / radius) ** 2
            n_steps = max(1, math.ceil(abs(math.log(target) - param.value) / 0.002))
            start = param.value
            step = (math.log(target) - start) / n_steps
            saved_dofs = np.asarray(p.get_current_dofs()[0]).copy()
            slope = (c_now - previous[1]) / (start - previous[0]) if previous else 0.0
            accepted = None
            for halving in range(5):
                p.set_current_dofs(saved_dofs)
                bg.set_predictor(ux, uy, c_now, c_now + slope * step)
                param.value = float(start + step)
                try:
                    p.solve(max_newton_iterations=8 if halving == 0 else settings.max_newton)
                    accepted = None
                    break
                except RuntimeError as error:
                    history = list(p.get_last_residual_convergence())
                    ok, bands = floor_test(p, xs, settings.newton_tolerance, rng)
                    if ok and history and history[-1] < 1e-3:
                        accepted = {"bands": bands, "history": history[-3:]}
                        break
                    param.value = start
                    step /= 2
                    record.setdefault("retries", []).append({"Ec": math.exp(start), "halving": halving,
                                                             "error": str(error)[:120], "history": history[:12],
                                                             "bands_res_floor": bands})
            else:
                record["failure"] = record["retries"][-1]
                (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
                print("FAILED", json.dumps(record["failure"])[:1500], flush=True)
                return
            radius = p.hole_radius()
            step_rec = {"Ec": math.exp(param.value), "R": radius, "beta": beta.value,
                        "newton": len(p.get_last_residual_convergence()),
                        "max_residual": float(np.abs(p.get_residuals()).max())}
            if accepted:
                step_rec["floor_accepted"] = accepted
            step_rec["residual_by_X"] = residual_bands(p, xs)
            step_rec["v_by_band"] = bg.v_bands()  # the predictor error
            step_rec["start_residual"] = float(p.get_last_residual_convergence()[0])
            previous = (start, c_now)
            ux, uy = bg.totals()
            c_now = bg.fit(ux)
            bg.c = c_now
            step_rec["c"] = c_now
            record["steps"].append(step_rec)
        fm = FieldMap.from_problem(p)
        point = {"Ec": target, "R": radius, "H": p.rim_height(), "beta": beta.value, "c": bg.c,
                 "J_by_band": j_by_band(fm), "residual_by_X": record["steps"][-1]["residual_by_X"] if record["steps"] else None,
                 "v_by_band": record["steps"][-1]["v_by_band"] if record["steps"] else None,
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
        print(json.dumps({k: v for k, v in point.items() if k not in ("residual_by_X", "J_by_band", "v_by_band")}),
              json.dumps({b: round(v, 4) for b, v in point["J_by_band"].items()}),
              json.dumps({b: float(f"{v:.2g}") for b, v in point["residual_by_X"].items()}), flush=True)
    (out / "results.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
