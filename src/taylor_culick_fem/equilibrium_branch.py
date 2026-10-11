"""Arclength continuation of steady equilibria in segments with pull-back remeshing.

A segment builds the problem on a mesh (the reference grids for the first segment; for
later ones, grids designed in the deformed state of the previous segment and pulled back,
see :func:`taylor_culick_fem.equilibrium_geometry.design_grids`), transfers the previous
displacement as the initial guess, solves at the stored ``log Ec`` and then continues by
arclength in ``log Ec`` until it has taken ``segment_steps`` steps, grown ``ln R`` by
``segment_log_radius_growth``, passed the stop value or failed. Each accepted point records
the hole radius, the full rim height, the far-field amplitude, the leading eigenvalues of
the energy flow and the incompressibility error near the edge.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from taylor_culick_fem.equilibrium_geometry import (
    FieldMap,
    design_grids,
    incompressibility_error,
    reference_grids,
)


@dataclass(frozen=True)
class SegmentSettings:
    """Numerical settings of every segment, read from the case."""

    r0: float
    surface_tension: float
    elements_per_quarter: int
    length: float
    growth: float
    remesh_growth: float
    max_spacing: float
    newton_tolerance: float
    max_newton: int
    eigenvalues: int
    initial_step: float
    max_step: float
    segment_steps: int
    segment_log_radius_growth: float
    ec_start: float
    cap_beta: float
    columns: str = "vertical"
    shoulder_angle: float = 10.0
    blend: float = 4.0

    @classmethod
    def from_case(cls, case: dict) -> SegmentSettings:
        physics, mesh = case["physics"], case["mesh"]
        solver, cont = case["solver"], case["continuation"]
        return cls(
            r0=float(physics["r0"]), surface_tension=float(physics["surface_tension"]),
            elements_per_quarter=int(mesh["elements_per_quarter"]), length=float(mesh["length"]),
            growth=float(mesh["growth"]), remesh_growth=float(mesh["remesh_growth"]),
            max_spacing=float(mesh["max_spacing"]),
            newton_tolerance=float(solver["newton_tolerance"]), max_newton=int(solver["max_newton"]),
            eigenvalues=int(solver["eigenvalues"]), initial_step=float(cont["initial_step"]),
            max_step=float(cont["max_step"]), segment_steps=int(cont["segment_steps"]),
            segment_log_radius_growth=float(cont["segment_log_radius_growth"]),
            ec_start=float(cont["Ec_start"]),
            cap_beta=float(case.get("regularisation", {}).get("cap_beta", 0.0)),
            columns=str(mesh.get("columns", "vertical")),
            shoulder_angle=float(mesh.get("shoulder_angle", 10.0)),
            blend=float(mesh.get("blend", 4.0)),
        )


def build_problem(settings: SegmentSettings, grids: dict, folder: Path):
    """An initialised steady problem on ``grids`` with the solver settings of the case."""
    from taylor_culick_fem.elastocapillary_equilibrium import ElastocapillaryEquilibriumProblem

    problem = ElastocapillaryEquilibriumProblem()
    problem.ignore_command_line = True
    problem.set_output_directory(str(folder))
    problem.r0 = settings.r0
    problem.surface_tension = settings.surface_tension
    problem.grids = grids
    problem.quadratic_grids = True
    problem.length = settings.length
    problem.cap_stiffening = settings.cap_beta
    problem.newton_solver_tolerance = settings.newton_tolerance
    problem.set_linear_solver("superlu")
    problem.set_eigensolver("scipy")
    problem.initialise()
    return problem


def _start(settings: SegmentSettings, grids: dict, folder: Path, old: FieldMap | None,
           log_ec: float, beta_in: float):
    """Build the problem on ``grids``, transfer ``old`` as the initial guess and solve."""
    problem = build_problem(settings, grids, folder / "pyoomph")
    parameter = problem.get_global_parameter("log_Ec")
    parameter.value = log_ec
    if settings.cap_beta > 0:
        problem.get_global_parameter("cap_beta").value = beta_in
    if old is not None:
        mesh = problem.get_mesh("solid")
        index = mesh.get_nodal_field_indices()
        ix, iy = index["displacement_x"], index["displacement_y"]
        for node in mesh.nodes():
            u = old.forward(np.array([node.x(0), node.x(1)]))
            node.set_value(ix, float(u[0]))
            node.set_value(iy, float(u[1]) if node.x(1) > 1e-14 else 0.0)
    problem.solve(max_newton_iterations=settings.max_newton)
    if settings.cap_beta > 0 and beta_in != settings.cap_beta:
        beta = problem.get_global_parameter("cap_beta")
        for value in np.geomspace(max(beta_in, settings.cap_beta * 1e-3), settings.cap_beta, 13):
            beta.value = float(value)
            problem.solve(max_newton_iterations=settings.max_newton)
    return problem, parameter


def _point(problem, settings: SegmentSettings, fmap: FieldMap) -> dict:
    eigenvalues, _ = problem.solve_eigenproblem(settings.eigenvalues, shift=0)
    return {
        "Ec": problem.elastocapillary_number, "log_Ec": problem.get_global_parameter("log_Ec").value,
        "R": problem.hole_radius(), "H": problem.rim_height(),
        "A": problem.far_edge_displacement() * settings.length,
        "leading_eigenvalue": float(max(e.real for e in eigenvalues)),
        "eigenvalues": sorted(float(e.real) for e in eigenvalues)[-3:],
        "J_err": incompressibility_error(fmap), "ndof": problem.ndof(),
        "cap_beta": settings.cap_beta,
    }


def solve_from_segment(settings: SegmentSettings, segment: Path, ec: float, folder: Path,
                       log_step: float = 0.02) -> dict:
    """The equilibrium at exactly ``ec``, from a segment's saved state on its own grids.

    The state is reloaded on the grids it was computed on (an exact transfer), then walked to
    ``ec`` in natural-parameter steps of at most ``log_step`` in ``log Ec``.
    """
    folder.mkdir(parents=True, exist_ok=False)
    grids = dict(np.load(segment / "grids.npz"))
    saved = np.load(segment / "state.npz")
    old = FieldMap(saved["Xe"], saved["ue"], saved["ij"])
    beta_in = float(saved["cap_beta"]) if "cap_beta" in saved.files else 0.0
    problem, parameter = _start(settings, grids, folder, old, float(saved["log_ec"]), beta_in)
    target = math.log(ec)
    steps = max(1, math.ceil(abs(target - parameter.value) / log_step))
    for value in np.linspace(parameter.value, target, steps + 1)[1:]:
        parameter.value = float(value)
        problem.solve(max_newton_iterations=settings.max_newton)
    return _point(problem, settings, FieldMap.from_problem(problem))


def run_segment(settings: SegmentSettings, state: Path | None, folder: Path, ec_stop: float) -> dict:
    """Run one segment into ``folder``; return its rows, status and final state path."""
    started = time.time()
    folder.mkdir(parents=True, exist_ok=False)
    if state is None:
        old = None
        grids = reference_grids(settings.r0, settings.length, settings.elements_per_quarter,
                                settings.growth, None, settings.max_spacing, quadratic=True)
        log_ec, step, beta_in = math.log(settings.ec_start), settings.initial_step, 0.0
    else:
        saved = np.load(state)
        old = FieldMap(saved["Xe"], saved["ue"], saved["ij"])
        log_ec, step = float(saved["log_ec"]), float(saved["ds"])
        beta_in = float(saved["cap_beta"]) if "cap_beta" in saved.files else 0.0
        grids = design_grids(old, settings.elements_per_quarter, settings.length,
                             growth=settings.remesh_growth, max_spacing=settings.max_spacing,
                             r0=settings.r0, columns=settings.columns,
                             shoulder_angle=settings.shoulder_angle, blend=settings.blend)
    np.savez(folder / "grids.npz", **grids)
    problem, parameter = _start(settings, grids, folder, old, log_ec, beta_in)

    rows: list[dict] = []

    def record(tag: str) -> FieldMap:
        fmap = FieldMap.from_problem(problem)
        rows.append({**_point(problem, settings, fmap), "tag": tag})
        return fmap

    good = (record("start"), parameter.value, step)
    r_start = rows[-1]["R"]
    status = "segment_steps"
    for _ in range(settings.segment_steps):
        previous = parameter.value
        try:
            step = problem.arclength_continuation("log_Ec", step, max_ds=settings.max_step)
        except RuntimeError as exc:  # pyoomph raises a RuntimeError; keep the last converged state
            status = f"failed: {str(exc)[:120]}"
            break
        good = (record(f"dlog={parameter.value - previous:+.5f}"), parameter.value, step)
        if parameter.value > previous:
            status = "parameter_reversed"
            break
        if math.log(rows[-1]["R"] / r_start) > settings.segment_log_radius_growth:
            status = "radius_growth"
            break
        if rows[-1]["Ec"] < ec_stop:
            status = "ec_stop"
            break
    fmap, log_ec, step = good
    state_path = folder / "state.npz"
    fmap.save(state_path, log_ec=log_ec, ds=step, cap_beta=settings.cap_beta)
    np.savetxt(folder / "surface.txt", problem.deformed_surface(), header="r z R0 Z0")
    record_ = {"rows": rows, "status": status, "seconds": time.time() - started}
    (folder / "rows.json").write_text(json.dumps(record_, indent=1) + "\n", encoding="utf-8")
    return {"rows": rows, "status": status, "state": state_path}
