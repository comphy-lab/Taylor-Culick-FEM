#!/usr/bin/env python3
"""Screen the restricted equilibrium spectrum on retained grids and total displacement.

    python simulationCases/run_equilibrium_stability.py <case> --grids <grids.npz> \
        --state <state.npz> --out <new folder> --counts 6 12 --shifts 0 0.1

Pressure, absent from the retained state, is reconstructed at fixed displacement.
An explicit stationary-correction option solves the same case on the same grid
in a translated trial space, with strict bounds on geometry and nodal changes.
Failed stationarity stops before screening. Counts and shifts do not establish
spectral completeness. The equations omit azimuthal, swirl and bending modes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

LAUNCH_CPUS = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np

from taylor_culick_fem.cases import EQUILIBRIUM, EQUILIBRIUM_FIELDS_USED, validate_case
from taylor_culick_fem.equilibrium_stability import (
    constrained_curvature_screen,
    eigenpair_residuals, matrix_diagnostics, max_abs, pressure_rows,
    recover_pressure, restore_displacement, retained_nodes, returned_spectrum, spectrum_scope,
    retained_eigen_matrices, stationary_correction,
)
from taylor_culick_fem.run_contract import (
    append_progress, enforce_launcher_affinity, output_problems, read_case, sha256,
    start_run, write_json_atomic,
)

PIN = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--grids", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new or empty output folder")
    parser.add_argument("--counts", type=int, nargs="+", default=[6])
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0])
    parser.add_argument("--ncv", type=int)
    parser.add_argument("--pressure-tolerance", type=float, default=1e-12)
    parser.add_argument("--pressure-method", choices=("augmented", "lsqr"), default="augmented",
                        help="fixed-displacement unweighted least-squares driver")
    parser.add_argument("--pressure-iterations", type=int, default=10000)
    parser.add_argument("--pressure-refinements", type=int, default=4,
                        help="bounded augmented-system iterative refinement")
    parser.add_argument("--stationary-correction", action="store_true",
                        help="opt in to one same-case/grid solve with fixed R/H/J and nodal gates")
    parser.add_argument("--residual-tolerance", type=float,
                        help="stationarity screen; default case solver.newton_tolerance")
    parser.add_argument("--eigenpair-tolerance", type=float, default=1e-7,
                        help="relative backward residual screen, not a mesh-error bound")
    parser.add_argument("--curvature-weights", type=float, nargs="*", default=None,
                        help="opt in to a sufficient discrete constrained-energy screen; "
                             "flag without values uses weights 10 and 100")
    parser.add_argument("--check", action="store_true", help="validate inputs without importing pyoomph")
    return parser


def determinant_samples(fmap) -> dict:
    """Five fixed reference-element samples; extrema are not a pointwise certificate."""
    samples = ((0, 0), (-.77, -.77), (.77, -.77), (-.77, .77), (.77, .77))
    values, reference_dets = [], []
    for e in range(len(fmap.Xe)):
        for s, t in samples:
            N, Ns, Nt = fmap._shape(e, s, t)
            dX = np.array([Ns @ fmap.Xe[e], Nt @ fmap.Xe[e]]).T
            dy = np.array([Ns @ fmap.ye[e], Nt @ fmap.ye[e]]).T
            X, y = N @ fmap.Xe[e], N @ fmap.ye[e]
            if X[0] <= 0:
                raise ValueError("nonpositive sampled reference radius")
            reference_dets.append(float(np.linalg.det(dX)))
            values.append(float(np.linalg.det(dy @ np.linalg.inv(dX)) * y[0] / X[0]))
    if not np.isfinite(values).all() or not np.isfinite(reference_dets).all():
        raise ValueError("non-finite sampled determinant")
    return {"sampled_J_min": min(values), "sampled_J_max": max(values),
            "sampled_max_abs_J_minus_one": max_abs(np.asarray(values) - 1),
            "minimum_abs_reference_mapping_determinant": min(map(abs, reference_dets)),
            "sample_count": len(values), "samples_per_element": samples,
            "scope": "five samples per Q2 element; does not bound extrema between samples"}


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    case = read_case(args.case)
    validate_case(case, problem=EQUILIBRIUM)
    problems = output_problems(args.out)
    if any(n <= 0 for n in args.counts) or len(set(args.counts)) != len(args.counts):
        problems.append("--counts must be distinct positive integers")
    if not all(math.isfinite(s) for s in args.shifts) or len(set(args.shifts)) != len(args.shifts):
        problems.append("--shifts must be distinct finite real numbers")
    if args.ncv is not None and args.ncv <= max(args.counts) + 1:
        problems.append("--ncv must exceed the largest requested count plus one")
    residual_tolerance = (args.residual_tolerance if args.residual_tolerance is not None
                          else float(case["solver"]["newton_tolerance"]))
    for name, value in [("pressure-tolerance", args.pressure_tolerance),
                        ("residual-tolerance", residual_tolerance),
                        ("eigenpair-tolerance", args.eigenpair_tolerance)]:
        if not math.isfinite(value) or value <= 0:
            problems.append(f"--{name} must be finite and positive")
    if args.pressure_iterations <= 0:
        problems.append("--pressure-iterations must be positive")
    if args.pressure_refinements < 0:
        problems.append("--pressure-refinements must be nonnegative")
    curvature_weights = ([10.0, 100.0] if args.curvature_weights == []
                         else args.curvature_weights)
    if curvature_weights is not None and (
            not all(math.isfinite(w) and w > 0 for w in curvature_weights)
            or len(set(curvature_weights)) != len(curvature_weights)):
        problems.append("--curvature-weights must be distinct finite positive values")
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    with np.load(args.grids, allow_pickle=False) as source:
        grids = {name: source[name].copy() for name in source.files}
    with np.load(args.state, allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    if set(grids) != {"core", "lower", "upper", "sheet"}:
        raise SystemExit("refused: grids must contain exactly the four rounded-sheet blocks")
    if any(g.ndim != 3 or g.shape[-1] != 2 or not np.isfinite(g).all()
           or any((n - 1) % 2 or n < 3 for n in g.shape[:2]) for g in grids.values()):
        raise SystemExit("refused: grids must be finite quadratic block grids")
    retained_nodes(saved["Xe"], saved["ue"], saved["ij"])
    log_ec = float(saved["log_ec"])
    beta = float(saved.get("cap_beta", 0.0))
    if not math.isfinite(log_ec) or log_ec > math.log(sys.float_info.max) or not math.isfinite(beta) or beta < 0:
        raise SystemExit("refused: retained log_ec/cap_beta must be finite, with cap_beta>=0")
    if beta != float(case.get("regularisation", {}).get("cap_beta", 0.0)):
        raise SystemExit("refused: retained cap_beta differs from the supplied case")
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
              for name, path in [("case", args.case), ("grids", args.grids), ("state", args.state)]}
    if args.check:
        print(json.dumps({"check": "passed", "inputs": inputs, "spectrum_scope": spectrum_scope()}))
        return 0
    from taylor_culick_fem.provenance import component_commit
    commit = component_commit(ROOT)
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise SystemExit("refused: native screening requires committed, clean component source")
    for tool in [Path(__file__).resolve(), ROOT / "src/taylor_culick_fem/equilibrium_stability.py"]:
        relative = tool.relative_to(ROOT).as_posix()
        result = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                                capture_output=True, check=False)
        if result.returncode or result.stdout != tool.read_bytes():
            raise SystemExit("refused: screening tool is not identical to its committed source")
    from taylor_culick_fem.provenance import pyoomph_revision
    if pyoomph_revision()["commit"] != PIN:
        raise SystemExit("refused: the installed pyoomph commit does not match the exact API pin")
    import pyoomph
    from pyoomph.generic.jit_cache import set_enabled
    from pyoomph.generic.mpi import get_mpi_nproc
    from taylor_culick_fem.equilibrium_branch import SegmentSettings, build_problem
    from taylor_culick_fem.equilibrium_geometry import FieldMap

    if get_mpi_nproc() != 1:
        raise SystemExit("refused: this retained-state screen requires one MPI process")
    set_enabled(False)
    affinity = enforce_launcher_affinity(LAUNCH_CPUS)
    start_run(args.out, root=ROOT, runner=Path(__file__), argv=raw,
              case_path=args.case.resolve(), case=case, case_sha256=inputs["case"]["sha256"],
              case_fields_used=EQUILIBRIUM_FIELDS_USED, pyoomph_module=pyoomph.__file__)
    write_json_atomic(args.out / "input-receipt.json", {"inputs": inputs, "affinity": affinity,
                      "tool_sha256": sha256(Path(__file__)),
                      "library_sha256": sha256(ROOT / "src/taylor_culick_fem/equilibrium_stability.py")})
    append_progress(args.out, "started")
    report = {"schema": "equilibrium-stability-screen-v1", "spectrum_scope": spectrum_scope(),
              "inputs": inputs, "screens": [], "classification": "inconclusive",
              "model_interpretation": "supplied case at the imported total displacement",
              "originating_model_identity": "not automatically established; compare the originating case receipt",
              "source_commit": commit}
    try:
        settings = SegmentSettings.from_case(case)
        problem = build_problem(settings, grids, args.out / "pyoomph",
                                displacement_background=args.stationary_correction)
        for i in range(problem.ntime_stepper()):
            problem.time_stepper_pt(i).make_steady()
        problem.get_global_parameter("log_Ec").value = log_ec
        if beta > 0:
            problem.get_global_parameter("cap_beta").value = beta
        report["restoration"] = restore_displacement(problem, saved)
        pressure = pressure_rows(problem)
        if max(args.counts) >= problem.ndof() - 1:
            raise ValueError("counts would select pyoomph's dense whole-pencil fallback; reduce counts")
        report["pressure_reconstruction"] = recover_pressure(
            problem, pressure, tolerance=args.pressure_tolerance, iterations=args.pressure_iterations,
            method=args.pressure_method, refinements=args.pressure_refinements)
        reconstructed_map = FieldMap.from_problem(problem)
        original_nodes = retained_nodes(saved["Xe"], saved["ue"], saved["ij"])
        current_nodes = retained_nodes(reconstructed_map.Xe, reconstructed_map.ue, reconstructed_map.ij)
        if any(not np.array_equal(current_nodes[X], u) for X, u in original_nodes.items()):
            raise ValueError("pressure reconstruction changed the retained displacement")
        report["pressure_reconstruction"]["total_displacement_unchanged"] = True
        uncorrected = np.asarray(problem.get_residuals())
        report["uncorrected_stationarity"] = {
            "max_equilibrium_residual": max_abs(uncorrected),
            "stationarity_tolerance": residual_tolerance,
            "passed": bool(max_abs(uncorrected) <= residual_tolerance
                           and report["pressure_reconstruction"]["converged"]),
            "total_displacement_unchanged": True,
        }
        if args.stationary_correction:
            if not report["pressure_reconstruction"]["converged"]:
                raise ValueError("stationary correction requires converged fixed-displacement pressure recovery")
            report["stationary_correction"] = stationary_correction(
                problem, max_iterations=settings.max_newton)
            report["model_interpretation"] = "same supplied case/grid at the bounded corrected discrete equilibrium"
            reconstructed_map = FieldMap.from_problem(problem)
            mesh = problem.get_mesh("solid")
            fields = mesh.get_nodal_field_indices()
            element_field = lambda prefix: np.asarray([
                [[element.node_pt(k).value(fields[prefix + axis]) for axis in ("x", "y")]
                 for k in range(element.nnode())] for element in mesh.elements()])
            dofs, _ = problem.get_current_dofs()
            np.savez(args.out / "corrected-state.npz", Xe=reconstructed_map.Xe,
                     ue=reconstructed_map.ue, ij=reconstructed_map.ij, log_ec=log_ec, cap_beta=beta,
                     background_ue=element_field("displacement_background_"),
                     increment_ue=element_field("displacement_"), global_dofs=np.asarray(dofs),
                     pressure_mask=pressure)
            report["corrected_state"] = {"path": str((args.out / "corrected-state.npz").resolve()),
                                         "sha256": sha256(args.out / "corrected-state.npz"),
                                         "contents": "total/background/increment Q2 coefficients and serial global DOFs"}
        residual, tangent = problem.assemble_jacobian(with_residual=True)
        residual, tangent = np.array(residual, copy=True), tangent.copy()
        mass, eigen_tangent = retained_eigen_matrices(problem)
        report["matrices"] = matrix_diagnostics(tangent, mass, residual, pressure)
        report["matrices"]["storage"] = "independent Python-owned CSR arrays retained across native eigensolver assembly"
        report["matrices"]["eigen_vs_stationary_tangent_max_difference"] = max_abs((eigen_tangent - tangent).data)
        report["geometry"] = {"R": problem.hole_radius(), "H": problem.rim_height(),
                               "Ec": problem.elastocapillary_number,
                               **determinant_samples(reconstructed_map)}
        report["stationarity_tolerance"] = residual_tolerance
        report["stationarity_screen_passed"] = (report["matrices"]["max_equilibrium_residual"] <= residual_tolerance
                                               and report["pressure_reconstruction"]["converged"]
                                               and report.get("stationary_correction", {}).get("geometry_gate_passed", True))
        if report["stationarity_screen_passed"]:
            if curvature_weights is not None:
                report["constrained_curvature"] = constrained_curvature_screen(
                    tangent, pressure, weights=curvature_weights)
                write_json_atomic(args.out / "screen.json", report)
            for count in args.counts:
                for shift in args.shifts:
                    values, vectors = problem.solve_eigenproblem(count, shift=shift, which="LM", ncv=args.ncv)
                    screen = returned_spectrum(values, count=count, shift=shift)
                    screen["eigenpairs"] = eigenpair_residuals(eigen_tangent, mass, values, vectors, pressure)
                    screen["eigenpair_tolerance"] = args.eigenpair_tolerance
                    screen["eigenpair_screen_passed"] = all(
                        p["relative_backward_residual"] <= args.eigenpair_tolerance
                        and p["relative_action_residual"] <= args.eigenpair_tolerance
                        for p in screen["eigenpairs"])
                    screen["ncv_requested"] = args.ncv
                    solver = problem.get_eigen_solver()
                    screen["backend"] = "scipy"
                    screen["driver"] = "eigsh" if getattr(solver, "last_symmetry_decision", False) else "eigs"
                    screen["ncv_configured"] = solver.ncv
                    screen["retry_krylov_dimension"] = "not exposed by the pinned backend"
                    report["screens"].append(screen)
                    write_json_atomic(args.out / "screen.json", report)
            report["leading_returned_eigenvalue"] = max(s["leading_returned_eigenvalue"] for s in report["screens"])
            report["positive_returned_mode_detected"] = report["leading_returned_eigenvalue"] > 0
            report["positive_residual_screened_mode_detected"] = any(
                s["eigenpair_screen_passed"] and s["positive_returned_mode_detected"]
                for s in report["screens"])
            energy_passed = report.get("constrained_curvature", {}).get("numerical_screen_passed", False)
            report["energy_classification"] = (
                "numerical sufficient-curvature screen passed; no rigorous factor-error bound"
                if energy_passed else "restricted constrained-energy minimum unestablished")
            report["classification"] = ("restricted returned-mode screen completed; completeness unestablished"
                                        if all(s["eigenpair_screen_passed"] for s in report["screens"])
                                        else "inconclusive: eigenpair residual screen failed")
            if energy_passed and all(s["eigenpair_screen_passed"] for s in report["screens"]):
                report["classification"] = (
                    "restricted numerical sufficient-curvature screen passed; "
                    "no rigorous factor-error bound; continuum and excluded sectors unestablished")
            if report["positive_residual_screened_mode_detected"]:
                report["classification"] = "positive residual-screened growth mode detected in the restricted discrete sector"
                if energy_passed:
                    report["curvature_spectrum_consistency"] = "conflict requires review; no stability conclusion"
                    report["classification"] = "inconclusive: positive growth mode conflicts with constrained-energy screen"
        else:
            report["classification"] = (
                "inconclusive: bounded stationary correction did not pass geometry or stationarity gates"
                if args.stationary_correction else
                "inconclusive: fixed-displacement pressure reconstruction did not establish stationarity")
        write_json_atomic(args.out / "screen.json", report)
        append_progress(args.out, "completed", classification=report["classification"])
        print(json.dumps({"classification": report["classification"], "output": str(args.out / "screen.json")}))
        completed = (report["screens"] and all(s["eigenpair_screen_passed"] for s in report["screens"])
                     and "curvature_spectrum_consistency" not in report)
        return 0 if completed else 2
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json_atomic(args.out / "screen.json", report)
        append_progress(args.out, "failed", **report["error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
