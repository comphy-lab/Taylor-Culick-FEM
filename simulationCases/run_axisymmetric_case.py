"""Run a fresh finite-De Oldroyd-B or zero-modulus Newtonian case.

This entry point supports neither non-relaxing nor nonzero-modulus De=0
sweep endpoints; it is scoped to the two indicated constitutive branches.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name: str):
    # The package __init__ imports pyoomph; --check must remain solver-free.
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/taylor_culick_fem" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_case = _load_helper("cases").validate_case
provenance = _load_helper("provenance")

CASE_FIELDS_USED = [
    "schema", "case_id", "geometry", "nondimensionalization", "physics.Oh",
    "physics.elastic_modulus", "physics.relaxation_time", "physics.constitutive_model",
    "physics.stabilization", "physics.supg_factor", "physics.passive_conformation",
    "physics.far_field", "mesh.length", "mesh.r0", "mesh.resolution", "mesh.far_resolution",
    "tolerances.temporal_error", "tolerances.max_volume_error", "tolerances.max_tip_pin_residual",
    "tolerances.min_radius_increase", "tolerances.min_polymer_stress", "tolerances.max_positive_sponge_power",
    "output.cadence", "output.field_stride", "output.checkpoint_every", "output.write_states",
    "solver.max_step", "solver.max_newton", "arrest.opening_ustar", "arrest.arrest_ustar",
    "arrest.consecutive_samples", "arrest.trend_tolerance", "arrest.radius_span_tolerance",
    "completion.kind", "completion.value", "completion.allow_recoil",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def append_progress(out: Path, event: str, **values) -> None:
    path = out / "progress.jsonl"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "pyoomph-runner-progress-v1", "recorded_at": datetime.now(timezone.utc).isoformat(), "event": event, **values}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--linear-solver", choices=("pardiso", "superlu"))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    case_path = args.case.resolve()
    try:
        case_bytes = case_path.read_bytes()
        case = json.loads(case_bytes)
        validate_case(case)
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        raise SystemExit(f"refused: invalid case: {exc}") from exc
    if args.out.is_symlink() or (args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir()))):
        raise SystemExit("refused: output folder must be new or empty")
    case_hash = hashlib.sha256(case_bytes).hexdigest()
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_hash}))
        return 0
    try:
        import pyoomph
        from pyoomph.solvers.generic import get_default_linear_solver, set_default_linear_solver
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "campaign"))
        from run_viscoelastic_case import case_to_driver_args
        from run_viscoelastic_axi import run_case
    except ImportError as exc:
        raise SystemExit(f"refused: solver import failed: {exc}") from exc
    driver_args = case_to_driver_args(case_path, args.out)
    if driver_args.serialized_case != case_bytes:
        raise SystemExit("refused: case changed after validation")
    commit = provenance.component_commit(ROOT)
    revision = provenance.pyoomph_revision()
    if commit == "unknown" or commit.endswith("-dirty"):
        raise SystemExit(f"refused: source commit is unavailable or dirty: {commit}")
    if not revision.get("commit"):
        raise SystemExit("refused: pyoomph revision is unavailable")
    if args.linear_solver is not None:
        set_default_linear_solver(args.linear_solver)
    effective_solver = get_default_linear_solver()
    if effective_solver is None:
        raise SystemExit("refused: pyoomph has no default linear solver")
    effective_solver = str(effective_solver)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "pyoomph-runner-manifest-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(case_path), "case_sha256": case_hash, "case_id": case["case_id"],
        "case_fields_used": CASE_FIELDS_USED,
        "units": "nondimensional, with scales declared in the case's nondimensionalization",
        "argv": [str(Path(__file__).resolve()), *raw],
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "component_commit": commit, "pyoomph": revision, "pyoomph_module": getattr(pyoomph, "__file__", None),
        "linear_solver": effective_solver,
    }
    try:
        provenance.create_run_manifest(args.out, manifest)
    except FileExistsError as exc:
        raise SystemExit("refused: run manifest already exists") from exc
    append_progress(args.out, "started", case_id=case["case_id"])
    driver_args.progress_callback = lambda event, **values: append_progress(args.out, event, **values)
    try:
        # pyoomph parses the process argv during initialise(); our --out is an
        # abbreviation of its --outdir and would redirect the solver output.
        saved_argv = sys.argv
        try:
            sys.argv = [saved_argv[0]]
            summary = run_case(driver_args)
        finally:
            sys.argv = saved_argv
    except BaseException as exc:
        append_progress(args.out, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    reached = summary.get("t_reached")
    passed = (summary.get("status") == "passed" and summary.get("completion_reason") == "fixed_horizon"
              and isinstance(reached, (int, float)) and math.isclose(reached, case["completion"]["value"], rel_tol=1e-9, abs_tol=1e-8))
    append_progress(args.out, "completed" if passed else "failed", status=summary.get("status"), physical_time=reached)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
