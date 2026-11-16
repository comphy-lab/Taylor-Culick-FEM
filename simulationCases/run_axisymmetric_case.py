"""Run a finite-De Oldroyd-B or zero-modulus Newtonian case.

This entry point supports neither non-relaxing nor nonzero-modulus De=0
sweep endpoints; it is scoped to the two indicated constitutive branches.
Either branch may use the rest far edge or the thin-film membrane exterior.
Bounded segments retain the same case and require a completed parent checkpoint;
a membrane parent must also supply its hash-bound exterior state and history.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# This compatibility edge covers the original fresh-run entry point only.
# Later restarts must use the normal same-case, same-runner contract.
MIGRATION_PARENT_COMMIT = "d3d3a48514e976faf19dbd3468fdc21f0a939218"
MIGRATION_PARENT_RUNNER = "c83d60ab7a19c886a9b68dfbfaddb0859021f3a3a0ace9d23073d8bdde2fb41d"
MIGRATION_PYOOMPH_COMMIT = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
# SHA-256 over sorted path NUL + binary SHA-256 for top-level Python files in
# the four source directories (excluding this runner), plus .python-version,
# pyproject.toml and uv.lock, at commit 57a7297. Extra matching files change
# the digest; these are filesystem files, not a query for tracked Git files.
# The driver changes since the pinned parent concern restart evidence and its
# first post-restart interval; the constitutive implementation is unchanged.
MIGRATION_SOURCE_DIGEST = "cb89d54e716421c5b1bdea2e5baeb29e7f1a3035984c69e235e95ae88900d745"


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


def migration_source_digest(root: Path) -> str:
    """Bind the non-runner Python source and environment inputs to the baseline."""
    paths = [p for pattern in ("campaign/*.py", "simulationCases/*.py",
                                   "src/taylor_culick_fem/*.py", "validation/*.py")
                   for p in root.glob(pattern)
                   if p.relative_to(root).as_posix() != "simulationCases/run_axisymmetric_case.py"]
    paths = sorted(paths + [root / name for name in (".python-version", "pyproject.toml", "uv.lock")])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def endpoint_reached(reached: object, case: dict, target: float | None = None) -> bool:
    """Check pyoomph's bounded final-step landing against the requested horizon."""
    if isinstance(reached, bool) or not isinstance(reached, (int, float)) or not math.isfinite(reached):
        return False
    target = case["completion"]["value"] if target is None else target
    step = min(case["solver"]["max_step"], case["output"]["cadence"])
    # Problem.run clamps its final step to 1.00001 times the remaining
    # interval and skips an undershoot smaller than 1e-8 of the current step.
    rounding = 8 * max(math.ulp(target), math.ulp(reached))
    return target - 1e-8 * step - rounding <= reached <= target + 1.1e-5 * step + rounding


def append_progress(out: Path, event: str, **values) -> None:
    path = out / "progress.jsonl"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "pyoomph-runner-progress-v1", "recorded_at": datetime.now(timezone.utc).isoformat(), "event": event, **values}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def segment_target(case: dict, requested: float | None) -> float:
    """Admit intermediate endpoints only on a scheduled checkpoint boundary."""
    target = case["completion"]["value"]
    end = target if requested is None else requested
    if not math.isfinite(end) or not 0 < end <= target:
        raise ValueError("segment end must be positive and no later than the case horizon")
    if end < target:
        checkpoint = case["output"]["checkpoint_every"]
        if not checkpoint:
            raise ValueError("an intermediate segment requires scheduled checkpoints")
        interval = case["output"]["cadence"] * checkpoint
        multiple = round(end / interval)
        if multiple < 1 or not math.isclose(end, multiple * interval, rel_tol=0, abs_tol=1e-9):
            raise ValueError("segment end must land on a scheduled checkpoint")
    return end


def endpoint_checkpoint(parent_case: Path, case: dict, case_hash: str, parent_end: float) -> dict:
    """Check the completed segment's checkpoint, boundary and restart inputs."""
    metadata_path = parent_case / "case.json"
    metadata = json.loads(metadata_path.read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError("restart parent metadata must be an object")
    if metadata.get("status") != "passed" or metadata.get("completion_reason") != "fixed_horizon":
        raise ValueError("restart parent did not complete successfully")
    if (case["physics"]["constitutive_model"] == "Oldroyd-B"
            and metadata.get("far_boundary_remained_sampled_inflow") is not True):
        raise ValueError("restart parent did not retain sampled far-boundary inflow")
    if not endpoint_reached(metadata.get("t_reached"), case, parent_end):
        raise ValueError("restart parent did not reach its segment endpoint")
    checkpoint = metadata.get("latest_checkpoint")
    if (not isinstance(checkpoint, dict) or checkpoint.get("boundary_gate_passed") is not True
            or not endpoint_reached(checkpoint.get("time"), case, parent_end)):
        raise ValueError("restart parent has no passing endpoint checkpoint")
    control_index, field_index = checkpoint.get("control_index"), checkpoint.get("field_index")
    if (type(control_index) is not int or control_index <= 0 or type(field_index) is not int
            or field_index != control_index // case["output"]["field_stride"]
            or not case["output"]["checkpoint_every"]
            or control_index % case["output"]["checkpoint_every"]
            or not math.isclose(control_index * case["output"]["cadence"], parent_end,
                                rel_tol=0, abs_tol=1e-9)):
        raise ValueError("restart checkpoint indices do not match the segment endpoint")
    reference = checkpoint.get("checkpoint")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("restart parent checkpoint reference is invalid")
    state = (parent_case / reference["path"]).resolve(strict=True)
    if (state.parent != (parent_case / "checkpoints").resolve()
            or not state.is_file() or state.stat().st_size == 0
            or reference.get("sha256") != sha256(state)):
        raise ValueError("restart checkpoint does not match the parent receipt")
    paths = {
        "checkpoint": state,
        "input_case": parent_case / "input-case.json",
        "metadata": metadata_path,
        "control": parent_case / "run_viscoelastic_axi/control.txt",
        "rim_height": parent_case / "run_viscoelastic_axi/rim-height.txt",
        "pvd": parent_case / "run_viscoelastic_axi/liquid-history.pvd",
    }
    if case["physics"].get("far_field", {"mode": "rest"}).get("mode") == "membrane":
        # The exterior is part of the state: restart needs the sidecar saved
        # with this exact checkpoint and the far-field history it continues.
        far = checkpoint.get("far_field_state")
        if not isinstance(far, dict) or not isinstance(far.get("path"), str):
            raise ValueError("restart parent membrane checkpoint has no far-field state")
        far_state = (parent_case / far["path"]).resolve(strict=True)
        if (far_state != state.with_suffix(".farfield.npz") or not far_state.is_file()
                or far.get("sha256") != sha256(far_state)):
            raise ValueError("restart membrane state does not match the parent receipt")
        paths["far_field_state"] = far_state
        paths["far_field_history"] = parent_case / "run_viscoelastic_axi/far-field.txt"
    if sha256(paths["input_case"]) != case_hash:
        raise ValueError("restart parent input case differs from its manifest")
    if not paths["pvd"].is_file():
        paths["pvd"] = paths["pvd"].with_name("liquid.pvd")
    refs = {key: {"path": str(path.resolve(strict=True)), "sha256": sha256(path)}
            for key, path in paths.items()}
    return {**refs, "time": checkpoint["time"],
            "control_index": control_index, "field_index": field_index}


def restart_lineage(state: Path, out: Path, case: dict, case_hash: str, end: float,
                    migrate_horizon: bool = False) -> dict:
    """Bind a completed parent checkpoint and its histories to the same case."""
    state = state.resolve(strict=True)
    if not state.is_file() or state.parent.name != "checkpoints":
        raise ValueError("restart state must be a parent checkpoint file")
    parent_case = state.parent.parent
    parent_run = parent_case.parent
    if out.resolve().is_relative_to(parent_run):
        raise ValueError("restart output must not be inside the parent run")
    manifest_path = parent_run / "run-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("restart parent manifest must be an object")
    if (manifest.get("schema") != "pyoomph-runner-manifest-v1"
            or manifest.get("case_id") != case["case_id"]):
        raise ValueError("restart parent manifest does not bind the same case")
    runner_info, pyoomph_info = manifest.get("runner"), manifest.get("pyoomph")
    argv, fields = manifest.get("argv"), manifest.get("case_fields_used")
    if (not isinstance(runner_info, dict) or not isinstance(pyoomph_info, dict)
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("component_commit")))
            or not re.fullmatch(r"[0-9a-f]{40}", str(pyoomph_info.get("commit")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(runner_info.get("sha256")))
            or not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv)
            or not isinstance(fields, list) or not fields or not all(isinstance(x, str) and x for x in fields)
            or len(set(fields)) != len(fields)):
        raise ValueError("restart parent manifest is incomplete")
    parent_hash = manifest.get("case_sha256")
    migration = None
    if migrate_horizon:
        parent_path = parent_case / "input-case.json"
        parent_bytes = parent_path.read_bytes()
        parent_case_data = json.loads(parent_bytes)
        validate_case(parent_case_data)
        if (hashlib.sha256(parent_bytes).hexdigest() != parent_hash
                or manifest.get("component_commit") != MIGRATION_PARENT_COMMIT
                or runner_info.get("sha256") != MIGRATION_PARENT_RUNNER
                or pyoomph_info.get("commit") != MIGRATION_PYOOMPH_COMMIT
                or "segment_end" in manifest):
            raise ValueError("migration requires the pinned completed fresh-run parent")
        parent_completion = parent_case_data.get("completion")
        new_completion = case.get("completion")
        if (not isinstance(parent_completion, dict) or not isinstance(new_completion, dict)
                or {**parent_case_data, "completion": new_completion} != case
                or {**parent_completion, "value": new_completion.get("value")} != new_completion
                or not parent_completion["value"] < new_completion["value"]):
            raise ValueError("migration may increase only the original case horizon")
        parent_end = parent_completion["value"]
        migration = {"kind": "fresh-run-horizon-extension-v1",
                     "parent_case_sha256": parent_hash,
                     "parent_horizon": parent_end,
                     "parent_component_commit": manifest["component_commit"],
                     "parent_runner_sha256": runner_info["sha256"],
                     "continuation_source_digest": MIGRATION_SOURCE_DIGEST}
    else:
        if parent_hash != case_hash:
            raise ValueError("restart parent manifest does not bind the same case")
        parent_end = manifest.get("segment_end", case["completion"]["value"])
    if isinstance(parent_end, bool) or not isinstance(parent_end, (int, float)) or not math.isfinite(parent_end):
        raise ValueError("invalid parent segment endpoint")
    if not 0 < parent_end < end:
        raise ValueError("parent segment endpoint must precede this segment")
    evidence = endpoint_checkpoint(parent_case, case, parent_hash, parent_end)
    if Path(evidence["checkpoint"]["path"]) != state:
        raise ValueError("restart checkpoint does not match the parent receipt")
    progress = [json.loads(line) for line in (parent_run / "progress.jsonl").read_text().splitlines()]
    if (len(progress) < 2 or not all(isinstance(item, dict) for item in progress)
            or progress[0].get("event") != "started" or progress[0].get("case_id") != case["case_id"]
            or progress[-1].get("event") not in ("segment_completed", "completed")
            or progress[-1].get("status") != "passed"
            or not endpoint_reached(progress[-1].get("physical_time"), case, parent_end)):
        raise ValueError("restart parent has no matching completion event")
    return {"evidence": evidence,
            "parent_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "parent_commit": manifest.get("component_commit"),
            "parent_pyoomph": (manifest.get("pyoomph") or {}).get("commit"),
            "parent_runner_sha256": (manifest.get("runner") or {}).get("sha256"),
            "parent_linear_solver": manifest.get("linear_solver"), "migration": migration}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--linear-solver", choices=("pardiso", "superlu"))
    parser.add_argument("--segment-end", type=float, help="bounded endpoint no later than the case horizon")
    parser.add_argument("--restart-from", type=Path, help="completed parent checkpoint")
    parser.add_argument("--migrate-horizon", action="store_true",
                        help="migrate a pinned completed fresh-run parent to a longer case horizon")
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
        end = segment_target(case, args.segment_end)
        case_hash = hashlib.sha256(case_bytes).hexdigest()
        if args.migrate_horizon and not args.restart_from:
            raise ValueError("horizon migration requires a parent checkpoint")
        lineage = restart_lineage(args.restart_from, args.out, case, case_hash, end,
                                  migrate_horizon=args.migrate_horizon) if args.restart_from else None
        if lineage is not None and lineage["migration"] is not None and migration_source_digest(ROOT) != MIGRATION_SOURCE_DIGEST:
            raise ValueError("migration continuation source differs from reviewed baseline")
    except (OSError, ValueError, TypeError, ArithmeticError, KeyError) as exc:
        raise SystemExit(f"refused: invalid case: {exc}") from exc
    if args.out.is_symlink() or (args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir()))):
        raise SystemExit("refused: output folder must be new or empty")
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
    driver_args.t_end = end
    if lineage is not None:
        from run_viscoelastic_axi import _read_restart
        # Recheck after imports so a changed source cannot pass the earlier
        # solver-free validation and then be used for the actual launch.
        if lineage["migration"] is not None and migration_source_digest(ROOT) != MIGRATION_SOURCE_DIGEST:
            raise SystemExit("refused: migration continuation source differs from reviewed baseline")
        if (lineage["parent_pyoomph"] != revision["commit"]
                or (lineage["migration"] is None and
                    (lineage["parent_commit"] != commit or
                     lineage["parent_runner_sha256"] != sha256(Path(__file__).resolve())))):
            raise SystemExit("refused: restart parent code, runner or pyoomph differs")
        driver_args.restart_case = case_path
        driver_args.restart_evidence = lineage["evidence"]
        try:
            _read_restart(driver_args)
        except (OSError, ValueError, TypeError, ArithmeticError, KeyError) as exc:
            raise SystemExit(f"refused: invalid restart lineage: {exc}") from exc
    if args.linear_solver is not None:
        set_default_linear_solver(args.linear_solver)
    effective_solver = get_default_linear_solver()
    if effective_solver is None:
        raise SystemExit("refused: pyoomph has no default linear solver")
    effective_solver = str(effective_solver)
    if lineage is not None and lineage["parent_linear_solver"] != effective_solver:
        raise SystemExit("refused: restart parent linear solver differs")
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
        "segment_end": end,
    }
    if lineage is not None:
        manifest["restart_from"] = {**lineage["evidence"]["checkpoint"],
                                    "parent_manifest": lineage["parent_manifest"],
                                    "evidence": lineage["evidence"]}
        if lineage["migration"] is not None:
            manifest["restart_from"]["migration"] = lineage["migration"]
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
              and endpoint_reached(reached, case, end))
    # The strong far-edge conformation condition is valid only for sampled
    # inflow, irrespective of whether this case was run in segments.
    inflow_error = None
    if (passed and case["physics"]["constitutive_model"] == "Oldroyd-B"
            and summary.get("far_boundary_remained_sampled_inflow") is not True):
        passed = False
        inflow_error = "far boundary did not remain sampled inflow"
    checkpoint_error = None
    if passed and end < case["completion"]["value"]:
        try:
            from run_viscoelastic_axi import _case_name, _read_restart
            physics = case["physics"]
            name = _case_name(physics["Oh"], physics["elastic_modulus"], physics["relaxation_time"],
                              physics.get("passive_conformation", False))
            case_root = args.out.resolve() / name
            saved = json.loads((case_root / "case.json").read_bytes())
            for key in ("status", "completion_reason", "t_reached", "latest_checkpoint",
                        "far_boundary_remained_sampled_inflow"):
                if saved.get(key) != summary.get(key):
                    raise ValueError(f"saved segment metadata differs: {key}")
            evidence = endpoint_checkpoint(case_root, case, case_hash, end)
            driver_args.restart_case = case_path
            driver_args.restart_evidence = evidence
            driver_args.t_end = case["completion"]["value"]
            _read_restart(driver_args)
        except Exception as exc:
            passed = False
            checkpoint_error = f"intermediate checkpoint is not restart-ready: {exc}"
    event = "completed" if end == case["completion"]["value"] else "segment_completed"
    append_progress(args.out, event if passed else "failed", status=summary.get("status"), physical_time=reached,
                    **({"error": checkpoint_error or inflow_error} if checkpoint_error or inflow_error else {}))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
