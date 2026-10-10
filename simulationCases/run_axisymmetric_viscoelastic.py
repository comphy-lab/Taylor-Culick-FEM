#!/usr/bin/env python3
"""axisymmetric_viscoelastic: retract one hole in an elastic sheet into one new output folder.

The case file supplies the physics and numerics of one run (``simulationCases/README.md``
lists its fields); the run manifest records the case and the options. The default case is
``simulationCases/axisymmetric_viscoelastic.json``.

    python simulationCases/run_axisymmetric_viscoelastic.py <case> --out <folder> [options]

Before it creates anything, the runner refuses an invalid case or option and an output
folder that already holds anything, for a fresh run and a restart alike. A restart
(``--restart-from <parent>/checkpoints/ckpt_NNNNN.dump``) continues a run of the same case
into its own new folder, and may only extend the stop time (``--t-end``) or lengthen the
far-field membrane (``--far-field-extension``); the run list joins it to its parent with
``continues``. ``run-manifest.json`` is written before any other output and is never
replaced. ``--check`` stops after the checks that need neither pyoomph nor the solver.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from taylor_culick_fem.cases import CASE_FIELDS_USED, validate_case
from taylor_culick_fem.lineage import restart_evidence
from taylor_culick_fem.provenance import (
    MANIFEST,
    component_commit,
    create_run_manifest,
    pyoomph_revision,
)

RUNNER_SCHEMA = "pyoomph-runner-manifest-v1"
PROGRESS = "progress.jsonl"
UNITS = ("nondimensional: lengths on the full sheet thickness h0, velocities on "
         "sqrt(sigma/(rho h0)), times on sqrt(rho h0^3/sigma); V_TC = sqrt(2)")
# "passed": the declared stop (time, arrest or height plateau) was reached and every check
# held. "horizon_reached": an event stop did not occur by its horizon, which the case
# declares as the alternative endpoint. "failed" and "error" exit non-zero.
STOP_STATUSES = ("passed", "horizon_reached")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exit_status(summary: dict) -> int:
    """0 only for a run that ended at its declared stop with every check passing."""
    return 0 if summary.get("status") in STOP_STATUSES else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="pyoomph-case-v1 case file")
    parser.add_argument("--out", type=Path, required=True,
                        help="new or empty output folder; never one that already holds a run")
    parser.add_argument("--restart-from", type=Path, default=None,
                        help="checkpoint <parent>/checkpoints/ckpt_NNNNN.dump of an earlier run of "
                             "the same case; continues into --out")
    parser.add_argument("--t-end", type=float, default=None,
                        help="replace the case's stop time (time value or event horizon)")
    parser.add_argument("--far-field-extension", type=float, default=None,
                        help="lengthen the far-field membrane beyond the case's extension")
    parser.add_argument("--check", action="store_true",
                        help="validate the case, options and output folder, then exit without output")
    return parser


def option_problems(args: argparse.Namespace) -> list[str]:
    """Every invalid option; empty when all are valid."""
    problems = []
    for name, value in (("--t-end", args.t_end), ("--far-field-extension", args.far_field_extension)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            problems.append(f"{name} must be finite and positive")
    return problems


def output_problems(out: Path) -> list[str]:
    """A run, fresh or restart, needs a new folder or an empty one."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        return [(f"{out} already holds a run or other files; every run, including a restart, "
                 "needs its own new output folder")]
    return []


def parent_run(state: Path) -> Path | None:
    """The nearest folder holding a run manifest, from the state's folder up two levels."""
    folder = state.resolve().parent
    for candidate in (folder, *list(folder.parents)[:2]):
        if (candidate / MANIFEST).is_file():
            return candidate
    return None


def check_restart(state: Path, out: Path, case_sha256: str) -> tuple[Path, dict]:
    """Refuse a state that is missing, unattributable or of another case; return its record."""
    if not state.is_file():
        raise SystemExit(f"refused: restart state {state} does not exist")
    parent = parent_run(state)
    if parent is None:
        raise SystemExit(f"refused: restart state {state} cannot be checked: "
                         f"no {MANIFEST} in its run folder")
    record = json.loads((parent / MANIFEST).read_text(encoding="utf-8"))
    if record.get("case_sha256") != case_sha256:
        raise SystemExit(f"refused: restart state {state} belongs to another case (case_sha256 differs)")
    if out.resolve().is_relative_to(parent):
        raise SystemExit("refused: a restart writes to its own folder, not inside its parent run")
    parent_manifest = {"path": str(parent / MANIFEST), "sha256": sha256(parent / MANIFEST)}
    return parent, {"path": str(state.resolve()), "sha256": sha256(state),
                    "parent_manifest": parent_manifest}


def append_progress(out: Path, event: str, **values) -> None:
    """Append one fsynced progress event to ``out/progress.jsonl``."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(out / PROGRESS, flags, 0o644)
    payload = {"schema": "pyoomph-runner-progress-v1",
               "recorded_at": dt.datetime.now(dt.UTC).isoformat(), "event": event, **values}
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    case_path = args.case.resolve()
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"refused: cannot read case {args.case}: {exc}") from exc
    validate_case(case)
    problems = option_problems(args) + output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    restart, parent = None, None
    if args.restart_from is not None:
        parent, restart = check_restart(args.restart_from, args.out, case_sha256)
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_sha256}))
        return 0
    try:
        import pyoomph
    except ImportError as exc:
        raise SystemExit("refused: pyoomph is not importable; use the pinned environment "
                         "(pyproject.toml, uv.lock)") from exc
    from taylor_culick_fem.case_adapter import case_to_driver_args
    from taylor_culick_fem.viscoelastic_driver import (
        _enforce_launcher_affinity,
        _read_restart,
        run_case,
    )

    # Everything the driver would refuse is refused here, before the folder exists.
    try:
        evidence = restart_evidence(args.restart_from, parent) if restart is not None else None
        driver_args = case_to_driver_args(case_path, args.out, restart_evidence=evidence,
                                          t_end=args.t_end, far_field_extension=args.far_field_extension,
                                          run_dir=args.out)
        _read_restart(driver_args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"refused: {exc}") from exc
    affinity = _enforce_launcher_affinity()

    args.out.mkdir(parents=True, exist_ok=True)
    commit = component_commit(ROOT)
    runner = Path(__file__).resolve()
    manifest = {
        "schema": RUNNER_SCHEMA,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "case_file": str(case_path),
        "case_sha256": case_sha256,
        "case_id": case["case_id"],
        "case_fields_used": CASE_FIELDS_USED,
        "units": UNITS,
        "argv": [str(runner), *raw],
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "component_commit": commit,
        "pyoomph": pyoomph_revision(),
        "pyoomph_module": getattr(pyoomph, "__file__", None),
    }
    if restart is not None:
        manifest["restart_from"] = restart
    try:
        create_run_manifest(args.out, manifest)
    except FileExistsError as exc:
        raise SystemExit(f"refused: {args.out / MANIFEST} exists; a run manifest is never replaced") from exc
    append_progress(args.out, "started", case_id=case["case_id"], component_commit=commit,
                    launcher_affinity=affinity)
    try:
        summary = run_case(driver_args)
    except BaseException as exc:
        append_progress(args.out, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    append_progress(args.out, "completed", status=summary.get("status"),
                    physical_time=summary.get("t_reached"), primary_observable=summary.get("ustar_final"))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return exit_status(summary)


if __name__ == "__main__":
    raise SystemExit(main())
