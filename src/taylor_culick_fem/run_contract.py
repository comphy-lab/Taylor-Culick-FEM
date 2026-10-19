"""The run contract every runner in this repository follows.

Before it creates anything, a runner refuses an invalid case or option and an output
folder that already holds anything, for a fresh run and a restart alike. A restart names
a state of an earlier run of the same case and writes to its own new folder. The runner
then writes ``run-manifest.json`` exclusively, before any other output, and appends
progress events to ``progress.jsonl``.

``run_case_runner`` carries that sequence for the option-driven study runners; the
viscoelastic case runner uses the same pieces directly. Nothing here imports pyoomph.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

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


def enforce_launcher_affinity(allowed_cpus: set[int] | None = None) -> dict[str, object]:
    """Restore native auxiliary threads to the launcher's Linux CPU mask.

    A runner can retain its mask before importing native libraries and supply it here.
    Other platforms report an unsupported operation and retain their launcher binding.
    """
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"supported": False, "repaired_threads": 0}
    allowed = os.sched_getaffinity(0) if allowed_cpus is None else set(allowed_cpus)
    if not allowed:
        raise RuntimeError("the launcher CPU mask must be non-empty")
    task_root = Path("/proc/self/task")
    if not task_root.is_dir():
        return {"supported": False, "repaired_threads": 0}
    repaired = checked = 0
    for task in task_root.iterdir():
        try:
            tid = int(task.name)
            checked += 1
            if os.sched_getaffinity(tid) != allowed:
                os.sched_setaffinity(tid, allowed)
                if os.sched_getaffinity(tid) != allowed:
                    raise RuntimeError(f"thread {tid} rejected the launcher CPU mask")
                repaired += 1
        except ProcessLookupError:
            continue
    return {"supported": True, "allowed_cpus": sorted(allowed),
            "checked_threads": checked, "repaired_threads": repaired}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_case(path: Path) -> dict:
    """The case at ``path``; an unreadable file is refused."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"refused: cannot read case {path}: {exc}") from exc


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
    descriptor = os.open(Path(out) / PROGRESS, flags, 0o644)
    payload = {"schema": "pyoomph-runner-progress-v1",
               "recorded_at": dt.datetime.now(dt.UTC).isoformat(), "event": event, **values}
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, data: dict, *, sort_keys: bool = True,
                      allow_nan: bool = False) -> None:
    """Replace ``path`` atomically: a reader sees the old record or the new one, never a torn one."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=sort_keys, allow_nan=allow_nan) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def start_run(out: Path, *, root: Path, runner: Path, argv: list[str], case_path: Path,
              case: dict, case_sha256: str, case_fields_used: list[str], units: str = UNITS,
              restart: dict | None = None, pyoomph_module: str | None = None) -> str:
    """Create ``out`` and write its run manifest exclusively; return the component commit."""
    out.mkdir(parents=True, exist_ok=True)
    commit = component_commit(root)
    runner = Path(runner).resolve()
    manifest = {
        "schema": RUNNER_SCHEMA,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "case_file": str(case_path),
        "case_sha256": case_sha256,
        "case_id": case["case_id"],
        "case_fields_used": case_fields_used,
        "units": units,
        "argv": [str(runner), *argv],
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "component_commit": commit,
        "pyoomph": pyoomph_revision(),
        "pyoomph_module": pyoomph_module,
    }
    if restart is not None:
        manifest["restart_from"] = restart
    try:
        create_run_manifest(out, manifest)
    except FileExistsError as exc:
        raise SystemExit(f"refused: {out / MANIFEST} exists; a run manifest is never replaced") from exc
    return commit


def figure_problem(folder: Path) -> str | None:
    """Why the run in ``folder`` cannot feed a figure, or ``None`` when it can.

    A run feeds a figure only when its ``run-manifest.json`` names a clean, full component
    commit and the pyoomph commit, so the figure can state both.
    """
    manifest = Path(folder) / MANIFEST
    if not manifest.is_file():
        return f"no {MANIFEST}"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable {MANIFEST}: {exc}"
    commit = str(record.get("component_commit", ""))
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        return f"component commit {commit!r} is not a clean full commit"
    if not (record.get("pyoomph") or {}).get("commit"):
        return "the manifest records no pyoomph commit"
    return None


def add_contract_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The positional case and the contract options every study runner takes."""
    parser.add_argument("case", type=Path, help="pyoomph-case-v1 case file")
    parser.add_argument("--out", type=Path, required=True,
                        help="new or empty output folder; never one that already holds a run")
    parser.add_argument("--restart-from", type=Path, default=None,
                        help="state written by an earlier run of the same case; continues into --out")
    parser.add_argument("--check", action="store_true",
                        help="validate the case, options and output folder, then exit without output")
    return parser


CONTRACT_DESTS = ("case", "out", "restart_from", "check", "parent_run")


def continuation_problems(parser: argparse.ArgumentParser, args: argparse.Namespace,
                          parent: Path, mutable: tuple[str, ...]) -> list[str]:
    """Every numerical option a continuation changes, other than those in ``mutable``.

    The parent's options are its manifest's argv parsed by the same parser, so defaults
    and spellings compare equal; the case is already bound by its hash.
    """
    record = json.loads((parent / MANIFEST).read_text(encoding="utf-8"))
    try:
        before = vars(parser.parse_args(record.get("argv", [])[1:]))
    except SystemExit:
        return ["the parent run's command line does not parse with this runner's options"]
    now = vars(args)
    return [f"--{name.replace('_', '-')} differs from the parent run ({before.get(name)!r} -> {value!r})"
            for name, value in now.items()
            if name not in CONTRACT_DESTS and name not in mutable and before.get(name) != value]


def run_case_runner(argv: list[str] | None, *, runner: Path, root: Path,
                    parser: argparse.ArgumentParser, problem: str,
                    option_problems: Callable[[argparse.Namespace], list[str]],
                    solve: Callable[[argparse.Namespace, dict, Path | None], dict],
                    continuation_options: tuple[str, ...] = ("tstar_end", "n_outputs")) -> int:
    """Run one option-driven study case under the run contract.

    ``solve(args, case, restart_state)`` runs the solver into ``args.out`` and returns its
    metadata; it is called only after every check has passed and the manifest exists. A
    continuation may change only ``continuation_options``: by default the stop time, and the
    output count, which in these drivers also sets the step and must be rescaled to keep the
    parent's cadence over the remaining span. ``args.parent_run`` names the parent's folder
    so the driver can reuse values the parent resolved.
    """
    from taylor_culick_fem.cases import NEWTONIAN_FIELDS_USED, validate_case

    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=problem)
    problems = option_problems(args) + output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    restart, args.parent_run = None, None
    if args.restart_from is not None:
        args.parent_run, restart = check_restart(args.restart_from, args.out, case_sha256)
        changed = continuation_problems(parser, args, args.parent_run, continuation_options)
        if changed:
            raise SystemExit("refused: a continuation may change only "
                             + ", ".join(f"--{o.replace('_', '-')}" for o in continuation_options)
                             + ": " + "; ".join(changed))
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_sha256}))
        return 0
    try:
        import pyoomph
    except ImportError as exc:
        raise SystemExit("refused: pyoomph is not importable; use the pinned environment "
                         "(pyproject.toml, uv.lock)") from exc
    out = args.out.resolve()
    commit = start_run(out, root=root, runner=runner, argv=raw, case_path=case_path, case=case,
                       case_sha256=case_sha256, case_fields_used=NEWTONIAN_FIELDS_USED,
                       restart=restart, pyoomph_module=getattr(pyoomph, "__file__", None))
    append_progress(out, "started", case_id=case["case_id"], component_commit=commit)
    state = args.restart_from.resolve() if args.restart_from is not None else None
    try:
        summary = solve(args, case, state)
    except BaseException as exc:
        append_progress(out, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    append_progress(out, "completed", primary_observable=summary.get("ustar_final"))
    return 0
