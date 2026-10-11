#!/usr/bin/env python3
"""axisymmetric_elastocapillary_equilibrium: follow the arrested-hole equilibria in Ec.

With an infinite relaxation time the elastic sheet is a Kelvin--Voigt neo-Hookean solid, so
an arrested hole is a static equilibrium between the elastic stress and surface tension,
independent of Oh and of inertia. This runner follows those equilibria from large Ec
towards small Ec by arclength continuation in log Ec, in segments: each segment designs a
mesh in the deformed state of the previous one and pulls it back to the reference body
(``taylor_culick_fem.equilibrium_geometry``), then continues until the hole radius has grown
by the case's factor. The default case is ``simulationCases/elastocapillary_equilibrium.json``.

    python simulationCases/run_elastocapillary_equilibrium.py <case> --out <folder> [--ec-stop EC]

Output: ``run-manifest.json`` first, ``progress.jsonl``, the live metadata ``case.json``,
``branch.csv`` (every accepted point: Ec, hole radius R, full rim height H, far-field
amplitude A, leading eigenvalue of the energy flow, incompressibility error near the edge)
and ``segments/sNNN/`` (grids, final state, rows and deformed free surface of each segment).
A continuation (``--restart-from <run>/segments/sNNN/state.npz``) uses the same case, writes
to its own new folder and may change only the stop value (``--ec-stop``). ``--check`` stops
after the checks that need neither pyoomph nor the solver.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from taylor_culick_fem.cases import EQUILIBRIUM, EQUILIBRIUM_FIELDS_USED, validate_case
from taylor_culick_fem.run_contract import (
    add_contract_options,
    append_progress,
    check_restart,
    continuation_problems,
    output_problems,
    read_case,
    sha256,
    start_run,
    write_json_atomic,
)

CASE_FIELDS_USED = EQUILIBRIUM_FIELDS_USED
UNITS = ("nondimensional static problem: lengths on the full sheet thickness h0, stresses on "
         "sigma/h0; Ec = G h0/sigma is the shear modulus; no time scale enters")
BRANCH_COLUMNS = ("segment", "Ec", "R", "H", "A", "leading_eigenvalue", "J_err", "ndof", "cap_beta",
                  "tag")


def build_parser() -> argparse.ArgumentParser:
    parser = add_contract_options(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--ec-stop", type=float, default=None,
                        help="replace the case's continuation.Ec_stop")
    return parser


def option_problems(args: argparse.Namespace) -> list[str]:
    """Every invalid option; empty when all are valid."""
    if args.ec_stop is not None and (not math.isfinite(args.ec_stop) or args.ec_stop <= 0):
        return ["--ec-stop must be finite and positive"]
    return []


def _append_rows(path: Path, segment: int, rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8") as stream:
        if new:
            stream.write(",".join(BRANCH_COLUMNS) + "\n")
        for row in rows:
            values = [segment] + [row[key] for key in BRANCH_COLUMNS[1:]]
            stream.write(",".join(f"{v:.10g}" if isinstance(v, float) else str(v) for v in values) + "\n")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=EQUILIBRIUM)
    problems = option_problems(args) + output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    restart = None
    if args.restart_from is not None:
        if args.restart_from.name != "state.npz" or args.restart_from.parent.parent.name != "segments":
            raise SystemExit("refused: --restart-from must name <run>/segments/sNNN/state.npz")
        parent, restart = check_restart(args.restart_from, args.out, case_sha256)
        args.parent_run = parent
        changed = continuation_problems(parser, args, parent, ("ec_stop",))
        if changed:
            raise SystemExit("refused: a continuation may change only --ec-stop: " + "; ".join(changed))
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_sha256}))
        return 0
    try:
        import pyoomph
        from pyoomph.generic.jit_cache import set_enabled
    except ImportError as exc:
        raise SystemExit("refused: pyoomph is not importable; use the pinned environment "
                         "(pyproject.toml, uv.lock)") from exc
    from taylor_culick_fem.equilibrium_branch import SegmentSettings, run_segment

    # Each segment compiles its own problem; a shared code cache can be stale across them.
    set_enabled(False)
    settings = SegmentSettings.from_case(case)
    ec_stop = args.ec_stop if args.ec_stop is not None else float(case["continuation"]["Ec_stop"])
    out = args.out.resolve()
    commit = start_run(out, root=ROOT, runner=Path(__file__), argv=raw, case_path=case_path,
                       case=case, case_sha256=case_sha256, case_fields_used=CASE_FIELDS_USED,
                       units=UNITS, restart=restart, pyoomph_module=getattr(pyoomph, "__file__", None))
    append_progress(out, "started", case_id=case["case_id"], component_commit=commit)
    state = args.restart_from.resolve() if args.restart_from is not None else None
    stalled, segment, status = 0, 0, "running"
    last: dict = {}
    try:
        while True:
            try:
                result = run_segment(settings, state, out / "segments" / f"s{segment:03d}", ec_stop)
            except RuntimeError as exc:
                append_progress(out, "segment_failed", segment=segment, error=str(exc)[:300])
                stalled += 1
                result = None
            if result is not None:
                rows = result["rows"]
                _append_rows(out / "branch.csv", segment, rows)
                last = rows[-1]
                append_progress(out, "segment", segment=segment, status=result["status"],
                                Ec=last["Ec"], R=last["R"], H=last["H"])
                stalled = stalled + 1 if len(rows) <= 1 else 0
                state = result["state"]
            if last and last["Ec"] < ec_stop:
                status = "passed"
            elif stalled >= int(case["continuation"]["max_stalled_segments"]):
                status = "stalled"
            write_json_atomic(out / "case.json", {"case_id": case["case_id"], "status": status,
                                                  "segments": segment + 1, "Ec_stop": ec_stop,
                                                  "last": last})
            if status != "running":
                break
            segment += 1
    except BaseException as exc:
        append_progress(out, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    append_progress(out, "completed", status=status, primary_observable=last.get("R"),
                    Ec=last.get("Ec"))
    print(json.dumps({"status": status, "last": last}, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
