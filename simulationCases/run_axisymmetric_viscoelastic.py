#!/usr/bin/env python3
"""axisymmetric_viscoelastic: retract one hole in an elastic sheet into one new output folder.

The case file supplies the physics and numerics of one run (``simulationCases/README.md``
lists its fields); the run manifest records the case and the options. The default case is
``simulationCases/axisymmetric_viscoelastic.json``.

    python simulationCases/run_axisymmetric_viscoelastic.py <case> --out <folder> [options]

Before it creates anything, the runner refuses an invalid case or option and an output
folder that already holds anything, for a fresh run and a restart alike. A restart
(``--restart-from <parent>/checkpoints/ckpt_NNNNN.dump``) continues a run into its own new
folder. It may extend the stop time (``--t-end``), lengthen the far-field membrane
(``--far-field-extension``) or set the Newton acceptance as a multiple of the roundoff floor
(``--newton-tolerance-safety``). Its case may also differ from the parent's in numerics only
(``cases.CONTINUATION_NUMERICS``: output cadence, step policy, error control, initial guess
and far-edge treatment); the manifest lists the changed fields and the driver records them
in the numerics schedule. The run list joins it to its parent with ``continues``. ``run-manifest.json`` is written before any other output and is never
replaced. ``--check`` stops after the checks that need neither pyoomph nor the solver.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from taylor_culick_fem.cases import (
    CASE_FIELDS_USED,
    RELAXING_MEMBRANE,
    VISCOELASTIC,
    continuation_case_problems,
    relaxing_endpoint_problem,
    stop_time,
    validate_case,
)
from taylor_culick_fem.lineage import restart_evidence, restart_origin
from taylor_culick_fem.run_contract import (
    UNITS,
    append_progress,
    check_restart,
    output_problems,
    read_case,
    sha256,
    start_run,
)

# "passed": the declared stop (time, arrest or height plateau) was reached and every check
# held. "horizon_reached": an event stop did not occur by its horizon, which the case
# declares as the alternative endpoint. "failed" and "error" exit non-zero.
STOP_STATUSES = ("passed", "horizon_reached")


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
                             "this case (or of one that differs only in numerics); continues into --out")
    parser.add_argument("--t-end", type=float, default=None,
                        help="replace the case's stop time (time value or event horizon)")
    parser.add_argument("--far-field-extension", type=float, default=None,
                        help="lengthen the far-field membrane beyond the case's extension")
    parser.add_argument("--newton-tolerance-safety", type=float, default=None,
                        help="accept a Newton solve below this multiple of the eps*R^2/dt "
                             "roundoff floor (default 10; at least 1)")
    parser.add_argument("--check", action="store_true",
                        help="validate the case, options and output folder, then exit without output")
    return parser


def option_problems(args: argparse.Namespace) -> list[str]:
    """Every invalid option; empty when all are valid."""
    problems = []
    for name, value in (("--t-end", args.t_end), ("--far-field-extension", args.far_field_extension)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            problems.append(f"{name} must be finite and positive")
    safety = args.newton_tolerance_safety
    if safety is not None and (not math.isfinite(safety) or safety < 1):
        problems.append("--newton-tolerance-safety must be finite and at least 1")
    return problems






def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=VISCOELASTIC)
    problems = option_problems(args) + output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    restart, parent = None, None
    if args.restart_from is not None:
        parent, restart = check_restart(args.restart_from, args.out, case_sha256, case=case,
                                        continuation_check=continuation_case_problems)
    evidence = None
    far_field = case["physics"].get("far_field")
    if isinstance(far_field, dict) and far_field.get("mode") == RELAXING_MEMBRANE:
        # The endpoint rule needs no solver, so --check applies it too: the same count the
        # driver makes, from the restart row when this is a continuation.
        try:
            origin = None
            if restart is not None:
                evidence = restart_evidence(args.restart_from, parent)
                origin = restart_origin(evidence, float(case["output"]["cadence"]))
            end = stop_time(case) if args.t_end is None else float(args.t_end)
            endpoint = relaxing_endpoint_problem(case, end, origin)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"refused: {exc}") from exc
        if endpoint is not None:
            raise SystemExit(f"refused: {endpoint}")
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
        _check_relaxing_endpoint,
        _enforce_launcher_affinity,
        _read_restart,
        run_case,
    )

    # Everything the driver would refuse is refused here, before the folder exists.
    try:
        if evidence is None and restart is not None:
            evidence = restart_evidence(args.restart_from, parent)
        driver_args = case_to_driver_args(case_path, args.out, restart_evidence=evidence,
                                          t_end=args.t_end, far_field_extension=args.far_field_extension,
                                          newton_tolerance_safety=args.newton_tolerance_safety,
                                          run_dir=args.out)
        _check_relaxing_endpoint(driver_args, _read_restart(driver_args))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"refused: {exc}") from exc
    affinity = _enforce_launcher_affinity()

    commit = start_run(args.out, root=ROOT, runner=Path(__file__), argv=raw, case_path=case_path,
                       case=case, case_sha256=case_sha256, case_fields_used=CASE_FIELDS_USED,
                       units=UNITS, restart=restart,
                       pyoomph_module=getattr(pyoomph, "__file__", None))
    append_progress(args.out, "started", case_id=case["case_id"], component_commit=commit,
                    launcher_affinity=affinity)
    # The driver reports the relaxing far field's per-sample boundary gates through this.
    driver_args.progress_callback = functools.partial(append_progress, args.out)
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
