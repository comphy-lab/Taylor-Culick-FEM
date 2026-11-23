#!/usr/bin/env python3
"""depth_averaged_equilibrium: arrested holes of the depth-averaged elastic sheet for many Ec.

The thin-sheet special case of ``run_elastocapillary_equilibrium.py``: an incompressible
neo-Hookean sheet in plane stress, depth-averaged, with a sharp free edge at the reference
radius ``r0`` (``taylor_culick_fem.depth_averaged_equilibrium``). In ``y = ln(r/rho)`` its
equilibrium is one ``Ec``-independent trajectory, ``dT/dy = -(1 + k)``, and a sheet with
elastocapillary number ``Ec`` stops it at ``T = -2/Ec``. The runner integrates that trajectory
once and evaluates every listed ``Ec``. The default case is
``simulationCases/depth_averaged_equilibrium.json``.

    python simulationCases/run_depth_averaged_equilibrium.py <case> --out <folder> [--check]

Output: ``run-manifest.json`` first, ``progress.jsonl``, ``arrest.csv`` (one row per ``Ec``:
``Y = ln(R/r0)``, its lower bound, the log of the stopping remainder, the edge thickness and
the thickness at each listed distance behind the edge) and the metadata ``case.json`` with
the constant ``C``. The problem is static and cheap, so there is no restart. ``--check``
stops after the case and output-folder checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from taylor_culick_fem.cases import DEPTH_AVERAGED, DEPTH_AVERAGED_FIELDS_USED, validate_case
from taylor_culick_fem.run_contract import (
    append_progress,
    output_problems,
    read_case,
    sha256,
    start_run,
    write_json_atomic,
)

CASE_FIELDS_USED = DEPTH_AVERAGED_FIELDS_USED
UNITS = ("nondimensional static problem: lengths on the full sheet thickness h0, tensions on "
         "sigma; Ec = G h0/sigma; no time scale enters")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="pyoomph-case-v1 case file")
    parser.add_argument("--out", type=Path, required=True,
                        help="new or empty output folder; never one that already holds a run")
    parser.add_argument("--check", action="store_true",
                        help="validate the case and output folder, then exit without output")
    return parser


def distance_column(distance: float) -> str:
    return f"h_at_{distance!r}"


def solve(case: dict) -> tuple[list[dict], dict]:
    """Rows of ``arrest.csv`` and the trajectory summary."""
    from taylor_culick_fem.depth_averaged_equilibrium import (
        arrest_log_radius_lower_bound,
        edge_thickness_from_log_radius,
        universal_trajectory,
    )

    numerics = case["solver"]
    trajectory = universal_trajectory(float(numerics["far_seed"]), float(numerics["rtol"]),
                                      float(numerics["atol"]), float(numerics["log_radius_max"]),
                                      float(numerics["max_step"]))
    r0 = float(case["physics"]["r0"])
    distances = [float(d) for d in case["evaluation"]["distances"]]
    rows = []
    for ec in (float(v) for v in case["physics"]["Ec"]):
        log_radius = trajectory.arrest_log_radius(ec)
        edge = trajectory.edge_thickness(ec)
        row = {"Ec": ec, "log_radius_ratio": log_radius,
               "log_radius_ratio_lower_bound": arrest_log_radius_lower_bound(ec),
               "log_remainder": trajectory.log_remainder(log_radius),
               "edge_thickness": edge,
               "edge_quartic_residual": edge - edge_thickness_from_log_radius(ec, log_radius)}
        heights = trajectory.thickness_at_distances(ec, distances, r0) if distances else []
        row.update({distance_column(d): float(h) for d, h in zip(distances, heights, strict=True)})
        rows.append(row)
    summary = {"constant_C": trajectory.constant, "far_seed": trajectory.far_seed,
               "rtol": trajectory.rtol, "atol": trajectory.atol,
               "log_radius_max": trajectory.log_radius_max, "max_step": trajectory.max_step,
               "nfev": int(trajectory._solution.nfev)}
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=DEPTH_AVERAGED)
    problems = output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    case_sha256 = sha256(case_path)
    if args.check:
        print(json.dumps({"check": "passed", "case_id": case["case_id"], "case_sha256": case_sha256}))
        return 0
    out = args.out.resolve()
    commit = start_run(out, root=ROOT, runner=Path(__file__), argv=raw, case_path=case_path,
                       case=case, case_sha256=case_sha256, case_fields_used=CASE_FIELDS_USED,
                       units=UNITS)
    append_progress(out, "started", case_id=case["case_id"], component_commit=commit)
    try:
        rows, summary = solve(case)
        with (out / "arrest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: repr(value) for key, value in row.items()} for row in rows)
        write_json_atomic(out / "case.json", {"case_id": case["case_id"], "status": "passed",
                                              "r0": float(case["physics"]["r0"]),
                                              "points": len(rows), **summary})
    except BaseException as exc:
        append_progress(out, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    append_progress(out, "completed", status="passed", primary_observable=summary["constant_C"])
    print(json.dumps({"status": "passed", "points": len(rows), "constant_C": summary["constant_C"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
