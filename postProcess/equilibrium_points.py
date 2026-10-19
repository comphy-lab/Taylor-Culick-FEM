#!/usr/bin/env python3
"""Equilibria at exact Ec values from the saved segment states of an equilibrium run.

    python postProcess/equilibrium_points.py <equilibrium run> --case <case> \
        --ec 10 5 2 1 0.5 --out <new folder>

For each requested ``Ec`` the segment whose saved state is nearest in ``log Ec`` is
reloaded on its own grids, walked to the target and solved, and its leading eigenvalues
are computed (``taylor_culick_fem.equilibrium_branch.solve_from_segment``). The case must
be the run's own: its SHA-256 is checked against the run manifest. Output: ``points.json``
in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

LAUNCH_CPUS = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np

from taylor_culick_fem.cases import EQUILIBRIUM, validate_case
from taylor_culick_fem.run_contract import enforce_launcher_affinity, output_problems, read_case, sha256


def segment_states(run: Path) -> list[tuple[float, Path]]:
    """``(log Ec, segment folder)`` of every segment with a saved state."""
    found = []
    for state in sorted((run / "segments").glob("s[0-9][0-9][0-9]/state.npz")):
        found.append((float(np.load(state)["log_ec"]), state.parent))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="output folder of an equilibrium run")
    parser.add_argument("--case", type=Path, required=True, help="the run's case file")
    parser.add_argument("--ec", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new or empty output folder")
    args = parser.parse_args(argv)
    case = read_case(args.case)
    validate_case(case, problem=EQUILIBRIUM)
    try:
        manifest = json.loads((args.run / "run-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"refused: {args.run} has no readable run-manifest.json ({exc})") from exc
    if manifest.get("case_sha256") != sha256(args.case):
        raise SystemExit("refused: --case is not the case of this run (case_sha256 differs)")
    problems = output_problems(args.out)
    if any(not (ec > 0 and math.isfinite(ec)) for ec in args.ec):
        problems.append("--ec values must be finite and positive")
    names = [f"Ec_{ec:g}" for ec in args.ec]
    if len(set(names)) != len(names):
        problems.append("--ec values must give distinct folder names (Ec_<value> at %g precision)")
    states = segment_states(args.run)
    if not states:
        problems.append(f"{args.run} holds no segment state")
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    from pyoomph.generic.jit_cache import set_enabled

    set_enabled(False)
    from taylor_culick_fem.equilibrium_branch import SegmentSettings, solve_from_segment
    enforce_launcher_affinity(LAUNCH_CPUS)

    settings = SegmentSettings.from_case(case)
    args.out.mkdir(parents=True, exist_ok=True)
    points = []
    for ec in args.ec:
        target = math.log(ec)
        _, segment = min(states, key=lambda item, target=target: abs(item[0] - target))
        point = solve_from_segment(settings, segment, ec, args.out / f"Ec_{ec:g}")
        points.append({**point, "segment": segment.name})
        (args.out / "points.json").write_text(json.dumps(points, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(points[-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
