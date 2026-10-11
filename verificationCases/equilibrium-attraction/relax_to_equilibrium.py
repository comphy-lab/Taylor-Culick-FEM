#!/usr/bin/env python3
"""Nonlinear attraction check: relax one equilibrium under the energy flow of another Ec.

    python verificationCases/equilibrium-attraction/relax_to_equilibrium.py \
        verificationCases/equilibrium-attraction/relaxation.json \
        --ec-start 0.8 --ec-target 0.7 --t-end 60 --out <new folder>

The case supplies the reference geometry, mesh and solver settings (its continuation block
is unused). The equilibrium at ``--ec-start`` is reached by natural continuation from
``Ec = 10`` on the reference mesh. The parameter is then switched to ``--ec-target`` and the overdamped energy
flow ``M du/dt = -dE/du`` (the stability mass of the steady problem) is integrated with
adaptive BDF2. The relaxed state is compared with an independent steady solve at the target,
continued from ``Ec = 10`` on a separate problem with the same mesh.
Starting once below and once above the target shows the equilibrium attracting from both
sides; the late decay rate should equal minus the leading eigenvalue at the target.
Output: ``run-manifest.json`` first, then ``relax.csv`` (t, R, H) and ``summary.json`` in
``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np

from taylor_culick_fem.cases import EQUILIBRIUM, validate_case
from taylor_culick_fem.equilibrium_geometry import reference_grids
from taylor_culick_fem.run_contract import (
    append_progress,
    output_problems,
    read_case,
    sha256,
    start_run,
)

#: The case fields this check reads.
CASE_FIELDS_USED = ["schema", "case_id", "geometry.kind", "geometry.coordinates", "geometry.frame",
                    "physics.problem", "physics.r0", "physics.surface_tension",
                    "mesh.elements_per_quarter", "mesh.length", "mesh.growth", "mesh.max_spacing",
                    "solver.newton_tolerance", "solver.max_newton", "regularisation.cap_beta"]


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="equilibrium case supplying geometry, mesh and solver")
    parser.add_argument("--ec-start", type=float, required=True)
    parser.add_argument("--ec-target", type=float, required=True)
    parser.add_argument("--t-end", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True, help="new or empty output folder")
    args = parser.parse_args(raw)
    case_path = args.case.resolve()
    case = read_case(case_path)
    validate_case(case, problem=EQUILIBRIUM)
    problems = output_problems(args.out)
    if not all(math.isfinite(v) and v > 0 for v in (args.ec_start, args.ec_target, args.t_end)):
        problems.append("--ec-start, --ec-target and --t-end must be finite and positive")
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    import pyoomph
    from pyoomph.generic.jit_cache import set_enabled

    set_enabled(False)
    from taylor_culick_fem.equilibrium_branch import SegmentSettings, build_problem

    out = args.out.resolve()
    start_run(out, root=ROOT, runner=Path(__file__), argv=raw, case_path=case_path, case=case,
              case_sha256=sha256(case_path), case_fields_used=CASE_FIELDS_USED,
              units="nondimensional: lengths on h0, stresses on sigma/h0; the time of the energy "
                    "flow is set by its stability mass and is not a physical time",
              pyoomph_module=getattr(pyoomph, "__file__", None))
    append_progress(out, "started", case_id=case["case_id"])
    settings = SegmentSettings.from_case(case)
    grids = reference_grids(settings.r0, settings.length, settings.elements_per_quarter,
                            settings.growth, None, settings.max_spacing, quadratic=True)

    def continued(folder: Path, ec: float):
        """A problem holding the equilibrium at ``ec``, reached by natural continuation."""
        problem = build_problem(settings, grids, folder)
        parameter = problem.get_global_parameter("log_Ec")
        for value in np.linspace(math.log(10.0), math.log(ec), 60):
            parameter.value = float(value)
            problem.solve(max_newton_iterations=settings.max_newton)
        return problem, parameter

    reference, _ = continued(out / "pyoomph-steady", args.ec_target)
    steady = (reference.hole_radius(), reference.rim_height())
    problem, parameter = continued(out / "pyoomph", args.ec_start)
    start = (problem.hole_radius(), problem.rim_height())
    parameter.value = math.log(args.ec_target)
    rows = [(0.0, *start)]
    t, dt = 0.0, 1e-3
    while t < args.t_end:
        problem.run(t + dt, outstep=False, startstep=dt, temporal_error=1e-3)
        t = float(problem.get_current_time())
        rows.append((t, problem.hole_radius(), problem.rim_height()))
        dt = min(2 * dt, 0.5)
    relaxed = rows[-1][1:]
    late = np.array([row for row in rows if row[0] > 0.5 * args.t_end])
    gap = np.abs(late[:, 1] - steady[0]) if len(late) else np.array([])
    rate = float(-np.polyfit(late[:, 0], np.log(gap), 1)[0]) \
        if len(late) > 3 and gap.min() > 0 else math.nan
    with (out / "relax.csv").open("w", encoding="utf-8") as stream:
        stream.write("t,R,H\n")
        stream.writelines(",".join(f"{v:.10g}" for v in row) + "\n" for row in rows)
    summary = {"Ec_start": args.ec_start, "Ec_target": args.ec_target, "R_start": start[0],
               "H_start": start[1], "R_relaxed": relaxed[0], "H_relaxed": relaxed[1],
               "R_steady": steady[0], "H_steady": steady[1], "late_decay_rate": rate}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_progress(out, "completed", primary_observable=relaxed[0])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
