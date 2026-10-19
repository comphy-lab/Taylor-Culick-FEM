#!/usr/bin/env python3
"""Compare a steady axisymmetric equilibrium branch with the static depth-averaged theory.

    python modelComparisonCases/equilibrium-depth-averaged/compare_equilibria.py \
        --branch <equilibrium run> --case <its case> --out <new folder> [--min-radius 25] \
        [--allow-legacy]

The depth-averaged theory (``taylor_culick_fem.depth_averaged_equilibrium``) has a sharp
edge at the reference radius ``r0`` and no resolved rim; it scales exactly with ``r0``. For
every point of the branch the script writes the theory hole radius, the ratio
``D = ln(R_axi / R_theory)`` and the deformed distance ``d*`` from the theory edge at which
the theory thickness equals the axisymmetric rim height. It fits
``D(R) = D_inf - c R^(-alpha)`` to the points with ``R >= --min-radius`` and reports
``D_inf`` beside ``ln(r0_eff / r0)``, where ``r0_eff = r0 + 1/2 - pi/8`` is the radius of the
sharp-edged hole that holds the same material as the rounded edge cap of the reference
body. Output: ``comparison.csv`` and ``fit.json`` in ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.optimize import brentq, curve_fit

from taylor_culick_fem.cases import EQUILIBRIUM, validate_case
from taylor_culick_fem.depth_averaged_equilibrium import profile, thickness_at_distance
from taylor_culick_fem.run_contract import figure_problem, output_problems, read_case, sha256


def edge_equivalent_radius(r0: float) -> float:
    """Radius of the sharp-edged hole holding the material of the rounded reference edge."""
    return r0 + 0.5 - math.pi / 8


def read_branch(run: Path) -> list[dict]:
    """Rows of ``branch.csv``; a duplicated ``Ec`` keeps the later (remeshed) row."""
    with (run / "branch.csv").open(encoding="utf-8") as stream:
        rows = {round(float(row["Ec"]), 9): row for row in csv.DictReader(stream)}
    return [{key: float(value) if key not in ("tag",) else value for key, value in row.items()}
            for _, row in sorted(rows.items(), reverse=True)]


def compare(rows: list[dict], r0: float) -> list[dict]:
    out = []
    for row in rows:
        prof = profile(row["Ec"], r0)
        try:
            distance = brentq(lambda d, pr=prof, h=row["H"]: thickness_at_distance(pr, d) - h,
                              1e-6, 1e3)
        except ValueError:
            distance = math.nan
        out.append({"Ec": row["Ec"], "R_axi": row["R"], "H_axi": row["H"], "R_theory": prof["R"],
                    "h_edge_theory": float(prof["lambda_z"][0]),
                    "D": math.log(row["R"] / prof["R"]), "d_star": distance})
    return out


def fit(points: list[dict], r0: float, min_radius: float) -> dict:
    data = np.array([(p["R_axi"], p["D"]) for p in points if p["R_axi"] >= min_radius])
    if len(data) < 4:
        raise SystemExit(f"refused: fewer than four branch points with R >= {min_radius}")
    model = lambda radius, d_inf, c, alpha: d_inf - c * radius ** (-alpha)
    (d_inf, c, alpha), cov = curve_fit(model, data[:, 0], data[:, 1], p0=(0.1, 1.0, 0.7), maxfev=20000)
    return {"D_inf": float(d_inf), "D_inf_err": float(math.sqrt(cov[0, 0])), "c": float(c),
            "alpha": float(alpha), "ln_r0_eff_over_r0": math.log(edge_equivalent_radius(r0) / r0),
            "points": len(data), "min_radius": min_radius}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branch", type=Path, required=True, help="output folder of an equilibrium run")
    parser.add_argument("--case", type=Path, required=True, help="the run's case file")
    parser.add_argument("--out", type=Path, required=True, help="new or empty output folder")
    parser.add_argument("--min-radius", type=float, default=25.0,
                        help="smallest hole radius entering the fit of D(R)")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="accept a run without a manifest naming a clean commit")
    args = parser.parse_args(argv)
    problem = figure_problem(args.branch)
    if problem and not args.allow_legacy:
        raise SystemExit(f"refused: {args.branch}: {problem} (--allow-legacy to accept it)")
    case = read_case(args.case)
    validate_case(case, problem=EQUILIBRIUM)
    manifest_path = args.branch / "run-manifest.json"
    case_checked = manifest_path.is_file()
    if case_checked:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("case_sha256") != sha256(args.case):
            raise SystemExit("refused: --case is not the case of this run (case_sha256 differs)")
    problems = output_problems(args.out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    r0 = float(case["physics"]["r0"])
    points = compare(read_branch(args.branch), r0)
    result = fit(points, r0, args.min_radius)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (args.out / "fit.json").write_text(json.dumps({**result, "r0": r0, "source_problem": problem,
                                                  "case_sha256": sha256(args.case),
                                                  "case_sha256_checked": case_checked},
                                                  indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
