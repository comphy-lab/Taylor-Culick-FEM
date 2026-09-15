#!/usr/bin/env python3
"""Emit JSON settling metrics for one or more Taylor--Culick histories.

Examples::

  python postProcess/analyse_viscoelastic_campaign.py \
    --case 0 newtonian /path/to/control.txt \
    --case 1e-2 viscoelastic /path/to/control.txt --output metrics.json

Each ``--case`` is ``EC SCHEMA CONTROL``.  Supported schemas are
``newtonian`` (seven columns) and ``viscoelastic`` (eight columns).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from viscoelastic_campaign_history import SettlingThresholds, analyse_settling, load_control


def _positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("EC", "SCHEMA", "CONTROL"),
        required=True,
        help="repeatable case: Ec, newtonian|viscoelastic, control.txt",
    )
    result.add_argument("--output", type=Path, help="JSON destination (stdout if omitted)")
    result.add_argument("--r0", type=_positive, default=1.0)
    result.add_argument("--window-duration", type=_positive, default=50.0)
    result.add_argument("--ustar-threshold", type=_positive, default=1e-3)
    result.add_argument("--radius-span-threshold", type=_positive, default=1e-3)
    result.add_argument("--radius-slope-threshold", type=_positive, default=1e-5)
    result.add_argument("--opening-ustar", type=_positive, default=0.02)
    result.add_argument("--minimum-radius-increase", type=_positive, default=0.1)
    return result


def main() -> None:
    args = parser().parse_args()
    thresholds = SettlingThresholds(
        window_duration=args.window_duration,
        maximum_absolute_ustar=args.ustar_threshold,
        maximum_radius_span=args.radius_span_threshold,
        maximum_absolute_radius_slope=args.radius_slope_threshold,
        opening_ustar=args.opening_ustar,
        minimum_radius_increase=args.minimum_radius_increase,
    )
    cases = []
    seen = set()
    for ec_text, schema, path_text in args.case:
        ec = _nonnegative(ec_text)
        if ec in seen:
            raise ValueError(f"duplicate Ec={ec:g}")
        seen.add(ec)
        history = load_control(path_text, schema=schema, r0=args.r0)
        cases.append({"Ec": ec, "source": str(history.source), **analyse_settling(history, thresholds)})
    payload = {"schema": "taylor-culick-settling-v1", "cases": cases}
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
