#!/usr/bin/env python3
"""Run one validated ``pyoomph-case-v1`` viscoelastic axisymmetric case."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "validation"))

from run_viscoelastic_axi import (
    _enforce_launcher_affinity,
    build_parser,
    run_case,
    validate_args,
)


def append_progress(event: str, **values) -> None:
    """Append one fsynced runner-owned progress event inside the attempt."""
    runtime = Path.cwd() / "runtime"
    runtime.mkdir(exist_ok=True)
    path = runtime / "progress.jsonl"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    payload = {
        "schema": "pyoomph-runner-progress-v1",
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
        "event": event,
        **values,
    }
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _value(mapping: dict, key: str, context: str):
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"missing {context}.{key}") from exc


def _mapping(mapping: dict, key: str, context: str) -> dict:
    value = _value(mapping, key, context)
    if not isinstance(value, dict):
        raise TypeError(f"{context}.{key} must be an object")
    return value


def case_to_driver_args(case_path: Path, results: Path) -> argparse.Namespace:
    """Translate the repository's serialized case into the existing driver CLI."""
    serialized_case = case_path.read_bytes()
    case = json.loads(serialized_case)
    if not isinstance(case, dict):
        raise TypeError("case must be a JSON object")
    if case.get("schema") != "pyoomph-case-v1":
        raise ValueError("case must use schema pyoomph-case-v1")
    geometry = _mapping(case, "geometry", "case")
    if geometry != {
        "kind": "axisymmetric",
        "coordinates": ["r", "z"],
        "frame": "laboratory",
    }:
        raise ValueError("case geometry must be axisymmetric laboratory coordinates [r,z]")

    physics = _mapping(case, "physics", "case")
    mesh = _mapping(case, "mesh", "case")
    tolerances = _mapping(case, "tolerances", "case")
    output = _mapping(case, "output", "case")
    completion = _mapping(case, "completion", "case")
    solver = _mapping(case, "solver", "case")
    arrest = _mapping(case, "arrest", "case")
    if output.get("write_states") is not False:
        raise ValueError("driver requires output.write_states=false")

    relaxation = _value(physics, "relaxation_time", "physics")
    relaxation_text = "inf" if str(relaxation).lower() == "inf" else str(float(relaxation))
    arguments = [
        "--results",
        str(results.resolve()),
        "--oh",
        str(_value(physics, "Oh", "physics")),
        "--elastic-modulus",
        str(_value(physics, "elastic_modulus", "physics")),
        "--relaxation-time",
        relaxation_text,
        "--stabilization",
        str(_value(physics, "stabilization", "physics")),
        "--length",
        str(_value(mesh, "length", "mesh")),
        "--r0",
        str(_value(mesh, "r0", "mesh")),
        "--resolution",
        str(_value(mesh, "resolution", "mesh")),
        "--far-resolution",
        str(_value(mesh, "far_resolution", "mesh")),
        "--output-step",
        str(_value(output, "cadence", "output")),
        "--checkpoint-every",
        str(_value(output, "checkpoint_every", "output")),
        "--max-step",
        str(_value(solver, "max_step", "solver")),
        "--max-newton",
        str(_value(solver, "max_newton", "solver")),
        "--temporal-error",
        str(_value(tolerances, "temporal_error", "tolerances")),
        "--max-volume-error",
        str(_value(tolerances, "max_volume_error", "tolerances")),
        "--max-tip-pin-residual",
        str(_value(tolerances, "max_tip_pin_residual", "tolerances")),
        "--min-radius-increase",
        str(_value(tolerances, "min_radius_increase", "tolerances")),
        "--min-polymer-stress",
        str(_value(tolerances, "min_polymer_stress", "tolerances")),
        "--opening-ustar",
        str(_value(arrest, "opening_ustar", "arrest")),
        "--arrest-ustar",
        str(_value(arrest, "arrest_ustar", "arrest")),
        "--arrest-consecutive",
        str(_value(arrest, "consecutive_samples", "arrest")),
        "--arrest-trend-tolerance",
        str(_value(arrest, "trend_tolerance", "arrest")),
        "--arrest-radius-span",
        str(_value(arrest, "radius_span_tolerance", "arrest")),
    ]
    sponge = physics.get("sponge")
    if sponge is not None:
        if not isinstance(sponge, dict):
            raise TypeError("physics.sponge must be an object")
        required = {"start", "width", "gamma_max", "profile"}
        if set(sponge) != required:
            raise ValueError(
                "physics.sponge requires exactly start, width, gamma_max and profile"
            )
        arguments.extend(
            [
                "--sponge-start",
                str(sponge["start"]),
                "--sponge-width",
                str(sponge["width"]),
                "--sponge-gamma-max",
                str(sponge["gamma_max"]),
                "--sponge-profile",
                str(sponge["profile"]),
            ]
        )
    if "max_positive_sponge_power" in tolerances:
        arguments.extend(
            [
                "--max-positive-sponge-power",
                str(tolerances["max_positive_sponge_power"]),
            ]
        )

    kind = _value(completion, "kind", "completion")
    if kind == "event":
        if completion.get("value") != "first_post_startup_turning_point":
            raise ValueError("unsupported event completion")
        arguments.extend(["--t-end", str(_value(completion, "horizon", "completion"))])
        arguments.append("--stop-when-arrested")
    elif kind == "time":
        arguments.extend(["--t-end", str(_value(completion, "value", "completion"))])
        if completion.get("allow_recoil") is True:
            arguments.append("--allow-recoil")
    else:
        raise ValueError(f"unsupported completion kind: {kind!r}")
    if "restart" in case:
        arguments.extend(["--restart-case", str(case_path.resolve())])

    parser = build_parser()
    parsed = parser.parse_args(arguments)
    validate_args(parser, parsed)
    # Carry the bytes which were actually validated, rather than rereading a
    # project-context file that may change before the immutable attempt starts.
    parsed.serialized_case = serialized_case
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    args = parser.parse_args()
    case_path = args.case.resolve()
    affinity = _enforce_launcher_affinity()
    append_progress("started", case=str(case_path), launcher_affinity=affinity)
    try:
        driver_args = case_to_driver_args(case_path, Path.cwd())
        metadata = run_case(driver_args)
    except BaseException as exc:
        append_progress("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    append_progress(
        "completed",
        status=metadata["status"],
        physical_time=metadata["t_reached"],
        primary_observable=metadata["ustar_final"],
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
