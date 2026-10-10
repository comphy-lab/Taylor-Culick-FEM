"""Validate ``pyoomph-case-v1`` cases for the axisymmetric viscoelastic runner.

This module imports neither pyoomph nor the solver, so a case can be checked anywhere
(``run_axisymmetric_viscoelastic.py --check``). ``case_problems`` returns every problem at
once; ``validate_case`` refuses a case naming all of them. ``driver_arguments`` maps a valid
case onto the solver driver's command-line options.

A case carries the physics and the numerics of one run. Continuation is not part of the
case: a restart uses the same case file with ``--restart-from``, and may only extend the
stop time (``--t-end``) or lengthen the far-field membrane (``--far-field-extension``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

SCHEMA = "pyoomph-case-v1"
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
GEOMETRY = {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"}
ARREST_EVENT = "first_post_startup_turning_point"
HEIGHT_EVENT = "height_plateau"
HEIGHT_FIELDS = ("hold", "minimum_window", "window_time_fraction", "slope_window_fraction",
                 "instantaneous_slope_fraction", "residual_rms_fraction",
                 "adjacent_mean_fraction", "residual_envelope_growth_fraction")
TOLERANCES = ("temporal_error", "max_volume_error", "max_tip_pin_residual",
              "min_radius_increase", "min_polymer_stress")
ARREST = ("opening_ustar", "arrest_ustar", "trend_tolerance", "radius_span_tolerance")

#: Every case field the runner reads; the run manifest records this list.
CASE_FIELDS_USED = [
    "schema", "case_id", "geometry.kind", "geometry.coordinates", "geometry.frame",
    "nondimensionalization",
    "physics.Oh", "physics.elastic_modulus", "physics.relaxation_time",
    "physics.stabilization", "physics.supg_factor", "physics.passive_conformation",
    "physics.sponge", "physics.far_field",
    "mesh.length", "mesh.r0", "mesh.resolution", "mesh.far_resolution",
    "output.cadence", "output.write_states", "output.field_stride", "output.checkpoint_every",
    "solver.max_step", "solver.max_newton",
    *(f"tolerances.{name}" for name in TOLERANCES), "tolerances.max_positive_sponge_power",
    *(f"arrest.{name}" for name in ARREST), "arrest.consecutive_samples",
    "completion.kind", "completion.value", "completion.horizon", "completion.allow_recoil",
    *(f"completion.{name}" for name in HEIGHT_FIELDS), "completion.oscillation_period",
]


def _number(value: object) -> bool:
    """A finite int or float; booleans and integers too large for a float are not numbers here."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive(section: dict, key: str, label: str, problems: list[str]) -> None:
    value = section.get(key)
    if not _number(value) or value <= 0:
        problems.append(f"{label}.{key} must be a finite positive number")


def _integer(section: dict, key: str, label: str, minimum: int, problems: list[str]) -> None:
    value = section.get(key)
    if type(value) is not int or value < minimum:
        problems.append(f"{label}.{key} must be an integer of at least {minimum}")


def _section(case: dict, key: str, problems: list[str], required: bool = True) -> dict:
    value = case.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        problems.append(f"case.{key} must be an object")
        return {}
    return value


def _physics_problems(physics: dict, mesh: dict, problems: list[str]) -> None:
    _positive(physics, "Oh", "physics", problems)
    modulus = physics.get("elastic_modulus")
    if not _number(modulus) or modulus < 0:
        problems.append("physics.elastic_modulus must be a finite non-negative number")
    relaxation = physics.get("relaxation_time")
    if not (relaxation == "inf" or (_number(relaxation) and relaxation > 0)):
        problems.append('physics.relaxation_time must be "inf" or a finite positive number')
    if physics.get("stabilization") not in ("supg", "none"):
        problems.append('physics.stabilization must be "supg" or "none"')
    if "supg_factor" in physics:
        _positive(physics, "supg_factor", "physics", problems)
    passive = physics.get("passive_conformation", False)
    if type(passive) is not bool:
        problems.append("physics.passive_conformation must be a boolean")
    elif modulus == 0 and not passive:
        problems.append("physics.elastic_modulus = 0 requires physics.passive_conformation = true")
    sponge = physics.get("sponge")
    if sponge is not None:
        if not isinstance(sponge, dict):
            problems.append("physics.sponge must be an object")
        elif set(sponge) != {"start", "width", "gamma_max", "profile"}:
            problems.append("physics.sponge requires exactly start, width, gamma_max and profile")
        else:
            start, width, gamma = sponge["start"], sponge["width"], sponge["gamma_max"]
            if not _number(start) or start < 0:
                problems.append("physics.sponge.start must be a finite non-negative number")
            if not _number(width) or width <= 0:
                problems.append("physics.sponge.width must be a finite positive number")
            if not _number(gamma) or gamma < 0:
                problems.append("physics.sponge.gamma_max must be a finite non-negative number")
            if sponge["profile"] != "quintic":
                problems.append('physics.sponge.profile must be "quintic"')
            length = mesh.get("length")
            if _number(start) and _number(width) and _number(length) and start + width > length:
                problems.append("physics.sponge start + width must not exceed the window length")
    far_field = physics.get("far_field")
    if far_field is not None:
        if not isinstance(far_field, dict):
            problems.append("physics.far_field must be an object")
        elif far_field.get("mode") == "rest":
            if set(far_field) != {"mode"}:
                problems.append("physics.far_field mode 'rest' takes no other keys")
        elif far_field.get("mode") == "membrane":
            if passive is True or relaxation != "inf":
                problems.append("physics.far_field membrane requires active elasticity with "
                                'physics.relaxation_time = "inf" and no passive_conformation')
            allowed = {"mode", "inner_fraction", "extension", "node_spacing"}
            if not set(far_field) <= allowed or "extension" not in far_field:
                problems.append("physics.far_field membrane requires extension and accepts only "
                                "inner_fraction, extension and node_spacing")
            else:
                _positive(far_field, "extension", "physics.far_field", problems)
                if "node_spacing" in far_field:
                    _positive(far_field, "node_spacing", "physics.far_field", problems)
                fraction = far_field.get("inner_fraction", 0.5)
                if not _number(fraction) or not 0 < fraction < 1:
                    problems.append("physics.far_field.inner_fraction must lie strictly between 0 and 1")
        else:
            problems.append(f"unsupported physics.far_field mode: {far_field.get('mode')!r}")


def _completion_problems(completion: dict, physics: dict, output: dict, problems: list[str]) -> None:
    kind, value = completion.get("kind"), completion.get("value")
    if kind == "time":
        if not _number(value) or value <= 0:
            problems.append("completion.value must be a finite positive stop time")
        if "allow_recoil" in completion and type(completion["allow_recoil"]) is not bool:
            problems.append("completion.allow_recoil must be a boolean")
    elif kind == "event" and value == ARREST_EVENT:
        _positive(completion, "horizon", "completion", problems)
        if output.get("field_stride", 1) != 1:
            problems.append("tip-arrest completion requires output.field_stride = 1")
    elif kind == "event" and value == HEIGHT_EVENT:
        _positive(completion, "horizon", "completion", problems)
        for name in HEIGHT_FIELDS:
            _positive(completion, name, "completion", problems)
        if completion.get("oscillation_period") is not None:
            _positive(completion, "oscillation_period", "completion", problems)
        if not (_number(physics.get("elastic_modulus")) and physics["elastic_modulus"] > 0):
            problems.append("height-plateau completion requires physics.elastic_modulus > 0")
    elif kind == "event":
        problems.append(f"unsupported completion event: {value!r} "
                        f"(use {ARREST_EVENT!r} or {HEIGHT_EVENT!r})")
    else:
        problems.append(f"unsupported completion kind: {kind!r} (use 'time' or 'event')")


def case_problems(case: object) -> list[str]:
    """Every problem with ``case``; an empty list means the runner accepts it."""
    if not isinstance(case, dict):
        return ["case must be a JSON object"]
    problems: list[str] = []
    if case.get("schema") != SCHEMA:
        problems.append(f"case.schema must be {SCHEMA!r}")
    if not isinstance(case.get("case_id"), str) or not CASE_ID.fullmatch(case["case_id"]):
        problems.append("case.case_id must match " + CASE_ID.pattern)
    if "restart" in case:
        problems.append("case.restart is not accepted: a restart uses the same case with "
                        "--restart-from <checkpoint>")
    geometry = case.get("geometry")
    if not isinstance(geometry, dict):
        problems.append("case.geometry must be an object")
    else:
        for key in ("kind", "coordinates", "frame"):
            if geometry.get(key) != GEOMETRY[key]:
                problems.append(f"case geometry must be axisymmetric laboratory coordinates [r,z]: "
                                f"geometry.{key} is {geometry.get(key)!r}, not {GEOMETRY[key]!r}")
        extra = sorted(set(geometry) - set(GEOMETRY))
        if extra:
            problems.append(f"case.geometry has unexpected keys: {', '.join(extra)}")
    nondim = case.get("nondimensionalization")
    if not isinstance(nondim, dict) or not nondim:
        problems.append("case.nondimensionalization must be a non-empty object")
    physics = _section(case, "physics", problems)
    mesh = _section(case, "mesh", problems)
    output = _section(case, "output", problems)
    solver = _section(case, "solver", problems)
    tolerances = _section(case, "tolerances", problems)
    arrest = _section(case, "arrest", problems)
    completion = _section(case, "completion", problems)
    _physics_problems(physics, mesh, problems)
    for key in ("length", "r0", "resolution", "far_resolution"):
        _positive(mesh, key, "mesh", problems)
    if _number(mesh.get("length")) and _number(mesh.get("resolution")) \
            and mesh["length"] <= mesh["resolution"]:
        problems.append("mesh.length must exceed mesh.resolution")
    _positive(output, "cadence", "output", problems)
    if output.get("write_states") is not False:
        problems.append("output.write_states must be false (the driver writes its own checkpoints)")
    _integer(output, "checkpoint_every", "output", 0, problems)
    if "field_stride" in output:
        _integer(output, "field_stride", "output", 1, problems)
    stride, every = output.get("field_stride", 1), output.get("checkpoint_every")
    if type(stride) is int and stride >= 1 and type(every) is int and every % stride:
        problems.append("output.checkpoint_every must be a multiple of output.field_stride")
    _positive(solver, "max_step", "solver", problems)
    _integer(solver, "max_newton", "solver", 1, problems)
    for key in TOLERANCES:
        _positive(tolerances, key, "tolerances", problems)
    if "max_positive_sponge_power" in tolerances:
        _positive(tolerances, "max_positive_sponge_power", "tolerances", problems)
    for key in ARREST:
        _positive(arrest, key, "arrest", problems)
    _integer(arrest, "consecutive_samples", "arrest", 2, problems)
    _completion_problems(completion, physics, output, problems)
    return problems


def validate_case(case: object) -> None:
    """Refuse an invalid case, naming every problem."""
    problems = case_problems(case)
    if problems:
        raise SystemExit("refused: invalid case:\n  - " + "\n  - ".join(problems))


def stop_time(case: dict) -> float:
    """The declared stop time: the time value, or the event's horizon."""
    completion = case["completion"]
    return float(completion["value"] if completion["kind"] == "time" else completion["horizon"])


def driver_arguments(case: dict, out: Path, *, t_end: float | None = None,
                     far_field_extension: float | None = None) -> list[str]:
    """Map a valid case onto the driver's options; ``ValueError`` names every problem.

    ``t_end`` replaces the declared stop time and ``far_field_extension`` lengthens the
    membrane; both are runner options, so the run manifest records them in argv.
    """
    problems = case_problems(case)
    if problems:
        raise ValueError("invalid case: " + "; ".join(problems))
    physics, mesh, output = case["physics"], case["mesh"], case["output"]
    solver, tolerances = case["solver"], case["tolerances"]
    arrest, completion = case["arrest"], case["completion"]
    relaxation = physics["relaxation_time"]
    arguments = [
        "--results", str(Path(out).resolve()),
        "--oh", str(physics["Oh"]),
        "--elastic-modulus", str(physics["elastic_modulus"]),
        "--relaxation-time", "inf" if relaxation == "inf" else str(float(relaxation)),
        "--stabilization", str(physics["stabilization"]),
        "--supg-factor", str(physics.get("supg_factor", 1.0)),
        "--length", str(mesh["length"]),
        "--r0", str(mesh["r0"]),
        "--resolution", str(mesh["resolution"]),
        "--far-resolution", str(mesh["far_resolution"]),
        "--output-step", str(output["cadence"]),
        "--field-stride", str(output.get("field_stride", 1)),
        "--checkpoint-every", str(output["checkpoint_every"]),
        "--max-step", str(solver["max_step"]),
        "--max-newton", str(solver["max_newton"]),
        "--temporal-error", str(tolerances["temporal_error"]),
        "--max-volume-error", str(tolerances["max_volume_error"]),
        "--max-tip-pin-residual", str(tolerances["max_tip_pin_residual"]),
        "--min-radius-increase", str(tolerances["min_radius_increase"]),
        "--min-polymer-stress", str(tolerances["min_polymer_stress"]),
        "--opening-ustar", str(arrest["opening_ustar"]),
        "--arrest-ustar", str(arrest["arrest_ustar"]),
        "--arrest-consecutive", str(arrest["consecutive_samples"]),
        "--arrest-trend-tolerance", str(arrest["trend_tolerance"]),
        "--arrest-radius-span", str(arrest["radius_span_tolerance"]),
    ]
    if physics.get("passive_conformation", False):
        arguments.append("--passive-conformation")
    sponge = physics.get("sponge")
    if sponge is not None:
        arguments += ["--sponge-start", str(sponge["start"]), "--sponge-width", str(sponge["width"]),
                      "--sponge-gamma-max", str(sponge["gamma_max"]),
                      "--sponge-profile", str(sponge["profile"])]
    far_field = physics.get("far_field")
    if far_field is not None and far_field["mode"] == "membrane":
        extension = far_field["extension"]
        if far_field_extension is not None:
            if far_field_extension < extension:
                raise ValueError("--far-field-extension may lengthen but never shorten "
                                 "physics.far_field.extension")
            extension = far_field_extension
        arguments += ["--far-field", "membrane", "--far-field-extension", str(extension)]
        for key, flag in (("inner_fraction", "--far-field-inner-fraction"),
                          ("node_spacing", "--far-field-node-spacing")):
            if key in far_field:
                arguments += [flag, str(far_field[key])]
    elif far_field_extension is not None:
        raise ValueError("--far-field-extension requires a membrane far field")
    if "max_positive_sponge_power" in tolerances:
        arguments += ["--max-positive-sponge-power", str(tolerances["max_positive_sponge_power"])]
    end = stop_time(case) if t_end is None else float(t_end)
    if completion["kind"] == "time":
        arguments += ["--t-end", str(end)]
        if completion.get("allow_recoil") is True:
            arguments.append("--allow-recoil")
    elif completion["value"] == ARREST_EVENT:
        arguments += ["--t-end", str(end), "--stop-when-arrested"]
    else:
        arguments += [
            "--t-end", str(end), "--stop-on-height-plateau",
            "--height-hold", str(completion["hold"]),
            "--height-minimum-window", str(completion["minimum_window"]),
            "--height-window-time-fraction", str(completion["window_time_fraction"]),
            "--height-slope-window-fraction", str(completion["slope_window_fraction"]),
            "--height-instantaneous-slope-fraction", str(completion["instantaneous_slope_fraction"]),
            "--height-residual-rms-fraction", str(completion["residual_rms_fraction"]),
            "--height-adjacent-mean-fraction", str(completion["adjacent_mean_fraction"]),
            "--height-envelope-growth-fraction", str(completion["residual_envelope_growth_fraction"]),
        ]
        if completion.get("oscillation_period") is not None:
            arguments += ["--height-oscillation-period", str(completion["oscillation_period"])]
    return arguments
