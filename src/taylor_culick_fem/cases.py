"""Validate ``pyoomph-case-v1`` cases for this repository's runners.

The axisymmetric viscoelastic runner reads physics and numerics from its case, and so does
the equilibrium runner (``physics.problem`` = ``axisymmetric_elastocapillary_equilibrium``),
whose case carries the reference geometry, mesh, continuation and solver settings. The
Newtonian study runners (``physics.problem`` = ``planar_comoving``, ``axisymmetric_lab``
or ``axisymmetric_comoving``) read only the physics, ``physics.Oh``, and take every
numerical parameter from their options.

This module imports neither pyoomph nor the solver, so a case can be checked anywhere
(``run_axisymmetric_viscoelastic.py --check``). ``case_problems`` returns every problem at
once; ``validate_case`` refuses a case naming all of them. ``driver_arguments`` maps a valid
case onto the solver driver's command-line options.

A case carries the physics and the numerics of one run. Continuation is not part of the
case: a restart names its parent checkpoint with ``--restart-from``, and may extend the stop
time (``--t-end``) lengthen the far-field membrane (``--far-field-extension``) or set the Newton
acceptance relative to the roundoff floor (``--newton-tolerance-safety``). A
viscoelastic continuation may also use a case that differs from its parent's only in the
numerics listed in ``CONTINUATION_NUMERICS`` (``continuation_case_problems``); the driver
records each change in the run's numerics schedule. An equilibrium continuation may only
move the stop value of ``Ec`` (``--ec-stop``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

SCHEMA = "pyoomph-case-v1"
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
GEOMETRY = {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"}
VISCOELASTIC = "axisymmetric_viscoelastic"
#: Geometry of each Newtonian study problem, keyed by ``physics.problem``.
NEWTONIAN_GEOMETRY = {
    "planar_comoving": {"kind": "planar", "coordinates": ["x", "y"], "frame": "co-moving"},
    "axisymmetric_lab": {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"},
    "axisymmetric_comoving": {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "co-moving"},
}
#: Every case field the Newtonian study runners read.
NEWTONIAN_FIELDS_USED = ["schema", "case_id", "geometry.kind", "geometry.coordinates",
                         "geometry.frame", "nondimensionalization", "physics.problem", "physics.Oh"]
EQUILIBRIUM = "axisymmetric_elastocapillary_equilibrium"
EQUILIBRIUM_GEOMETRY = {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "reference"}
EQUILIBRIUM_SECTIONS = {
    "physics": (("r0", "positive"), ("surface_tension", "positive")),
    "mesh": (("elements_per_quarter", "integer"), ("length", "positive"), ("growth", "growth"),
             ("remesh_growth", "growth"), ("max_spacing", "positive")),
    "continuation": (("Ec_start", "positive"), ("Ec_stop", "positive"), ("initial_step", "nonzero"),
                     ("max_step", "positive"), ("segment_steps", "integer"),
                     ("segment_log_radius_growth", "positive"), ("max_stalled_segments", "integer")),
    "solver": (("newton_tolerance", "positive"), ("max_newton", "integer"), ("eigenvalues", "integer")),
}
EQUILIBRIUM_FIELDS_USED = (["schema", "case_id", "geometry.kind", "geometry.coordinates",
                            "geometry.frame", "nondimensionalization", "physics.problem"]
                           + [f"{section}.{key}" for section, keys in EQUILIBRIUM_SECTIONS.items()
                              for key, _ in keys]
                           + ["mesh.columns", "mesh.shoulder_angle", "mesh.blend", "mesh.template",
                              "regularisation.cap_beta", "regularisation.volume_penalty"])
ARREST_EVENT = "first_post_startup_turning_point"
HEIGHT_EVENT = "height_plateau"
HEIGHT_FIELDS = ("hold", "minimum_window", "window_time_fraction", "slope_window_fraction",
                 "instantaneous_slope_fraction", "residual_rms_fraction",
                 "adjacent_mean_fraction", "residual_envelope_growth_fraction")
TOLERANCES = ("temporal_error", "max_volume_error", "max_tip_pin_residual",
              "min_radius_increase", "min_polymer_stress")
ARREST = ("opening_ustar", "arrest_ustar", "trend_tolerance", "radius_span_tolerance")
INTERVAL_STARTS = ("legacy", "max_step", "carry")
INITIAL_GUESSES = ("previous", "extrapolate")
#: The finite-relaxation far field and the settings it requires (none has a default).
RELAXING_MEMBRANE = "relaxing_membrane"
RELAXING_MEMBRANE_KEYS = ("inner_fraction", "extension", "node_spacing")
#: Dotted case fields a viscoelastic continuation may change from its parent's case. They are
#: numerics: the output cadence (which is also the far-field coupling interval), the step
#: policy and error control, and the far-edge treatment. The membrane extension may only grow.
CONTINUATION_NUMERICS = (
    "output.cadence", "solver.max_step", "solver.interval_start", "solver.initial_guess",
    "solver.temporal_error_fields", "solver.newton_floor_max_dt", "tolerances.temporal_error",
    "physics.far_field.extension", "physics.far_field.interpolation",
    "physics.far_field.history", "physics.far_field.grading", "physics.far_field.auto_extend",
)

#: Every case field the runner reads; the run manifest records this list.
CASE_FIELDS_USED = [
    "schema", "case_id", "geometry.kind", "geometry.coordinates", "geometry.frame",
    "nondimensionalization",
    "physics.Oh", "physics.elastic_modulus", "physics.relaxation_time",
    "physics.stabilization", "physics.supg_factor", "physics.passive_conformation",
    "physics.sponge", "physics.far_field",
    "mesh.length", "mesh.r0", "mesh.resolution", "mesh.far_resolution",
    "output.cadence", "output.write_states", "output.field_stride", "output.checkpoint_every",
    "solver.max_step", "solver.max_newton", "solver.interval_start", "solver.initial_guess",
    "solver.temporal_error_fields", "solver.newton_floor_max_dt",
    "solver.relaxation_start_fraction",
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
    if not (relaxation == "inf" or (_number(relaxation) and relaxation >= 0)):
        problems.append('physics.relaxation_time must be "inf" or a finite non-negative number '
                        "(0 is the Newtonian endpoint)")
    if physics.get("stabilization") not in ("supg", "none"):
        problems.append('physics.stabilization must be "supg" or "none"')
    if "supg_factor" in physics:
        _positive(physics, "supg_factor", "physics", problems)
    passive = physics.get("passive_conformation", False)
    if type(passive) is not bool:
        problems.append("physics.passive_conformation must be a boolean")
    # elastic_modulus = 0 without passive_conformation, like relaxation_time = 0, is the exact
    # Newtonian endpoint: the run carries no conformation unknowns.
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
            if passive is True or relaxation != "inf" or modulus == 0:
                problems.append("physics.far_field membrane requires active elasticity with "
                                'physics.relaxation_time = "inf" and no passive_conformation')
            allowed = {"mode", "inner_fraction", "extension", "node_spacing", "interpolation",
                       "history", "grading", "auto_extend"}
            if not set(far_field) <= allowed or "extension" not in far_field:
                problems.append("physics.far_field membrane requires extension and accepts only "
                                "inner_fraction, extension, node_spacing, interpolation, history, "
                                "grading and auto_extend")
            else:
                _positive(far_field, "extension", "physics.far_field", problems)
                if "node_spacing" in far_field:
                    _positive(far_field, "node_spacing", "physics.far_field", problems)
                fraction = far_field.get("inner_fraction", 0.5)
                if not _number(fraction) or not 0 < fraction < 1:
                    problems.append("physics.far_field.inner_fraction must lie strictly between 0 and 1")
                _far_edge_problems(far_field, problems)
        elif far_field.get("mode") == RELAXING_MEMBRANE:
            _relaxing_membrane_problems(far_field, physics, problems)
        else:
            problems.append(f"unsupported physics.far_field mode: {far_field.get('mode')!r}")


def _relaxing_membrane_problems(far_field: dict, physics: dict, problems: list[str]) -> None:
    """The finite-relaxation far field: every coupling setting explicit, active elasticity."""
    if set(far_field) != {"mode", *RELAXING_MEMBRANE_KEYS}:
        problems.append("physics.far_field relaxing_membrane requires exactly mode, "
                        + ", ".join(RELAXING_MEMBRANE_KEYS))
        return
    _positive(far_field, "extension", "physics.far_field", problems)
    _positive(far_field, "node_spacing", "physics.far_field", problems)
    fraction = far_field["inner_fraction"]
    if not _number(fraction) or not 0 < fraction < 1:
        problems.append("physics.far_field.inner_fraction must lie strictly between 0 and 1")
    modulus, relaxation = physics.get("elastic_modulus"), physics.get("relaxation_time")
    active = (_number(modulus) and modulus > 0
              and (relaxation == "inf" or (_number(relaxation) and relaxation > 0)))
    if physics.get("passive_conformation", False) is not False or not active:
        problems.append("physics.far_field relaxing_membrane requires active Oldroyd-B elasticity: "
                        "elastic_modulus > 0, relaxation_time > 0 and no passive_conformation")
    sponge = physics.get("sponge")
    if isinstance(sponge, dict) and _number(sponge.get("gamma_max")) and sponge["gamma_max"] > 0:
        problems.append("physics.far_field relaxing_membrane cannot be combined with a sponge")


def output_sample_count(t_end: float, cadence: float) -> int:
    """Number of control samples a fresh run writes up to ``t_end`` at ``cadence``.

    The driver's output grid (``viscoelastic_driver._output_times``): every whole cadence up
    to ``t_end``, plus a short final interval when ``t_end`` is not on the grid.
    """
    count = math.floor(t_end / cadence)
    if count < 1 or cadence * count < t_end * (1 - 1e-14):
        count += 1
    return count


def relaxing_last_sample(t_end: float, cadence: float, origin: dict | None = None) -> int:
    """The cumulative control index of a run's last sample: the driver's output grid.

    A fresh run (``origin`` None) samples every whole ``cadence`` up to ``t_end`` plus a
    short final interval (``output_sample_count``). A continuation counts on from its
    restart row; ``origin`` carries that row's ``control_index``, its landed ``time``, its
    ``nominal_time`` on the parent's schedule and whether the cadence changed
    (``cadence_changed``), as ``viscoelastic_driver._read_restart`` and
    ``lineage.restart_origin`` derive them. Its targets are those of
    ``viscoelastic_driver._continuation_targets``: the run's own zero-based grid when the
    cadence is unchanged and the restart row sits on it, otherwise a grid from the nominal
    restart time with ``t_end`` appended; only targets after the landed time count.
    """
    if origin is None:
        return output_sample_count(t_end, cadence)
    index, nominal, landed = int(origin["control_index"]), float(origin["nominal_time"]), float(origin["time"])
    aligned = (not origin["cadence_changed"]
               and math.isclose(nominal, cadence * index, rel_tol=0, abs_tol=1e-9))
    if aligned:
        count = math.floor(t_end / cadence)
        times = [cadence * float(k) for k in range(1, count + 1)]
        if not times or times[-1] < t_end * (1 - 1e-14):
            times.append(t_end)
    else:
        count = math.floor((t_end - nominal) / cadence + 1e-9)
        times = [nominal + cadence * float(k) for k in range(1, count + 1)]
        if not times or times[-1] < t_end * (1 - 1e-14):
            times.append(t_end)
    return index + sum(1 for time in times if time > landed + 1e-6)


def relaxing_endpoint_message(last_sample: int, field_stride: int) -> str | None:
    """Why a relaxing-membrane run ending at ``last_sample`` cannot be launched, if it cannot.

    Its endpoint checkpoint and far-field sidecar are written on a field boundary only, so
    that field indices stay the deterministic stride sequence restarts rely on.
    """
    if last_sample % field_stride:
        return (f"relaxing membrane run would end at sample {last_sample}, not a multiple of "
                f"the field stride {field_stride}, so without a field and a bound checkpoint")
    return None


def relaxing_endpoint_problem(case: dict, t_end: float, origin: dict | None = None) -> str | None:
    """``relaxing_endpoint_message`` for a valid case run to ``t_end`` from ``origin``.

    None for any other far field. ``origin`` is the restart row of a continuation
    (``relaxing_last_sample``); the declared horizon of a case is not checked on its own,
    because a continuation counts its samples on from its restart.
    """
    far = case["physics"].get("far_field")
    if not (isinstance(far, dict) and far.get("mode") == RELAXING_MEMBRANE):
        return None
    output = case["output"]
    last = relaxing_last_sample(float(t_end), float(output["cadence"]), origin)
    return relaxing_endpoint_message(last, output.get("field_stride", 1))


def _far_edge_problems(far_field: dict, problems: list[str]) -> None:
    """The optional far-edge numerics of a membrane far field."""
    if far_field.get("interpolation", "hold") not in ("hold", "linear"):
        problems.append('physics.far_field.interpolation must be "hold" or "linear"')
    if far_field.get("history", "overwrite") not in ("overwrite", "keep"):
        problems.append('physics.far_field.history must be "overwrite" or "keep"')
    grading = far_field.get("grading")
    if grading is not None:
        if not isinstance(grading, dict) or set(grading) != {"near_length", "growth"}:
            problems.append("physics.far_field.grading must be an object with exactly near_length "
                            "and growth")
        elif not (_number(grading["near_length"]) and grading["near_length"] >= 0
                  and _number(grading["growth"]) and grading["growth"] > 1):
            problems.append("physics.far_field.grading needs a finite near_length >= 0 and a "
                            "finite growth > 1")
    auto = far_field.get("auto_extend")
    if auto is not None:
        if not isinstance(auto, dict) or set(auto) != {"tail_fraction", "factor"}:
            problems.append("physics.far_field.auto_extend must be an object with exactly "
                            "tail_fraction and factor")
        elif not (_number(auto["tail_fraction"]) and 0.1 < auto["tail_fraction"] < 1
                  and _number(auto["factor"]) and auto["factor"] > 1):
            problems.append("physics.far_field.auto_extend needs a finite tail_fraction between "
                            "0.1 and 1 and a finite factor > 1")


def _solver_policy_problems(solver: dict, problems: list[str]) -> None:
    """The optional step policy, error control and Newton settings of the solver."""
    if solver.get("interval_start", "legacy") not in INTERVAL_STARTS:
        problems.append("solver.interval_start must be one of " + ", ".join(INTERVAL_STARTS))
    if solver.get("initial_guess", "previous") not in INITIAL_GUESSES:
        problems.append("solver.initial_guess must be one of " + ", ".join(INITIAL_GUESSES))
    fields = solver.get("temporal_error_fields")
    if fields is not None:
        if not isinstance(fields, dict) or not fields or not all(
                isinstance(k, str) and _number(v) and v > 0 for k, v in fields.items()):
            problems.append("solver.temporal_error_fields must be a non-empty object of finite "
                            "positive weights")
        elif ("X" in fields or "U" in fields) and fields.get("X") != fields.get("U"):
            # The pinned pyoomph applies the first ODE field's scale to every global unknown.
            problems.append("solver.temporal_error_fields must weight X and U equally")
    if "newton_floor_max_dt" in solver:
        _positive(solver, "newton_floor_max_dt", "solver", problems)
    fraction = solver.get("relaxation_start_fraction")
    if fraction is not None and not (_number(fraction) and 0 < fraction <= 1):
        problems.append("solver.relaxation_start_fraction must be a number in (0, 1]")


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


def problem_of(case: object) -> str:
    """The problem a case declares: ``physics.problem``, or the viscoelastic case when absent."""
    physics = case.get("physics") if isinstance(case, dict) else None
    problem = physics.get("problem") if isinstance(physics, dict) else None
    return problem if isinstance(problem, str) and problem else VISCOELASTIC


def _newtonian_problems(case: dict, problem: str) -> list[str]:
    problems: list[str] = []
    if case.get("schema") != SCHEMA:
        problems.append(f"case.schema must be {SCHEMA!r}")
    if not isinstance(case.get("case_id"), str) or not CASE_ID.fullmatch(case["case_id"]):
        problems.append("case.case_id must match " + CASE_ID.pattern)
    if "restart" in case:
        problems.append("case.restart is not accepted: a restart uses the same case with "
                        "--restart-from <state>")
    expected = NEWTONIAN_GEOMETRY[problem]
    geometry = case.get("geometry")
    if not isinstance(geometry, dict):
        problems.append("case.geometry must be an object")
    else:
        for key in ("kind", "coordinates", "frame"):
            if geometry.get(key) != expected[key]:
                problems.append(f"case geometry for {problem}: geometry.{key} is "
                                f"{geometry.get(key)!r}, not {expected[key]!r}")
    nondim = case.get("nondimensionalization")
    if not isinstance(nondim, dict) or not nondim:
        problems.append("case.nondimensionalization must be a non-empty object")
    physics = _section(case, "physics", problems)
    if physics.get("problem") != problem:
        problems.append(f"physics.problem must be {problem!r}")
    _positive(physics, "Oh", "physics", problems)
    return problems


def _equilibrium_problems(case: dict) -> list[str]:
    problems: list[str] = []
    if case.get("schema") != SCHEMA:
        problems.append(f"case.schema must be {SCHEMA!r}")
    if not isinstance(case.get("case_id"), str) or not CASE_ID.fullmatch(case["case_id"]):
        problems.append("case.case_id must match " + CASE_ID.pattern)
    if "restart" in case:
        problems.append("case.restart is not accepted: a continuation uses the same case with "
                        "--restart-from <segment state>")
    geometry = case.get("geometry")
    if not isinstance(geometry, dict):
        problems.append("case.geometry must be an object")
    else:
        for key in ("kind", "coordinates"):
            if geometry.get(key) != EQUILIBRIUM_GEOMETRY[key]:
                problems.append(f"case geometry for {EQUILIBRIUM}: geometry.{key} is "
                                f"{geometry.get(key)!r}, not {EQUILIBRIUM_GEOMETRY[key]!r}")
        if geometry.get("frame") not in ("reference", "lagrangian"):
            problems.append("equilibrium geometry.frame must be 'lagrangian' "
                            "or the legacy spelling 'reference'")
    nondim = case.get("nondimensionalization")
    if not isinstance(nondim, dict) or not nondim:
        problems.append("case.nondimensionalization must be a non-empty object")
    for name, keys in EQUILIBRIUM_SECTIONS.items():
        section = _section(case, name, problems)
        for key, kind in keys:
            if kind == "integer":
                _integer(section, key, name, 1, problems)
            elif kind == "positive":
                _positive(section, key, name, problems)
            elif kind == "growth":
                if not _number(section.get(key)) or section[key] < 1:
                    problems.append(f"{name}.{key} must be a number of at least 1")
            elif not _number(section.get(key)) or section[key] == 0:
                problems.append(f"{name}.{key} must be a non-zero number")
    physics = case.get("physics") if isinstance(case.get("physics"), dict) else {}
    if physics.get("problem") != EQUILIBRIUM:
        problems.append(f"physics.problem must be {EQUILIBRIUM!r}")
    cont = case.get("continuation") if isinstance(case.get("continuation"), dict) else {}
    if _number(cont.get("Ec_start")) and _number(cont.get("Ec_stop")) \
            and cont["Ec_stop"] >= cont["Ec_start"]:
        problems.append("continuation.Ec_stop must be below continuation.Ec_start")
    if _number(cont.get("initial_step")) and cont["initial_step"] > 0:
        problems.append("continuation.initial_step must be negative (the branch is followed "
                        "towards smaller Ec)")
    mesh = case.get("mesh") if isinstance(case.get("mesh"), dict) else {}
    if "template" in mesh and mesh["template"] != "rounded-sheet-blocks":
        problems.append("equilibrium mesh.template must be 'rounded-sheet-blocks'")
    if _number(mesh.get("length")) and _number(physics.get("r0")) and mesh["length"] <= physics["r0"] + 1:
        problems.append("mesh.length must exceed physics.r0 + 1 (the edge cap and a sheet column)")
    if mesh.get("columns", "vertical") not in ("vertical", "fan"):
        problems.append("mesh.columns must be 'vertical' or 'fan'")
    if "shoulder_angle" in mesh and (not _number(mesh["shoulder_angle"])
                                     or not 0 < mesh["shoulder_angle"] < 90):
        problems.append("mesh.shoulder_angle must be a number of degrees between 0 and 90")
    if "blend" in mesh and (not _number(mesh["blend"]) or mesh["blend"] <= 0):
        problems.append("mesh.blend must be a positive number")
    regularisation = case.get("regularisation", {})
    if not isinstance(regularisation, dict):
        problems.append("case.regularisation must be an object")
    elif "cap_beta" in regularisation and (not _number(regularisation["cap_beta"])
                                           or regularisation["cap_beta"] < 0):
        problems.append("regularisation.cap_beta must be a non-negative number")
    if isinstance(regularisation, dict) and "volume_penalty" in regularisation \
            and (not _number(regularisation["volume_penalty"])
                 or regularisation["volume_penalty"] < 0):
        problems.append("regularisation.volume_penalty must be a non-negative number")
    return problems


def case_problems(case: object, problem: str | None = None) -> list[str]:
    """Every problem with ``case``; an empty list means the runner accepts it.

    ``problem`` names the runner's problem; by default the case's own ``physics.problem``.
    """
    if not isinstance(case, dict):
        return ["case must be a JSON object"]
    declared = problem_of(case)
    wanted = declared if problem is None else problem
    if wanted in NEWTONIAN_GEOMETRY:
        return _newtonian_problems(case, wanted)
    if wanted == EQUILIBRIUM:
        return _equilibrium_problems(case)
    if wanted != VISCOELASTIC:
        return [f"unknown problem {wanted!r}"]
    problems: list[str] = []
    if declared != VISCOELASTIC:
        problems.append(f"physics.problem must be {VISCOELASTIC!r} or absent for this runner")
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
    _solver_policy_problems(solver, problems)
    for key in TOLERANCES:
        _positive(tolerances, key, "tolerances", problems)
    if "max_positive_sponge_power" in tolerances:
        _positive(tolerances, "max_positive_sponge_power", "tolerances", problems)
    for key in ARREST:
        _positive(arrest, key, "arrest", problems)
    _integer(arrest, "consecutive_samples", "arrest", 2, problems)
    _completion_problems(completion, physics, output, problems)
    return problems


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Dotted leaf paths of a case; an empty object is a leaf."""
    if isinstance(value, dict) and value:
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _numerics_field(path: str) -> str | None:
    """The ``CONTINUATION_NUMERICS`` field that ``path`` belongs to, if any."""
    return next((field for field in CONTINUATION_NUMERICS
                 if path == field or path.startswith(field + ".")), None)


def continuation_case_problems(case: dict, parent: dict) -> tuple[list[str], list[str]]:
    """Problems with continuing ``parent`` (its input case) under ``case``, and the fields changed.

    Only ``CONTINUATION_NUMERICS`` may differ, and the membrane extension may only grow; the
    stop time changes through ``--t-end``. As in the driver, membrane grading is never removed
    and changes only together with a longer extension. Whether that extension also passes a
    membrane that automatic extension has already lengthened needs the checkpoint, so the driver
    checks it when it reads the restart. Both lists are empty for an unchanged case.
    """
    problems: list[str] = []
    changed: set[str] = set()
    current, previous = _flatten(case), _flatten(parent)
    for path in sorted(set(current) | set(previous)):
        if current.get(path, _MISSING) == previous.get(path, _MISSING):
            continue
        field = _numerics_field(path)
        if field is None:
            hint = " (use --t-end)" if path.startswith("completion.") else ""
            problems.append(f"a continuation may not change {path}{hint}")
        else:
            changed.add(field)
    new_ext = current.get("physics.far_field.extension")
    old_ext = previous.get("physics.far_field.extension")
    if _number(new_ext) and _number(old_ext) and new_ext < old_ext:
        problems.append("a continuation may lengthen but never shorten physics.far_field.extension")
    new_far, old_far = _membrane(case), _membrane(parent)
    if new_far is not None and old_far is not None and new_far.get("grading") != old_far.get("grading"):
        if new_far.get("grading") is None:
            problems.append("a continuation may not remove physics.far_field.grading")
        elif not (_number(new_ext) and _number(old_ext) and new_ext > old_ext):
            problems.append("a continuation may change physics.far_field.grading only together "
                            "with a longer physics.far_field.extension")
    return problems, sorted(changed)


def _membrane(case: dict) -> dict | None:
    """The case's membrane far-field settings, or None for another far field."""
    physics = case.get("physics")
    far = physics.get("far_field") if isinstance(physics, dict) else None
    return far if isinstance(far, dict) and far.get("mode") == "membrane" else None


_MISSING = object()


def validate_case(case: object, problem: str | None = None) -> None:
    """Refuse an invalid case for ``problem`` (by default its own), naming every problem."""
    problems = case_problems(case, problem)
    if problems:
        raise SystemExit("refused: invalid case:\n  - " + "\n  - ".join(problems))


def stop_time(case: dict) -> float:
    """The declared stop time: the time value, or the event's horizon."""
    completion = case["completion"]
    return float(completion["value"] if completion["kind"] == "time" else completion["horizon"])


def driver_arguments(case: dict, out: Path, *, t_end: float | None = None,
                     far_field_extension: float | None = None,
                     newton_tolerance_safety: float | None = None) -> list[str]:
    """Map a valid case onto the driver's options; ``ValueError`` names every problem.

    ``t_end`` replaces the declared stop time, ``far_field_extension`` lengthens the
    membrane and ``newton_tolerance_safety`` sets the Newton acceptance as a multiple of the
    roundoff floor; all are runner options, so the run manifest records them in argv.
    """
    problems = case_problems(case, VISCOELASTIC)
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
        "--interval-start", str(solver.get("interval_start", "legacy")),
        "--initial-guess", str(solver.get("initial_guess", "previous")),
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
    if "temporal_error_fields" in solver:
        arguments += ["--temporal-error-fields", json.dumps(solver["temporal_error_fields"], sort_keys=True)]
    if "newton_floor_max_dt" in solver:
        arguments += ["--newton-floor-max-dt", str(solver["newton_floor_max_dt"])]
    if "relaxation_start_fraction" in solver:
        arguments += ["--relaxation-start-fraction", str(solver["relaxation_start_fraction"])]
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
                          ("node_spacing", "--far-field-node-spacing"),
                          ("interpolation", "--far-field-interpolation"),
                          ("history", "--far-field-history")):
            if key in far_field:
                arguments += [flag, str(far_field[key])]
        for key, flag in (("grading", "--far-field-grading"), ("auto_extend", "--far-field-auto-extend")):
            if key in far_field:
                arguments += [flag, json.dumps(far_field[key], sort_keys=True)]
    elif far_field is not None and far_field["mode"] == RELAXING_MEMBRANE:
        extension = far_field["extension"]
        if far_field_extension is not None:
            if far_field_extension < extension:
                raise ValueError("--far-field-extension may lengthen but never shorten "
                                 "physics.far_field.extension")
            extension = far_field_extension
        arguments += ["--far-field", RELAXING_MEMBRANE, "--far-field-extension", str(extension),
                      "--far-field-inner-fraction", str(far_field["inner_fraction"]),
                      "--far-field-node-spacing", str(far_field["node_spacing"])]
    elif far_field_extension is not None:
        raise ValueError("--far-field-extension requires a membrane far field")
    if newton_tolerance_safety is not None:
        arguments += ["--newton-tolerance-safety", str(float(newton_tolerance_safety))]
    if "max_positive_sponge_power" in tolerances:
        arguments += ["--max-positive-sponge-power", str(tolerances["max_positive_sponge_power"])]
    # A relaxing-membrane endpoint off the field stride is refused by case validation for the
    # declared stop and by the driver's restart-aware check for overrides and continuations.
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
