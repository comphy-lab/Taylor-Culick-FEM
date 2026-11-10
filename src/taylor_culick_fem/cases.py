"""Validate the fresh axisymmetric free-surface case contract."""
from __future__ import annotations

import math
import re


def case_problems(case: object) -> list[str]:
    """Return all detected envelope and supported-physics problems."""
    problems = []
    if not isinstance(case, dict):
        return ["case must be an object"]
    if case.get("schema") != "pyoomph-case-v1":
        problems.append("schema must be pyoomph-case-v1")
    if not isinstance(case.get("case_id"), str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", case["case_id"]):
        problems.append("case_id must be a lowercase portable identifier")
    if case.get("geometry") != {"kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "laboratory"}:
        problems.append("geometry must be axisymmetric laboratory coordinates [r,z]")
    scales = {"length_scale": "h0", "time_scale": "sqrt(rho*h0^3/sigma)",
              "velocity_scale": "sqrt(sigma/(rho*h0))", "Ec": "G_dim*h0/sigma",
              "De": "lambda_dim/sqrt(rho*h0^3/sigma)", "Oh": "mu_s_dim/sqrt(2*rho*sigma*h0)"}
    if case.get("nondimensionalization") != scales:
        problems.append("nondimensionalization must match the FEM capillary scales")
    sections = {}
    for key in ("physics", "mesh", "tolerances", "output", "solver", "arrest", "completion"):
        value = case.get(key)
        if not isinstance(value, dict):
            problems.append(f"{key} must be an object")
            value = {}
        sections[key] = value

    def number(section: str, name: str, *, nonnegative: bool = False, integer: bool = False):
        value = sections[section].get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (integer and not isinstance(value, int))
                or value < 0 or (not nonnegative and value == 0)):
            problems.append(f"{section}.{name} must be a {'non-negative' if nonnegative else 'positive'} {'integer' if integer else 'finite number'}")
            return None
        return value

    p = sections["physics"]
    number("physics", "Oh", nonnegative=True)
    modulus = number("physics", "elastic_modulus", nonnegative=True)
    relaxation = number("physics", "relaxation_time", nonnegative=True)
    number("physics", "supg_factor")
    if p.get("stabilization") not in ("none", "supg"):
        problems.append("physics.stabilization must be none or supg")
    if p.get("passive_conformation", False) is not False:
        problems.append("physics.passive_conformation must be false")
    if p.get("far_field", {"mode": "rest"}) != {"mode": "rest"} or "sponge" in p:
        problems.append("only the fully resolved rest far-field is supported")
    if modulus is not None and relaxation is not None:
        active = modulus > 0 and relaxation > 0
        if p.get("constitutive_model") != ("Oldroyd-B" if active else "Newtonian"):
            problems.append("physics.constitutive_model disagrees with modulus and relaxation")
        if not active and (modulus != 0 or relaxation != 0):
            problems.append("Newtonian cases require zero modulus and relaxation")
    length = number("mesh", "length")
    resolution = number("mesh", "resolution")
    for key in ("r0", "far_resolution"):
        number("mesh", key)
    if length is not None and resolution is not None and length <= resolution:
        problems.append("mesh.length must exceed mesh.resolution")
    for key in ("temporal_error", "max_volume_error", "max_tip_pin_residual"):
        number("tolerances", key)
    for key in ("min_radius_increase", "min_polymer_stress"):
        number("tolerances", key, nonnegative=True)
    if "max_positive_sponge_power" in sections["tolerances"]:
        number("tolerances", "max_positive_sponge_power", nonnegative=True)
    number("output", "cadence")
    stride = number("output", "field_stride", integer=True)
    checkpoint = number("output", "checkpoint_every", integer=True, nonnegative=True)
    if stride is not None and checkpoint is not None and checkpoint % stride:
        problems.append("output.checkpoint_every must be a multiple of output.field_stride")
    if sections["output"].get("write_states") is not False:
        problems.append("output.write_states must be false")
    number("solver", "max_step")
    number("solver", "max_newton", integer=True)
    for key in ("opening_ustar", "arrest_ustar", "radius_span_tolerance"):
        number("arrest", key)
    number("arrest", "trend_tolerance", nonnegative=True)
    consecutive = number("arrest", "consecutive_samples", integer=True)
    if consecutive is not None and consecutive < 2:
        problems.append("arrest.consecutive_samples must be at least two")
    if sections["completion"].get("kind") != "time":
        problems.append("completion.kind must be time")
    number("completion", "value")
    if type(sections["completion"].get("allow_recoil")) is not bool:
        problems.append("completion.allow_recoil must be boolean")
    if "restart" in case:
        problems.append("restarts are not supported by this fresh-run entry point")
    return problems


def validate_case(case: object) -> None:
    problems = case_problems(case)
    if problems:
        raise ValueError("; ".join(problems))
