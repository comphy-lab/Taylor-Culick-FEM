"""Run one bounded axisymmetric viscoelastic Taylor--Culick case.

Time is the physical capillary time used by the FEM problem.  This driver does
not use the viscous ``t*`` clock, which is undefined when the solvent
Ohnesorge number is zero.  ``lambda=inf`` selects the exact non-relaxing
Oldroyd-B/Kelvin--Voigt limit.

The modulus is non-dimensionalised as ``G = G_dim h0 / sigma`` using the full
sheet thickness, consistently with the inherited FEM problem.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyoomph.output.states import copy_state_file

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "simulationCases"))

from axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem
from planar_comoving import V_TAYLOR_CULICK

from taylor_culick_fem.absorbing_layer import RayleighSpongeParameters


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def _relaxation_time(text: str) -> float:
    if text.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return _positive_float(text)


def _parameter_label(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def _case_name(
    oh: float,
    elastic_modulus: float,
    relaxation_time: float,
    passive_conformation: bool = False,
) -> str:
    name = (
        f"axi_viscoelastic_Oh_{oh:g}_G_{elastic_modulus:g}_"
        f"lambda_{_parameter_label(relaxation_time)}"
    )
    return f"{name}_passive_conformation" if passive_conformation else name


def _passive_conformation(args: argparse.Namespace) -> bool:
    """Return the explicit conformation-feedback mode carried by CLI args."""

    value = getattr(args, "passive_conformation", False)
    if type(value) is not bool:
        raise ValueError("passive_conformation must be a boolean")
    return value


def _sponge_parameters(args: argparse.Namespace) -> RayleighSpongeParameters | None:
    values = (args.sponge_start, args.sponge_width, args.sponge_gamma_max)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("sponge start, width and gamma_max must be supplied together")
    parameters = RayleighSpongeParameters(
        start=args.sponge_start,
        width=args.sponge_width,
        gamma_max=args.sponge_gamma_max,
        profile=args.sponge_profile,
    )
    parameters.validate_window(args.length)
    return parameters


# Largest |u| tolerated in the outer tenth of the membrane extension.
FAR_FIELD_OUTER_TOLERANCE = 1e-6
FAR_FIELD_HEADER = [
    "t", "drive", "u_edge", "h_edge", "log_c_rr", "log_c_zz", "log_c_th", "outflow",
    "outer_velocity",
]


def _far_field_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the far-edge condition carried by CLI args."""

    mode = getattr(args, "far_field", "rest")
    tuning = {
        "inner_fraction": getattr(args, "far_field_inner_fraction", None),
        "extension": getattr(args, "far_field_extension", None),
        "node_spacing": getattr(args, "far_field_node_spacing", None),
    }
    if mode == "rest":
        if any(value is not None for value in tuning.values()):
            raise ValueError("far-field membrane settings require --far-field=membrane")
        return {"mode": "rest"}
    if mode != "membrane":
        raise ValueError(f"unknown far-field mode {mode!r}")
    if tuning["extension"] is None:
        raise ValueError("--far-field=membrane requires an explicit --far-field-extension")
    config: dict[str, object] = {
        "mode": "membrane",
        "inner_fraction": 0.5 if tuning["inner_fraction"] is None else float(tuning["inner_fraction"]),
        "extension": float(tuning["extension"]),
        "node_spacing": 0.5 if tuning["node_spacing"] is None else float(tuning["node_spacing"]),
    }
    if not 0 < config["inner_fraction"] < 1:
        raise ValueError("far-field inner fraction must lie strictly inside the window")
    if not config["extension"] > 0 or not config["node_spacing"] > 0:
        raise ValueError("far-field extension and node spacing must be positive")
    return config


def _without_extension(config: object) -> object:
    """Far-field config minus the membrane length, which a restart may grow."""

    if isinstance(config, dict) and config.get("mode") == "membrane":
        return {key: value for key, value in config.items() if key != "extension"}
    return config


def _check_extension_not_shorter(current: object, parent: object) -> None:
    if (
        isinstance(current, dict) and isinstance(parent, dict)
        and current.get("mode") == parent.get("mode") == "membrane"
        and current["extension"] < parent["extension"]
    ):
        raise ValueError("restart may lengthen but never shorten the far-field extension")


class MembraneCoupling:
    """Overlapping coupling of the 2D window with the far-field membrane.

    The membrane covers ``[X + a L, X0 + L + extension]``. Each output interval
    it is advanced first, driven at its inner end with the 2D midplane
    velocity there (linearly extrapolated from the last two samples), and its
    state at the coming edge radius ``X + L`` then fixes the 2D edge velocity,
    thickness and conformation for that interval. Waves leave the window
    through the membrane instead of reflecting off a rest condition; ahead of
    the pulse the membrane is at rest, so this reduces to ``"rest"`` there.
    ``outflow`` integrates ``(X+L)(h_e u_e - (h_e-1) Xdot)`` dt for the
    window-volume identity.
    """

    def __init__(self, membrane, *, inner_fraction: float, length: float) -> None:
        self.membrane = membrane
        self.inner_fraction = inner_fraction
        self.length = length
        self.previous: tuple[float, float] | None = None
        self.outflow = 0.0
        self.outer_velocity = 0.0

    def before_interval(self, time: float, target: float, radius: float, speed: float,
                        drive: float):
        interval = target - time
        if self.previous is None or not time > self.previous[0]:
            drive_end = drive
        else:
            slope = (drive - self.previous[1]) / (time - self.previous[0])
            drive_end = drive + slope * interval
        self.previous = (time, drive)
        predicted = radius + speed * interval
        self.membrane.advance(
            target, predicted + self.inner_fraction * self.length, drive, drive_end
        )
        tail = self.membrane.radii() >= self.membrane.outer - 0.1 * (
            self.membrane.outer - self.membrane.inner
        )
        self.outer_velocity = max(self.outer_velocity, float(np.max(np.abs(self.membrane.u[tail]))))
        return self.membrane.state_at(predicted + self.length)

    def after_interval(self, edge, time_end: float, outer_start: float, outer_end: float,
                       speed_start: float, speed_end: float, interval: float) -> None:
        h, u = edge.thickness, edge.velocity
        # The 2D solver lands a hair past its target; carry the membrane clock
        # with it so the next interval, and any restart, starts from one time.
        self.membrane.time = time_end
        self.outflow += 0.5 * interval * (
            outer_start * (h * u - (h - 1.0) * speed_start)
            + outer_end * (h * u - (h - 1.0) * speed_end)
        )

    def save(self, path: Path) -> None:
        previous = np.array(self.previous if self.previous is not None else (np.nan, np.nan))
        with path.open("xb") as stream:
            np.savez(stream, **self.membrane.to_arrays(), previous=previous,
                     outflow=np.array(self.outflow), outer_velocity=np.array(self.outer_velocity))

    def load(self, path: Path) -> None:
        from taylor_culick_fem.farfield_membrane import FarFieldMembrane

        with np.load(path) as data:
            self.membrane = FarFieldMembrane.from_arrays(data)
            previous = data["previous"]
            self.previous = None if np.isnan(previous).any() else (float(previous[0]), float(previous[1]))
            self.outflow = float(data["outflow"])
            self.outer_velocity = float(data["outer_velocity"])


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _assert_finite(label: str, values) -> None:
    array = np.asarray(list(values), dtype=float)
    if array.size and not np.all(np.isfinite(array)):
        raise RuntimeError(f"non-finite {label}: {array!r}")


def _file_reference(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def _checked_reference(reference: dict) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("restart references require exactly path and sha256")
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("restart reference must name an existing absolute file")
    if _file_reference(path)["sha256"] != reference["sha256"]:
        raise ValueError(f"restart reference hash mismatch: {path.name}")
    return path.resolve()


def _read_restart(args: argparse.Namespace) -> dict | None:
    """Validate immutable parent evidence before creating a continuation directory."""
    if args.restart_case is None:
        return None
    passive_conformation = _passive_conformation(args)
    serialized_case = getattr(args, "serialized_case", None)
    case = json.loads(
        serialized_case if serialized_case is not None else args.restart_case.read_bytes()
    )
    restart = case.get("restart")
    legacy_required = {
        "checkpoint", "input_case", "metadata", "control", "pvd", "time", "output_index"
    }
    decoupled_required = {
        "checkpoint", "input_case", "metadata", "control", "rim_height", "pvd", "time",
        "control_index", "field_index",
    }
    keys = set(restart) if isinstance(restart, dict) else set()
    if keys not in (legacy_required, decoupled_required):
        raise ValueError(
            "restart requires either legacy output_index lineage or decoupled "
            "control_index, field_index and rim_height lineage"
        )
    legacy = keys == legacy_required
    if legacy:
        raise ValueError(
            "legacy restart lineage lacks rim-height history; create a fresh "
            "decoupled checkpoint before continuation"
        )
    restart = dict(restart)
    paths = {
        key: _checked_reference(restart[key])
        for key in keys - {
            "time", "output_index", "control_index", "field_index", "input_case"
        }
    }
    root = paths["metadata"].parent
    snapshot = root / "input-case.json"
    if snapshot.is_file():
        restart["input_case"] = {**restart["input_case"], "path": str(snapshot)}
    paths["input_case"] = _checked_reference(restart["input_case"])
    if not paths["input_case"].is_relative_to(root):
        raise ValueError(
            "restart requires a verified parent input-case snapshot inside its case directory"
        )
    parent = json.loads(paths["input_case"].read_text())
    for key in (
        "schema",
        "geometry",
        "nondimensionalization",
        "physics",
        "mesh",
        "tolerances",
        "output",
        "solver",
        "arrest",
    ):
        current = case.get(key)
        expected = parent.get(key)
        if key == "physics" and isinstance(current, dict) and isinstance(expected, dict):
            # A missing mode predates this switch and means the backwards-
            # compatible active default. Treat it as equivalent to false.
            current = {
                **current,
                "passive_conformation": current.get("passive_conformation", False),
            }
            expected = {
                **expected,
                "passive_conformation": expected.get("passive_conformation", False),
            }
            # A missing far field predates the far-field modes and means "rest".
            current["far_field"] = current.get("far_field", {"mode": "rest"})
            expected["far_field"] = expected.get("far_field", {"mode": "rest"})
            _check_extension_not_shorter(current["far_field"], expected["far_field"])
            current["far_field"] = _without_extension(current["far_field"])
            expected["far_field"] = _without_extension(expected["far_field"])
        if key not in case or current != expected:
            raise ValueError(f"restart case mismatch: {key}")
    metadata = json.loads(paths["metadata"].read_text())
    identity = {
        "Oh": args.oh,
        "elastic_modulus": args.elastic_modulus,
        "relaxation_time": "inf" if math.isinf(args.relaxation_time) else args.relaxation_time,
        "passive_conformation": passive_conformation,
        "far_field": _far_field_config(args),
        "length": args.length,
        "r0": args.r0,
        "resolution": args.resolution,
        "far_resolution": args.far_resolution,
        "output_step": args.output_step,
        "field_stride": args.field_stride,
        "max_step": args.max_step,
        "stabilization_option": args.stabilization,
        "conformation_supg_factor": args.supg_factor,
        "temporal_error": args.temporal_error,
        "sponge": (
            _sponge_parameters(args).as_json_dict()
            if _sponge_parameters(args) is not None
            else None
        ),
    }
    for key, value in identity.items():
        observed = metadata.get(key)
        if key == "field_stride" and observed is None:
            observed = 1
        if key == "conformation_supg_factor" and observed is None:
            # Attempts written before this control was exposed used the
            # problem class's factor-one default.
            observed = 1.0
        if key == "passive_conformation" and key not in metadata:
            # Attempts written before passive conformation was exposed were
            # active by construction.
            observed = False
        if key == "far_field" and key not in metadata:
            observed = {"mode": "rest"}
        if key == "far_field":
            _check_extension_not_shorter(value, observed)
            value, observed = _without_extension(value), _without_extension(observed)
        if observed != value:
            raise ValueError(f"restart metadata mismatch: {key}")
    control_index = restart["output_index"] if legacy else restart["control_index"]
    field_index = restart["output_index"] if legacy else restart["field_index"]
    time = restart["time"]
    if type(control_index) is not int or control_index <= 0:
        label = "output_index" if legacy else "control index"
        raise ValueError(f"restart {label} must be a positive integer")
    if type(field_index) is not int or field_index < 0 or field_index > control_index:
        raise ValueError("restart field index must be a non-negative integer at most control_index")
    if not isinstance(time, (int, float)) or not math.isfinite(time) or not 0 < time < args.t_end:
        raise ValueError("restart time must be finite, positive and before t_end")
    if args.output_step is None:
        raise ValueError("restart requires a fixed output cadence")
    rows = np.loadtxt(paths["control"], ndmin=2)
    if rows.shape[1] != 8 or len(rows) < control_index or not np.all(np.isfinite(rows)):
        raise ValueError("restart control history is incomplete or invalid")
    if np.any(np.diff(rows[:, 0]) <= 0) or rows[0, 0] <= 0:
        raise ValueError("restart control times must be strictly increasing and positive")
    rows = rows[:control_index]
    if not math.isclose(rows[-1, 0], time, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("restart time does not match selected control row")
    # Output times have a small solver landing offset; reject off-grid checkpoints
    # rather than silently changing the output interval after a short final step.
    if not np.allclose(
        rows[:, 0], args.output_step * np.arange(1, control_index + 1), rtol=0, atol=1e-6
    ):
        raise ValueError("restart control history does not match the output cadence")
    datasets = ET.parse(paths["pvd"]).findall("./Collection/DataSet")
    if len(datasets) < field_index:
        raise ValueError("restart PVD history is incomplete")
    expected_field_times = rows[args.field_stride - 1 :: args.field_stride, 0]
    if len(expected_field_times) != field_index:
        raise ValueError("restart field index disagrees with deterministic field stride")
    fields = []
    for expected_field_time, dataset in zip(
        expected_field_times, datasets[:field_index], strict=True
    ):
        field_time = float(dataset.attrib["timestep"])
        if not math.isclose(
            expected_field_time, field_time, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("restart PVD and control times disagree")
        path = (paths["pvd"].parent / dataset.attrib["file"]).resolve()
        if not path.is_file():
            raise ValueError("restart PVD refers to a missing field")
        fields.append((field_time, path))
    height_rows = None
    if not legacy:
        height_rows = np.loadtxt(paths["rim_height"], ndmin=2)
        if (
            height_rows.shape[1] != 2
            or len(height_rows) < control_index
            or not np.all(np.isfinite(height_rows))
            or np.any(height_rows[:, 1] <= 0)
        ):
            raise ValueError("restart rim-height history is incomplete or invalid")
        height_rows = height_rows[:control_index]
        if not np.allclose(height_rows[:, 0], rows[:, 0], rtol=1e-10, atol=1e-10):
            raise ValueError("restart rim-height and control times disagree")
    far_rows = None
    far_state = None
    if _far_field_config(args)["mode"] == "membrane":
        far_path = paths["control"].parent / "far-field.txt"
        width = len(FAR_FIELD_HEADER)
        far_rows = np.loadtxt(far_path, ndmin=2) if far_path.is_file() else np.empty((0, width))
        if (
            far_rows.shape[1] != width
            or len(far_rows) < control_index
            or not np.all(np.isfinite(far_rows))
        ):
            raise ValueError("restart far-field history is incomplete or invalid")
        far_rows = far_rows[:control_index]
        if not np.allclose(far_rows[:, 0], rows[:, 0], rtol=1e-10, atol=1e-10):
            raise ValueError("restart far-field and control times disagree")
        far_state = paths["checkpoint"].with_suffix(".farfield.npz")
        if not far_state.is_file():
            raise ValueError("restart checkpoint has no far-field membrane state")
    if paths["checkpoint"].parent != root / "checkpoints":
        raise ValueError("restart checkpoint does not belong to the parent case")
    if paths["control"].parent != root / "run_viscoelastic_axi":
        raise ValueError("restart control does not belong to the parent case")
    if paths["pvd"].parent != paths["control"].parent:
        raise ValueError("restart PVD does not belong to the parent case")
    if not legacy and paths["rim_height"].parent != paths["control"].parent:
        raise ValueError("restart rim-height history does not belong to the parent case")
    return {
        "evidence": restart,
        "paths": paths,
        "metadata": metadata,
        "rows": rows,
        "height_rows": height_rows,
        "far_rows": far_rows,
        "far_state": far_state,
        "fields": fields,
        "time": float(time),
        "control_index": control_index,
        "field_index": field_index,
        "output_index": control_index,
        "legacy": legacy,
    }


def _write_field_history(output_dir: Path, parent_fields: list) -> None:
    """Join a read-only parent prefix and this attempt's fresh field segment."""
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")
    fields = list(parent_fields)
    pvd = output_dir / "liquid.pvd"
    if pvd.exists():
        for dataset in ET.parse(pvd).findall("./Collection/DataSet"):
            fields.append(
                (float(dataset.attrib["timestep"]), (output_dir / dataset.attrib["file"]).resolve())
            )
    previous = -math.inf
    for time, path in fields:
        if time <= previous:
            raise ValueError("field history contains nonmonotone or duplicate times")
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(time),
            part="0",
            file=os.path.relpath(path, output_dir),
        )
        previous = time
    target = output_dir / "liquid-history.pvd"
    temporary = target.with_suffix(".pvd.tmp")
    ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(target)


def _output_times(args: argparse.Namespace) -> np.ndarray:
    if args.output_step is None:
        return np.linspace(args.t_end / args.n_outputs, args.t_end, args.n_outputs)
    count = math.floor(args.t_end / args.output_step)
    times = args.output_step * np.arange(1, count + 1, dtype=float)
    if not len(times) or times[-1] < args.t_end * (1 - 1e-14):
        times = np.append(times, args.t_end)
    return times


def _zero_crossing(previous: list[float], current: list[float]) -> dict[str, float]:
    """Linearly interpolate the sampled bracket in which ``U`` reaches zero."""

    t0, r0, u0 = previous[:3]
    t1, r1, u1 = current[:3]
    fraction = u0 / (u0 - u1)
    return {
        "time": t0 + fraction * (t1 - t0),
        "radius": r0 + fraction * (r1 - r0),
        "bracket_start": t0,
        "bracket_end": t1,
        "radius_bracket_start": r0,
        "radius_bracket_end": r1,
        "time_sampling_uncertainty": 0.5 * (t1 - t0),
        "radius_sampling_uncertainty": 0.5 * abs(r1 - r0),
    }


@dataclass
class ArrestTracker:
    """Detect a post-startup turning point or threshold-certified arrest."""

    opening_ustar: float
    minimum_radius_increase: float
    arrest_ustar: float
    consecutive_samples: int
    trend_tolerance: float
    radius_span_tolerance: float
    opening_established: bool = False
    previous_row: list[float] | None = None
    near_zero_rows: list[list[float]] = field(default_factory=list)

    def observe(self, row: list[float], radius_initial: float) -> dict[str, float | str] | None:
        time, radius, speed, ustar = row[:4]
        if ustar >= self.opening_ustar and radius - radius_initial >= self.minimum_radius_increase:
            self.opening_established = True

        result: dict[str, float | str] | None = None
        if self.opening_established:
            if self.previous_row is not None and self.previous_row[2] > 0 >= speed:
                result = {"kind": "sampled_zero_crossing", **_zero_crossing(self.previous_row, row)}
            elif 0 <= ustar <= self.arrest_ustar:
                self.near_zero_rows.append(row)
                self.near_zero_rows = self.near_zero_rows[-self.consecutive_samples :]
                if len(self.near_zero_rows) == self.consecutive_samples:
                    times = np.asarray([sample[0] for sample in self.near_zero_rows])
                    radii = np.asarray([sample[1] for sample in self.near_zero_rows])
                    speeds = np.asarray([sample[3] for sample in self.near_zero_rows])
                    slope = float(np.polyfit(times, speeds, 1)[0])
                    radius_span = float(np.ptp(radii))
                    if (
                        abs(slope) <= self.trend_tolerance
                        and radius_span <= self.radius_span_tolerance
                    ):
                        result = {
                            "kind": "threshold_certified_asymptotic",
                            "time": time,
                            "radius": float(np.mean(radii)),
                            "bracket_start": float(times[0]),
                            "bracket_end": float(times[-1]),
                            "radius_bracket_start": float(np.min(radii)),
                            "radius_bracket_end": float(np.max(radii)),
                            "time_sampling_uncertainty": 0.5 * float(times[-1] - times[-2]),
                            "radius_sampling_uncertainty": 0.5 * radius_span,
                            "ustar_trend": slope,
                        }
            else:
                self.near_zero_rows.clear()

        self.previous_row = row
        return result


@dataclass
class HeightPlateauTracker:
    """Certify a sustained plateau in the full rim height ``H``.

    A candidate must pass the campaign's trend, residual, adjacent-window and
    residual-envelope predicates continuously for ``hold`` physical-time
    units.  The tracker deliberately uses no inferred oscillation period; a
    period only enters when supplied explicitly.
    """

    hold: float = 10.0
    minimum_window: float = 10.0
    window_time_fraction: float = 0.1
    slope_window_fraction: float = 0.005
    instantaneous_slope_fraction: float = 0.02
    residual_rms_fraction: float = 0.005
    adjacent_mean_fraction: float = 0.005
    residual_envelope_growth_fraction: float = 0.05
    oscillation_period: float | None = None
    rows: list[tuple[float, float]] = field(default_factory=list)
    sum_time: list[float] = field(default_factory=lambda: [0.0])
    sum_height: list[float] = field(default_factory=lambda: [0.0])
    sum_time_squared: list[float] = field(default_factory=lambda: [0.0])
    sum_time_height: list[float] = field(default_factory=lambda: [0.0])
    sum_height_squared: list[float] = field(default_factory=lambda: [0.0])
    candidate_since: float | None = None
    latest_metrics: dict[str, float | bool] | None = None

    def _window(self, time: float) -> float:
        terms = [self.minimum_window, self.window_time_fraction * time]
        if self.oscillation_period is not None:
            terms.append(3.0 * self.oscillation_period)
        return max(terms)

    def _statistics(self, start: int, stop: int) -> tuple[float, float, float]:
        """Return mean, least-squares slope and residual RMS in O(1)."""

        count = stop - start
        sum_time = self.sum_time[stop] - self.sum_time[start]
        sum_height = self.sum_height[stop] - self.sum_height[start]
        sum_time_squared = self.sum_time_squared[stop] - self.sum_time_squared[start]
        sum_time_height = self.sum_time_height[stop] - self.sum_time_height[start]
        sum_height_squared = self.sum_height_squared[stop] - self.sum_height_squared[start]
        mean = sum_height / count
        centred_time_squared = sum_time_squared - sum_time**2 / count
        centred_cross = sum_time_height - sum_time * sum_height / count
        centred_height_squared = max(sum_height_squared - sum_height**2 / count, 0.0)
        slope = centred_cross / centred_time_squared
        residual_squared = max(centred_height_squared - slope * centred_cross, 0.0)
        return mean, slope, math.sqrt(residual_squared / count)

    def observe(self, time: float, height: float) -> dict[str, object] | None:
        if not math.isfinite(time) or not math.isfinite(height) or height <= 0:
            raise ValueError("height plateau samples must have finite time and positive height")
        if self.rows and time <= self.rows[-1][0]:
            raise ValueError("height plateau times must be strictly increasing")
        self.rows.append((time, height))
        self.sum_time.append(self.sum_time[-1] + time)
        self.sum_height.append(self.sum_height[-1] + height)
        self.sum_time_squared.append(self.sum_time_squared[-1] + time**2)
        self.sum_time_height.append(self.sum_time_height[-1] + time * height)
        self.sum_height_squared.append(self.sum_height_squared[-1] + height**2)
        window = self._window(time)
        time_key = lambda row: row[0]
        previous_start = bisect.bisect_left(
            self.rows, time - 2 * window, key=time_key
        )
        current_start = bisect.bisect_left(self.rows, time - window, key=time_key)
        stop = len(self.rows)
        previous_count = current_start - previous_start
        current_count = stop - current_start
        if (
            previous_count < 2
            or current_count < 4
            or self.rows[current_start - 1][0] - self.rows[previous_start][0] < 0.8 * window
            or self.rows[-1][0] - self.rows[current_start][0] < 0.8 * window
        ):
            self.candidate_since = None
            self.latest_metrics = None
            return None

        mean, slope, residual_rms = self._statistics(current_start, stop)
        previous_mean, _previous_slope, previous_rms = self._statistics(
            previous_start, current_start
        )
        envelope_limit = previous_rms * (1.0 + self.residual_envelope_growth_fraction)
        # Compare like-sized adjacent windows.  When an explicit period is
        # supplied each contains at least three periods, avoiding a fragile
        # half-cycle or spectral estimate.
        envelope_ok = bool(
            residual_rms <= envelope_limit + 1e-7 * mean
        )
        metrics: dict[str, float | bool] = {
            "time": time,
            "window": window,
            "height_mean": mean,
            "slope": float(slope),
            "slope_window_fraction": abs(float(slope)) * window / mean,
            "instantaneous_slope_fraction": time * abs(float(slope)) / mean,
            "residual_rms_fraction": residual_rms / mean,
            "adjacent_mean_fraction": abs(mean - previous_mean) / mean,
            "residual_previous_window_rms": previous_rms,
            "residual_current_window_rms": residual_rms,
            "residual_envelope_non_growing": envelope_ok,
        }
        passed = bool(
            metrics["slope_window_fraction"] < self.slope_window_fraction
            and metrics["instantaneous_slope_fraction"] < self.instantaneous_slope_fraction
            and metrics["residual_rms_fraction"] < self.residual_rms_fraction
            and metrics["adjacent_mean_fraction"] < self.adjacent_mean_fraction
            and envelope_ok
        )
        metrics["candidate"] = passed
        self.latest_metrics = metrics
        if not passed:
            self.candidate_since = None
            return None
        if self.candidate_since is None:
            self.candidate_since = time
        held_for = time - self.candidate_since
        if held_for + 1e-12 < self.hold:
            return None
        return {
            "kind": "height_plateau",
            "candidate_since": self.candidate_since,
            "certified_at": time,
            "hold": self.hold,
            "held_for": held_for,
            "oscillation_period": self.oscillation_period,
            "thresholds": {
                "slope_window_fraction": self.slope_window_fraction,
                "instantaneous_slope_fraction": self.instantaneous_slope_fraction,
                "residual_rms_fraction": self.residual_rms_fraction,
                "adjacent_mean_fraction": self.adjacent_mean_fraction,
                "residual_envelope_growth_fraction": self.residual_envelope_growth_fraction,
            },
            "metrics": metrics,
        }


def _enforce_launcher_affinity() -> dict[str, object]:
    """Confine auxiliary threads to the launcher's Linux CPU affinity.

    MPI initialisation creates an auxiliary ``fuse`` thread which widens its
    own affinity after inheriting the launcher's mask.  On Linux, restore every
    extant thread to the main thread's mask before any FEM work begins.  On
    other platforms this is a reported no-op; their launchers own affinity.
    """

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"supported": False, "repaired_threads": 0}
    allowed = os.sched_getaffinity(0)
    task_root = Path("/proc/self/task")
    if not task_root.is_dir():
        return {"supported": False, "repaired_threads": 0}
    repaired = 0
    checked = 0
    for task in task_root.iterdir():
        try:
            tid = int(task.name)
            checked += 1
            if os.sched_getaffinity(tid) != allowed:
                os.sched_setaffinity(tid, allowed)
                if os.sched_getaffinity(tid) != allowed:
                    raise RuntimeError(f"thread {tid} rejected the launcher CPU mask")
                repaired += 1
        except ProcessLookupError:
            # A short-lived library thread exited between enumeration and the
            # affinity calls; it cannot escape the later computation.
            continue
    return {
        "supported": True,
        "allowed_cpus": sorted(allowed),
        "checked_threads": checked,
        "repaired_threads": repaired,
    }


def run_case(args: argparse.Namespace) -> dict:
    passive_conformation = _passive_conformation(args)
    if args.elastic_modulus == 0 and not passive_conformation:
        raise ValueError("elastic_modulus=0 requires passive_conformation=True")
    restart = _read_restart(args)
    sponge = _sponge_parameters(args)
    far_field = _far_field_config(args)
    affinity = _enforce_launcher_affinity()
    run_dir = args.results.resolve() / _case_name(
        args.oh, args.elastic_modulus, args.relaxation_time, passive_conformation
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    relaxation_json: float | str = (
        "inf" if math.isinf(args.relaxation_time) else args.relaxation_time
    )
    metadata = {
        "case": "axisymmetric_viscoelastic_comoving",
        "Oh": args.oh,
        "mu_s": math.sqrt(2.0) * args.oh,
        "elastic_modulus": args.elastic_modulus,
        "elastic_modulus_definition": "G_dim*h0/sigma",
        "passive_conformation": passive_conformation,
        "relaxation_time": relaxation_json,
        "constitutive_model": "Oldroyd-B",
        "polymer_stress": "G*(C-I)",
        "conformation_formulation": "log-conf",
        "stabilization_option": args.stabilization,
        "conformation_stabilization": "SUPG" if args.stabilization == "supg" else None,
        "conformation_supg_factor": args.supg_factor,
        "coordinate_system": "axisymmetric",
        "velocity_frame": "laboratory",
        "time_scale": "capillary",
        "t_end": args.t_end,
        "resolution": args.resolution,
        "far_resolution": args.far_resolution,
        "length": args.length,
        "r0": args.r0,
        "n_outputs": args.n_outputs,
        "output_step": args.output_step,
        "field_stride": args.field_stride,
        "stop_when_arrested": args.stop_when_arrested,
        "stop_on_height_plateau": args.stop_on_height_plateau,
        "allow_recoil": args.allow_recoil,
        "sponge": sponge.as_json_dict() if sponge is not None else None,
        "sponge_force": "-rho*gamma(x_rel)*u_lab" if sponge is not None else None,
        "far_field": far_field,
        "arrest_ustar_threshold": args.arrest_ustar,
        "arrest_consecutive_samples": args.arrest_consecutive,
        "arrest_trend_tolerance": args.arrest_trend_tolerance,
        "arrest_radius_span_tolerance": args.arrest_radius_span,
        "opening_ustar_threshold": args.opening_ustar,
        "checkpoint_every_outputs": (
            args.checkpoint_every if args.field_stride == 1 else None
        ),
        "checkpoint_every_scalar_samples": args.checkpoint_every,
        "fixed_step": args.fixed_step,
        "fixed_step_semantics": "maximum-step cap after smaller BDF startup",
        "max_step": args.max_step,
        "temporal_error": None if args.fixed_step is not None else args.temporal_error,
        "write_states": False,
        "checkpoint_policy": (
            f"every {args.checkpoint_every} scalar samples" if args.checkpoint_every else "none"
        ),
        "height_plateau": {
            "hold": args.height_hold,
            "minimum_window": args.height_minimum_window,
            "window_time_fraction": args.height_window_time_fraction,
            "slope_window_fraction": args.height_slope_window_fraction,
            "instantaneous_slope_fraction": args.height_instantaneous_slope_fraction,
            "residual_rms_fraction": args.height_residual_rms_fraction,
            "adjacent_mean_fraction": args.height_adjacent_mean_fraction,
            "residual_envelope_growth_fraction": args.height_envelope_growth_fraction,
            "oscillation_period": args.height_oscillation_period,
        } if args.stop_on_height_plateau else None,
        "V_TC": V_TAYLOR_CULICK,
        "launcher_affinity": affinity,
        "status": "initialising",
    }
    serialized_case = getattr(args, "serialized_case", None)
    if serialized_case is not None:
        snapshot = run_dir / "input-case.json"
        with snapshot.open("xb") as stream:
            stream.write(serialized_case)
            stream.flush()
            os.fsync(stream.fileno())
        metadata["input_case"] = {**_file_reference(snapshot), "path": snapshot.name}
    _write_json(run_dir / "case.json", metadata)
    if restart is not None:
        lineage = {
            key: {**value, "path": os.path.relpath(value["path"], run_dir)}
            if isinstance(value, dict)
            else value
            for key, value in restart["evidence"].items()
        }
        _write_json(
            run_dir / "restart-lineage.json",
            {
                "schema": "viscoelastic-restart-lineage-v2",
                "parent": lineage,
                "authoritative_segment": "run_viscoelastic_axi/liquid.pvd",
                "authoritative_control_segment": "run_viscoelastic_axi/control-segment.txt",
                "authoritative_rim_height_segment": "run_viscoelastic_axi/rim-height-segment.txt",
                "derived_views": [
                    "run_viscoelastic_axi/control.txt",
                    "run_viscoelastic_axi/rim-height.txt",
                    "run_viscoelastic_axi/liquid-history.pvd",
                ],
                "archive_contract": "Retain all ancestor attempts with relative directory layout; derived views require ancestor fields.",
            },
        )

    previous_cwd = Path.cwd()
    try:
        with AxisymmetricViscoelasticProblem() as problem:
            problem.set_output_directory(str(run_dir / "run_viscoelastic_axi"))
            problem.solvent_oh = args.oh
            problem.elastic_modulus = args.elastic_modulus
            problem.relaxation_time = args.relaxation_time
            problem.passive_conformation = passive_conformation
            problem.sponge = sponge
            problem.far_field = far_field["mode"]
            problem.conformation_stabilization = "SUPG" if args.stabilization == "supg" else None
            problem.conformation_supg_factor = args.supg_factor
            problem.X0 = args.r0
            problem.window_length = args.length
            problem.resolution = args.resolution
            problem.far_resolution = args.far_resolution
            problem.max_newton_iterations = args.max_newton
            problem.initialise()
            # Preserve the original ALE reference before restoring the deformed state.
            volume0 = problem.volume()
            radius0 = problem.window_position()
            coupling = None
            if far_field["mode"] == "membrane":
                from taylor_culick_fem.farfield_membrane import FarFieldMembrane

                inner = radius0 + far_field["inner_fraction"] * args.length
                outer = radius0 + args.length + far_field["extension"]
                coupling = MembraneCoupling(
                    FarFieldMembrane(
                        elastic_modulus=args.elastic_modulus,
                        viscosity=math.sqrt(2.0) * args.oh,
                        inner=inner,
                        outer=outer,
                        nodes=math.ceil((outer - inner) / far_field["node_spacing"]) + 1,
                    ),
                    inner_fraction=far_field["inner_fraction"],
                    length=args.length,
                )
                if restart is not None:
                    coupling.load(restart["far_state"])
                    if not math.isclose(
                        coupling.membrane.time, restart["time"], rel_tol=1e-10, abs_tol=1e-10
                    ):
                        raise ValueError("far-field membrane state time disagrees with lineage")
                    metadata["restart_far_field_state"] = _file_reference(restart["far_state"])
                    if outer > coupling.membrane.outer + 1e-9:
                        metadata["restart_far_field_extended"] = {
                            "from_outer": coupling.membrane.outer, "to_outer": outer,
                        }
                        coupling.membrane = coupling.membrane.extended(
                            outer, far_field["node_spacing"]
                        )
            if restart is not None:
                checkpoint = restart["paths"]["checkpoint"]
                state_time, state_index = problem._get_time_of_state_file(str(checkpoint))
                if (
                    not math.isclose(state_time, restart["time"], rel_tol=1e-10, abs_tol=1e-10)
                    or state_index != restart["field_index"]
                ):
                    raise ValueError("checkpoint header time/field index disagrees with lineage")
                relocated = run_dir / "restart-input" / checkpoint.name
                relocated.parent.mkdir()
                copy_state_file(str(checkpoint), str(relocated))
                # Relocation rewrites embedded mesh paths, so byte equality is
                # informative, not required. The source itself must stay intact.
                source_reference = _file_reference(checkpoint)
                if source_reference["sha256"] != restart["evidence"]["checkpoint"]["sha256"]:
                    raise ValueError("checkpoint source hash changed during relocation")
                relocated_reference = _file_reference(relocated)
                metadata["checkpoint_relocation"] = {
                    "source_sha256": source_reference["sha256"],
                    "relocated_sha256": relocated_reference["sha256"],
                    "byte_identical": source_reference["sha256"] == relocated_reference["sha256"],
                }
                problem.load_state(str(relocated), ignore_outstep=False)
                if not math.isclose(
                    float(problem.get_current_time(as_float=True)),
                    restart["time"],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise ValueError("loaded checkpoint time disagrees with lineage")
                row = restart["rows"][-1]
                if not np.allclose(
                    [problem.window_position(), problem.retraction_speed(), problem.volume()],
                    row[[1, 2, 4]],
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise ValueError("checkpoint state disagrees with parent control row")
                # Carry the parent's ALE reference exactly, including older attempts
                # which did not store volume0 explicitly in case.json.
                inferred_volume0 = float(
                    row[5]
                    - math.pi / 2 * ((row[1] + args.length) ** 2 - (radius0 + args.length) ** 2)
                    + math.pi * (coupling.outflow if coupling is not None else 0.0)
                )
                volume0 = restart["metadata"].get("volume_initial_reference", inferred_volume0)
                if not math.isfinite(volume0) or not math.isclose(
                    volume0, inferred_volume0, rel_tol=1e-12, abs_tol=1e-10
                ):
                    raise ValueError("restart ALE reference disagrees with parent control row")
                metadata["restart_time"] = restart["time"]
                metadata["restart_control_index"] = restart["control_index"]
                metadata["restart_field_index"] = restart["field_index"]
                if restart["legacy"]:
                    metadata["restart_output_index"] = restart["output_index"]
                metadata["relocated_checkpoint"] = {
                    **relocated_reference,
                    "path": os.path.relpath(relocated, run_dir),
                }
            # Per-internal-step states are prohibitively wasteful. A longer
            # arrest run may retain explicit output-indexed checkpoints.
            problem.write_states = False
            checkpoint_dir = run_dir / "checkpoints"
            if args.checkpoint_every:
                checkpoint_dir.mkdir()

            metadata["volume_initial_reference"] = volume0
            history = problem.create_text_file_output(
                "control.txt",
                header=[
                    "t",
                    "R",
                    "U",
                    "ustar",
                    "volume",
                    "volume_expected",
                    "volume_rel_error",
                    "tip_pin_residual",
                ],
            )
            segment = problem.create_text_file_output(
                "control-segment.txt",
                header=[
                    "t",
                    "R",
                    "U",
                    "ustar",
                    "volume",
                    "volume_expected",
                    "volume_rel_error",
                    "tip_pin_residual",
                ],
            )
            height_history = problem.create_text_file_output(
                "rim-height.txt", header=["t", "H"]
            )
            height_segment = problem.create_text_file_output(
                "rim-height-segment.txt", header=["t", "H"]
            )
            far_history = far_segment = None
            if coupling is not None:
                far_history = problem.create_text_file_output(
                    "far-field.txt", header=FAR_FIELD_HEADER
                )
                far_segment = problem.create_text_file_output(
                    "far-field-segment.txt", header=FAR_FIELD_HEADER
                )
            sponge_history = None
            if sponge is not None and sponge.gamma_max > 0:
                sponge_history = problem.create_text_file_output(
                    "sponge-diagnostics.txt",
                    header=["t", "power", "core_velocity_rms"],
                )
            out_times = _output_times(args)
            previous_target = 0.0
            worst_volume_error = 0.0
            worst_tip_pin = 0.0
            minimum_sampled_speed = math.inf
            last_row: list[float] | None = None
            diagnostics: dict[str, float] = {}
            sponge_diagnostics: dict[str, float] = {}
            largest_positive_sponge_power = (
                float(restart["metadata"].get("largest_positive_sponge_power", 0.0))
                if restart is not None
                else 0.0
            )
            _assert_finite(
                "restart largest positive sponge power",
                [largest_positive_sponge_power],
            )
            arrest: dict[str, float | str] | None = None
            arrest_tracker = ArrestTracker(
                opening_ustar=args.opening_ustar,
                minimum_radius_increase=args.min_radius_increase,
                arrest_ustar=args.arrest_ustar,
                consecutive_samples=args.arrest_consecutive,
                trend_tolerance=args.arrest_trend_tolerance,
                radius_span_tolerance=args.arrest_radius_span,
            )
            height_plateau: dict[str, object] | None = None
            height_tracker = HeightPlateauTracker(
                hold=args.height_hold,
                minimum_window=args.height_minimum_window,
                window_time_fraction=args.height_window_time_fraction,
                slope_window_fraction=args.height_slope_window_fraction,
                instantaneous_slope_fraction=args.height_instantaneous_slope_fraction,
                residual_rms_fraction=args.height_residual_rms_fraction,
                adjacent_mean_fraction=args.height_adjacent_mean_fraction,
                residual_envelope_growth_fraction=args.height_envelope_growth_fraction,
                oscillation_period=args.height_oscillation_period,
            )
            control_offset = 0
            field_index = 0
            if restart is not None:
                for row in restart["rows"]:
                    history.add_row(*row)
                    arrest_tracker.observe(list(row), radius0)
                if restart["height_rows"] is not None:
                    for time, height in restart["height_rows"]:
                        height_history.add_row(time, height)
                        height_tracker.observe(float(time), float(height))
                if far_history is not None:
                    for far_row in restart["far_rows"]:
                        far_history.add_row(*far_row)
                control_offset = restart["control_index"]
                field_index = restart["field_index"]
                previous_target = restart["time"]
                out_times = out_times[out_times > restart["time"] + 1e-6]
                worst_volume_error = float(np.max(restart["rows"][:, 6]))
                worst_tip_pin = float(np.max(restart["rows"][:, 7]))
                minimum_sampled_speed = float(np.min(restart["rows"][:, 2]))
                if not len(out_times):
                    raise ValueError("restart has no output targets before the requested endpoint")

            for control_index, target in enumerate(out_times, start=control_offset + 1):
                interval = float(target - previous_target)
                if args.fixed_step is not None:
                    # This is a maximum-step cap, not a promise that every
                    # internal step is identical: BDF starts smaller to build
                    # its history, and output landing can add a short sliver.
                    maxstep = min(args.fixed_step, interval)
                    startstep = min(maxstep, max(interval * 1e-2, 1e-6))
                else:
                    maxstep = (
                        min(args.max_step, interval) if args.max_step is not None else interval
                    )
                    startstep = min(maxstep, max(interval * 1e-2, 1e-6))
                if coupling is not None:
                    # Integrate over the solver's own clock, not the nominal
                    # targets, so a restart lands on the same outflow history.
                    start_now = float(problem.get_current_time(as_float=True))
                    radius_start = problem.window_position()
                    speed_start = problem.retraction_speed()
                    drive = problem.midplane_radial_velocity(
                        radius_start + far_field["inner_fraction"] * args.length
                    )
                    edge = coupling.before_interval(
                        start_now, float(target), radius_start, speed_start, drive
                    )
                    problem.set_far_field(edge.velocity, edge.thickness, edge.log_conformation())
                problem.run(
                    float(target),
                    startstep=startstep,
                    maxstep=maxstep,
                    temporal_error=None if args.fixed_step is not None else args.temporal_error,
                    outstep=False,
                )
                now = float(problem.get_current_time(as_float=True))
                radius = problem.window_position()
                speed = problem.retraction_speed()
                volume = problem.volume()
                if coupling is not None:
                    coupling.after_interval(
                        edge,
                        now,
                        radius_start + args.length,
                        radius + args.length,
                        speed_start,
                        speed,
                        now - start_now,
                    )
                    far_row = [
                        now,
                        drive,
                        edge.velocity,
                        edge.thickness,
                        *edge.log_conformation(),
                        coupling.outflow,
                        coupling.outer_velocity,
                    ]
                    _assert_finite("far-field row", far_row)
                    far_history.add_row(*far_row)
                    far_segment.add_row(*far_row)
                expected = (
                    problem.expected_volume(volume0, radius0)
                    if coupling is None
                    else problem.expected_volume(volume0, radius0, far_outflow=coupling.outflow)
                )
                volume_error = abs(volume - expected) / max(abs(expected), 1e-300)
                tip_pin = abs(radius - problem.hole_radius())
                last_row = [
                    now,
                    radius,
                    speed,
                    speed / V_TAYLOR_CULICK,
                    volume,
                    expected,
                    volume_error,
                    tip_pin,
                ]
                _assert_finite("control row", last_row)
                history.add_row(*last_row)
                segment.add_row(*last_row)
                height = problem.rim_height()
                _assert_finite("rim height", [height])
                height_history.add_row(now, height)
                height_segment.add_row(now, height)
                diagnostics = problem.polymer_diagnostic_extrema()
                _assert_finite("polymer diagnostics", diagnostics.values())
                if sponge_history is not None:
                    sponge_diagnostics = problem.sponge_diagnostics()
                    _assert_finite("sponge diagnostics", sponge_diagnostics.values())
                    sponge_power = sponge_diagnostics["sponge_power"]
                    largest_positive_sponge_power = max(
                        largest_positive_sponge_power, sponge_power
                    )
                    sponge_history.add_row(
                        now,
                        sponge_power,
                        sponge_diagnostics["sponge_core_velocity_rms"],
                    )
                worst_volume_error = max(worst_volume_error, volume_error)
                worst_tip_pin = max(worst_tip_pin, tip_pin)
                minimum_sampled_speed = min(minimum_sampled_speed, speed)
                previous_target = float(target)

                if args.stop_when_arrested:
                    arrest = arrest_tracker.observe(last_row, radius0)
                if args.stop_on_height_plateau:
                    plateau_candidate = height_tracker.observe(now, height)
                    # Stop only on a scheduled field boundary so the certified
                    # endpoint always has a matching VTU and restart indices
                    # remain the deterministic stride sequence.
                    if (
                        plateau_candidate is not None
                        and control_index % args.field_stride == 0
                    ):
                        height_plateau = plateau_candidate

                if control_index % args.field_stride == 0:
                    problem.output(quiet=True)
                    field_index += 1

                if args.checkpoint_every and control_index % args.checkpoint_every == 0:
                    problem.save_state(
                        str(checkpoint_dir / f"ckpt_{control_index:05d}.dump"), quiet=True
                    )
                    if coupling is not None:
                        coupling.save(checkpoint_dir / f"ckpt_{control_index:05d}.farfield.npz")
                    _write_field_history(
                        Path(problem.get_output_directory()), restart["fields"] if restart else []
                    )
                if arrest is not None or height_plateau is not None:
                    break

            assert last_row is not None
            _write_field_history(
                Path(problem.get_output_directory()), restart["fields"] if restart else []
            )
            radius_final = last_row[1]
            radius_increase = radius_final - radius0
            failures = []
            if radius_increase < args.min_radius_increase:
                failures.append(
                    f"hole growth {radius_increase:.6e} is below {args.min_radius_increase:.6e}"
                )
            # At the far edge u=0 and the mesh moves with +U, so the relative
            # normal velocity is -U.  Psi=0 is an inflow condition only while
            # U>=0; elastic recoil would turn that edge into outflow and make
            # the strong conformation condition invalid.
            if coupling is not None and coupling.outer_velocity > FAR_FIELD_OUTER_TOLERANCE:
                failures.append(
                    "far-field pulse reached the end of the membrane extension "
                    f"(|u| = {coupling.outer_velocity:.3e} in its outer tenth)"
                )
            if args.max_recoil_speed is not None and minimum_sampled_speed < -args.max_recoil_speed:
                failures.append(
                    "far boundary ceased to be inflow at an output sample "
                    f"(minimum sampled U={minimum_sampled_speed:.6e})"
                )
            if (
                not args.stop_when_arrested
                and not args.stop_on_height_plateau
                and not args.allow_recoil
                and last_row[2] <= args.min_final_speed
            ):
                failures.append(
                    f"final retraction speed {last_row[2]:.6e} is not above "
                    f"{args.min_final_speed:.6e}"
                )
            if worst_volume_error > args.max_volume_error:
                failures.append(
                    f"window-volume relative error {worst_volume_error:.3e} exceeds "
                    f"{args.max_volume_error:.3e}"
                )
            if worst_tip_pin > args.max_tip_pin_residual:
                failures.append(
                    f"tip-pin residual {worst_tip_pin:.3e} exceeds {args.max_tip_pin_residual:.3e}"
                )
            if diagnostics:
                stress_abs_max = max(
                    abs(diagnostics["polymer_stress_eigenvalue_min"]),
                    abs(diagnostics["polymer_stress_eigenvalue_max"]),
                )
                if diagnostics["conformation_eigenvalue_min"] <= 0:
                    failures.append("conformation tensor lost positive definiteness")
                if args.elastic_modulus > 0 and stress_abs_max < args.min_polymer_stress:
                    failures.append(
                        f"polymer stress {stress_abs_max:.6e} is below "
                        f"{args.min_polymer_stress:.6e}"
                    )
            if largest_positive_sponge_power > args.max_positive_sponge_power:
                failures.append(
                    f"sponge power {largest_positive_sponge_power:.6e} exceeds "
                    f"{args.max_positive_sponge_power:.6e}"
                )

            completion_reason = "fixed_horizon"
            if args.stop_when_arrested:
                completion_reason = (
                    "arrest_detected" if arrest is not None else "not_arrested_by_horizon"
                )
            elif args.stop_on_height_plateau:
                completion_reason = (
                    "height_plateau_detected"
                    if height_plateau is not None
                    else "height_plateau_not_reached_by_horizon"
                )
            status = "failed" if failures else "passed"
            if not failures and completion_reason in {
                "not_arrested_by_horizon", "height_plateau_not_reached_by_horizon"
            }:
                status = "horizon_reached"

            metadata.update(
                {
                    "status": status,
                    "completion_reason": completion_reason,
                    "R_initial": radius0,
                    "R_final": radius_final,
                    "R_increase": radius_increase,
                    "minimum_sampled_U": minimum_sampled_speed,
                    "far_boundary_remained_sampled_inflow": minimum_sampled_speed >= 0,
                    "U_final": last_row[2],
                    "ustar_final": last_row[3],
                    "worst_volume_rel_error": worst_volume_error,
                    "worst_tip_pin_residual": worst_tip_pin,
                    "polymer_diagnostics_available": bool(diagnostics),
                    "polymer_diagnostics_final": diagnostics,
                    "sponge_diagnostics_available": bool(sponge_diagnostics),
                    "sponge_diagnostics_final": sponge_diagnostics,
                    "largest_positive_sponge_power": largest_positive_sponge_power,
                    "far_field_outflow": coupling.outflow if coupling is not None else None,
                    "far_field_outer_velocity_max": (
                        coupling.outer_velocity if coupling is not None else None
                    ),
                    "opening_established": arrest_tracker.opening_established,
                    "arrest_detected": arrest is not None,
                    "arrest": arrest,
                    "height_plateau_detected": height_plateau is not None,
                    "height_plateau_certificate": height_plateau,
                    "height_plateau_latest_metrics": height_tracker.latest_metrics,
                    "H_final": height,
                    "t_reached": last_row[0],
                    "outputs_written": field_index,
                    "scalar_samples_written": control_index,
                    "fields_written": field_index,
                    "failure_reasons": failures,
                }
            )
    except BaseException as exc:
        metadata["status"] = "error"
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        _write_json(run_dir / "case.json", metadata)
        raise
    finally:
        if Path.cwd() != previous_cwd:
            raise RuntimeError("viscoelastic driver changed its process working directory")

    _write_json(run_dir / "case.json", metadata)
    if metadata["failure_reasons"]:
        raise RuntimeError("; ".join(metadata["failure_reasons"]))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart-case", type=Path, help="serialized case with hash-bound parent lineage"
    )
    parser.add_argument(
        "--oh",
        type=_nonnegative_float,
        default=0.0,
        help="solvent Ohnesorge number; mu_s=sqrt(2)*Oh",
    )
    parser.add_argument(
        "--elastic-modulus",
        type=_nonnegative_float,
        default=1.0,
        help="dimensionless polymer modulus G=G_dim*h0/sigma",
    )
    parser.add_argument(
        "--passive-conformation",
        action="store_true",
        help=(
            "evolve log-conformation diagnostics without feeding polymer stress "
            "back into momentum; required when G=0"
        ),
    )
    parser.add_argument(
        "--relaxation-time",
        type=_relaxation_time,
        default=math.inf,
        help="dimensionless lambda, or 'inf' for no relaxation",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="output root; one canonical case directory is created below it",
    )
    parser.add_argument(
        "--t-end", type=_positive_float, default=0.2, help="end time in capillary units"
    )
    parser.add_argument("--resolution", type=_positive_float, default=0.25)
    parser.add_argument("--far-resolution", type=_positive_float, default=0.5)
    parser.add_argument(
        "--length",
        type=_positive_float,
        default=8.0,
        help="fixed radial length of the co-moving window",
    )
    parser.add_argument("--r0", type=_positive_float, default=1.0, help="initial hole radius")
    parser.add_argument("--n-outputs", type=int, default=5)
    parser.add_argument(
        "--output-step",
        type=_positive_float,
        default=None,
        help="uniform physical-time interval between arrest checks; overrides --n-outputs",
    )
    parser.add_argument(
        "--field-stride",
        type=int,
        default=1,
        help="write one VTU every N scalar samples",
    )
    parser.add_argument(
        "--stabilization",
        choices=("supg", "none"),
        default="supg",
        help="conformation stabilisation: SUPG or plain Galerkin",
    )
    parser.add_argument(
        "--supg-factor",
        type=_positive_float,
        default=1.0,
        help="multiplicative SUPG factor; recorded even when stabilisation is disabled",
    )
    parser.add_argument("--sponge-start", type=_nonnegative_float, default=None)
    parser.add_argument("--sponge-width", type=_positive_float, default=None)
    parser.add_argument("--sponge-gamma-max", type=_nonnegative_float, default=None)
    parser.add_argument("--sponge-profile", choices=("quintic",), default="quintic")
    parser.add_argument(
        "--max-positive-sponge-power",
        type=_nonnegative_float,
        default=1e-12,
        help="largest tolerated positive value of integral f_sponge dot u",
    )
    parser.add_argument(
        "--far-field",
        choices=("rest", "membrane"),
        default="rest",
        help="far-edge film: at rest (default) or coupled to the thin-film far-field membrane",
    )
    parser.add_argument("--far-field-inner-fraction", type=_positive_float, default=None)
    parser.add_argument("--far-field-extension", type=_positive_float, default=None)
    parser.add_argument("--far-field-node-spacing", type=_positive_float, default=None)
    parser.add_argument(
        "--fixed-step",
        type=_positive_float,
        default=None,
        help="disable temporal adaptivity and cap maxstep after BDF startup",
    )
    parser.add_argument(
        "--max-step",
        type=_positive_float,
        default=None,
        help="maximum adaptive step (default: each output interval)",
    )
    parser.add_argument("--temporal-error", type=_positive_float, default=1e-3)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--max-volume-error", type=_positive_float, default=5e-3)
    parser.add_argument("--max-tip-pin-residual", type=_positive_float, default=1e-6)
    parser.add_argument("--min-radius-increase", type=_nonnegative_float, default=1e-4)
    parser.add_argument(
        "--max-recoil-speed",
        type=_nonnegative_float,
        default=None,
        help="optional failure threshold for inward U at output samples; always recorded",
    )
    parser.add_argument("--min-final-speed", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--allow-recoil",
        action="store_true",
        help=(
            "permit a non-positive final U in fixed-horizon diagnostic runs; "
            "far-boundary inflow validity is still recorded"
        ),
    )
    parser.add_argument("--min-polymer-stress", type=_nonnegative_float, default=1e-10)
    parser.add_argument(
        "--stop-when-arrested",
        action="store_true",
        help="stop after post-startup U crosses zero or remains near zero",
    )
    parser.add_argument(
        "--stop-on-height-plateau",
        action="store_true",
        help="stop only after the rim-height plateau predicates survive their hold",
    )
    parser.add_argument("--height-hold", type=_positive_float, default=10.0)
    parser.add_argument("--height-minimum-window", type=_positive_float, default=10.0)
    parser.add_argument(
        "--height-window-time-fraction", type=_positive_float, default=0.1
    )
    parser.add_argument(
        "--height-slope-window-fraction", type=_positive_float, default=0.005
    )
    parser.add_argument(
        "--height-instantaneous-slope-fraction", type=_positive_float, default=0.02
    )
    parser.add_argument(
        "--height-residual-rms-fraction", type=_positive_float, default=0.005
    )
    parser.add_argument(
        "--height-adjacent-mean-fraction", type=_positive_float, default=0.005
    )
    parser.add_argument(
        "--height-envelope-growth-fraction", type=_nonnegative_float, default=0.05
    )
    parser.add_argument(
        "--height-oscillation-period",
        type=_positive_float,
        default=None,
        help="optional known period; no period is estimated when omitted",
    )
    parser.add_argument("--opening-ustar", type=_positive_float, default=0.02)
    parser.add_argument("--arrest-ustar", type=_positive_float, default=1e-3)
    parser.add_argument("--arrest-consecutive", type=int, default=11)
    parser.add_argument(
        "--arrest-trend-tolerance",
        type=_nonnegative_float,
        default=1e-4,
        help="largest fitted du*/dt accepted across a near-zero sample window",
    )
    parser.add_argument(
        "--arrest-radius-span",
        type=_positive_float,
        default=1e-3,
        help="largest R range accepted across a near-zero sample window",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write one restart state every N scalar samples; 0 disables checkpoints",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Apply cross-argument checks shared by CLI and serialized-case runners."""
    try:
        passive_conformation = _passive_conformation(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.elastic_modulus == 0 and not passive_conformation:
        parser.error("--elastic-modulus=0 requires --passive-conformation")
    if args.n_outputs < 1:
        parser.error("--n-outputs must be at least 1")
    if args.max_newton < 1:
        parser.error("--max-newton must be at least 1")
    if args.arrest_consecutive < 2:
        parser.error("--arrest-consecutive must be at least 2")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.field_stride < 1:
        parser.error("--field-stride must be at least 1")
    if args.checkpoint_every and args.checkpoint_every % args.field_stride:
        parser.error("--checkpoint-every must be a multiple of --field-stride")
    if args.stop_when_arrested and args.stop_on_height_plateau:
        parser.error("tip arrest and height plateau are distinct completion modes")
    if args.stop_when_arrested and args.field_stride != 1:
        parser.error("tip-arrest completion requires --field-stride=1")
    if args.length <= args.resolution:
        parser.error("--length must exceed --resolution")
    try:
        _sponge_parameters(args)
        _far_field_config(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    metadata = run_case(args)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
