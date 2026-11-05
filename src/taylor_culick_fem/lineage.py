"""Restart evidence for a continuation, derived from a parent run folder.

A restart names one checkpoint, ``<parent>/checkpoints/ckpt_NNNNN.dump``, of a run whose
folder holds ``run-manifest.json``. From that folder this module collects, each pinned by
SHA-256, the files the driver needs to continue it: the checkpoint, the parent's input-case
snapshot and live metadata, and its control, rim-height and field histories. The control
index is the checkpoint number; the field index follows from the parent's field stride.

A continuation may have changed the output cadence, so the control rows follow the parent's
recorded numerics schedule (``numerics_schedule`` in its ``case.json``). The schedule
helpers here are shared with the driver.

It imports neither pyoomph nor the solver. Hashes are valid only while the parent files are
final, so continue a run only after its process has stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

CHECKPOINT = re.compile(r"ckpt_(\d+)\.dump")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path: Path) -> dict[str, str]:
    """``{"path", "sha256"}`` of an existing parent file."""
    if not path.is_file():
        raise ValueError(f"restart parent file is missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def segment_steps(schedule: list[dict], last_step: float | None) -> list[float]:
    """The output cadence of each recorded schedule segment.

    Entries written before the cadence could change carry no ``output_step``. A segment
    followed by another takes its cadence from the two boundaries (their time and control
    index differences), which is exact for the uniform grid it ran on; the last such
    segment inherits the previous segment's cadence, or the run's recorded
    ``output_step`` when it is the only one. This keeps a first segment correct after a
    restart has changed the cadence (the parent's ``output_step`` is then the new one).
    """

    steps: list[float] = []
    for k, entry in enumerate(schedule):
        step = entry.get("output_step")
        if step is None and k + 1 < len(schedule):
            following = schedule[k + 1]
            rows = int(following["control_index"]) - int(entry["control_index"])
            if rows > 0:
                step = (float(following["time"]) - float(entry["time"])) / rows
        if step is None:
            step = steps[-1] if steps else last_step
        if step is None:
            raise ValueError("restart requires a fixed output cadence")
        steps.append(float(step))
    return steps


def schedule_row_times(schedule: list[dict], last_step: float | None, count: int) -> np.ndarray:
    """Nominal control-row times 1..count under the recorded cadence segments."""

    return schedule_rows(schedule, last_step, count)[0]


# pyoomph ends an output interval with a step of 1.00001 times the time remaining, so a
# row lands up to 1e-5 of its final step (at most the cadence) past its nominal time.
LANDING_ABS_TOLERANCE = 1e-6
LANDING_REL_TOLERANCE = 2e-5


def check_rows_on_schedule(
    times: np.ndarray, schedule: list[dict], last_step: float | None
) -> np.ndarray:
    """The nominal grid of recorded control rows, refusing rows off it.

    A row may sit past its nominal time by the solver's landing overshoot, which grows
    with the cadence; anything further (such as a shortened final interval) is refused.
    """

    grid, row_steps = schedule_rows(schedule, last_step, len(times))
    landing = LANDING_ABS_TOLERANCE + LANDING_REL_TOLERANCE * row_steps
    if not np.all(np.abs(np.asarray(times) - grid) <= landing):
        raise ValueError("restart control history does not match the output cadence")
    return grid


def schedule_rows(
    schedule: list[dict], last_step: float | None, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal times of control rows 1..count and the cadence each row was written at."""

    steps = segment_steps(schedule, last_step)
    times = np.empty(count)
    row_steps = np.empty(count)
    origin_time, origin_index, step = 0.0, 0, steps[0]
    # A segment's grid continues from the boundary row's nominal time, as the continuation
    # that started it built its targets; the entry's own time is where the solver landed.
    boundaries = [
        (int(entry["control_index"]), segment_step)
        for entry, segment_step in zip(schedule[1:], steps[1:])
    ]
    for j in range(1, count + 1):
        while boundaries and j > boundaries[0][0]:
            origin_index, step = boundaries.pop(0)
            origin_time = float(times[origin_index - 1]) if origin_index > 0 else 0.0
        if origin_index == 0:
            times[j - 1] = step * j
        else:
            times[j - 1] = origin_time + step * (j - origin_index)
        row_steps[j - 1] = step
    return times, row_steps


def control_time(control: Path, control_index: int, cadence: float,
                 schedule: list[dict] | None = None) -> float:
    """Time of control row ``control_index`` (1-based), checked against the output cadence.

    With the parent's ``schedule`` the nominal time follows its cadence segments; without
    it the cadence is uniform from t = 0. A row may land past its nominal time by the
    solver's overshoot, which grows with the cadence.
    """
    rows = [line.split() for line in control.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) < control_index:
        raise ValueError(f"parent control history has {len(rows)} rows, fewer than "
                         f"checkpoint index {control_index}")
    time = float(rows[control_index - 1][0])
    if schedule:
        grid, steps = schedule_rows(schedule, cadence, control_index)
        expected, step = float(grid[-1]), float(steps[-1])
    else:
        expected, step = control_index * cadence, cadence
    # Stored times carry float noise of order 1e-7; compare in time units, not index units.
    tolerance = max(1e-5, LANDING_ABS_TOLERANCE + LANDING_REL_TOLERANCE * step)
    if abs(time - expected) > tolerance:
        raise ValueError(f"parent control row {control_index} has t={time}, not its scheduled "
                         f"t={expected}")
    return time


def restart_evidence(state: Path, parent: Path) -> dict:
    """Pinned references and indices for continuing ``parent`` from checkpoint ``state``."""
    state = state.resolve()
    parent = parent.resolve()
    match = CHECKPOINT.fullmatch(state.name)
    if match is None or state.parent != parent / "checkpoints":
        raise ValueError("--restart-from must name <parent run>/checkpoints/ckpt_NNNNN.dump")
    control_index = int(match.group(1))
    if control_index < 1:
        raise ValueError("a restart needs a checkpoint after the first output")
    input_case = parent / "input-case.json"
    parent_case = json.loads(input_case.read_text(encoding="utf-8"))
    cadence = float(parent_case["output"]["cadence"])
    stride = int(parent_case["output"].get("field_stride", 1))
    metadata = json.loads((parent / "case.json").read_text(encoding="utf-8"))
    schedule = metadata.get("numerics_schedule")
    if control_index % stride:
        raise ValueError(f"checkpoint index {control_index} is not a multiple of the parent's "
                         f"field stride {stride}")
    run = parent / "run_viscoelastic_axi"
    history = run / "liquid-history.pvd"
    # A continued run's liquid.pvd lists only its own fields; liquid-history.pvd is the
    # parent prefix plus its own fields, the complete history a further restart needs.
    pvd = history if history.is_file() else run / "liquid.pvd"
    evidence = {
        "checkpoint": reference(state),
        "input_case": reference(input_case),
        "metadata": reference(parent / "case.json"),
        "control": reference(run / "control.txt"),
        "rim_height": reference(run / "rim-height.txt"),
        "pvd": reference(pvd),
        "time": control_time(run / "control.txt", control_index, cadence, schedule),
        "control_index": control_index,
        "field_index": control_index // stride,
    }
    far_field = parent_case.get("physics", {}).get("far_field")
    if isinstance(far_field, dict) and far_field.get("mode") == "relaxing_membrane":
        # The finite-relaxation exterior's state and history are pinned with the FEM state.
        evidence["far_field_state"] = reference(state.with_suffix(".farfield.npz"))
        evidence["far_field_history"] = reference(run / "far-field.txt")
    return evidence
