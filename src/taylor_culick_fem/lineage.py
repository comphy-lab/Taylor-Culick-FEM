"""Restart evidence for a continuation, derived from a parent run folder.

A restart names one checkpoint, ``<parent>/checkpoints/ckpt_NNNNN.dump``, of a run whose
folder holds ``run-manifest.json``. From that folder this module collects, each pinned by
SHA-256, the files the driver needs to continue it: the checkpoint, the parent's input-case
snapshot and live metadata, and its control, rim-height and field histories. The control
index is the checkpoint number; the field index follows from the parent's field stride.

It imports neither pyoomph nor the solver. Hashes are valid only while the parent files are
final, so continue a run only after its process has stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

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


def control_time(control: Path, control_index: int, cadence: float) -> float:
    """Time of control row ``control_index`` (1-based), checked against the output cadence."""
    rows = [line.split() for line in control.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) < control_index:
        raise ValueError(f"parent control history has {len(rows)} rows, fewer than "
                         f"checkpoint index {control_index}")
    time = float(rows[control_index - 1][0])
    # Stored times carry float noise of order 1e-7; compare in time units, not index units.
    if abs(time - control_index * cadence) > 1e-5:
        raise ValueError(f"parent control row {control_index} has t={time}, "
                         f"not {control_index} x {cadence}")
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
    if control_index % stride:
        raise ValueError(f"checkpoint index {control_index} is not a multiple of the parent's "
                         f"field stride {stride}")
    run = parent / "run_viscoelastic_axi"
    history = run / "liquid-history.pvd"
    # A continued run's liquid.pvd lists only its own fields; liquid-history.pvd is the
    # parent prefix plus its own fields, the complete history a further restart needs.
    pvd = history if history.is_file() else run / "liquid.pvd"
    return {
        "checkpoint": reference(state),
        "input_case": reference(input_case),
        "metadata": reference(parent / "case.json"),
        "control": reference(run / "control.txt"),
        "rim_height": reference(run / "rim-height.txt"),
        "pvd": reference(pvd),
        "time": control_time(run / "control.txt", control_index, cadence),
        "control_index": control_index,
        "field_index": control_index // stride,
    }
