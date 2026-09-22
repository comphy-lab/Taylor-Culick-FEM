#!/usr/bin/env python3
"""Plot manifest-declared Taylor--Culick radius and signed speed histories.

Data paths are resolved relative to the manifest.  The manifest supplies the
zero-based time, radius and velocity columns and the velocity normalisation,
so Newtonian ``globals.txt`` and viscoelastic ``control.txt`` histories can be
compared without silently conflating their different schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class History:
    label: str
    time: np.ndarray
    radius: np.ndarray
    speed: np.ndarray
    colour: str | None
    linestyle: str
    source: Path
    source_sha256: str


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _finite_positive(value: Any, *, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be finite and positive")
    return result


def _column(columns: dict[str, Any], name: str, *, context: str) -> int:
    value = columns.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.columns.{name} must be a non-negative integer")
    return value


def _prepend_initial(
    time: np.ndarray,
    radius: np.ndarray,
    speed: np.ndarray,
    initial: Any,
    velocity_divisor: float,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if initial is None:
        return time, radius, speed
    values = _mapping(initial, context=f"{context}.initial_state")
    initial_time = float(values["time"])
    initial_radius = float(values["radius"])
    initial_speed = float(values["velocity"]) / velocity_divisor
    if not np.all(np.isfinite([initial_time, initial_radius, initial_speed])):
        raise ValueError(f"{context}.initial_state must be finite")
    if initial_time < 0 or initial_time >= time[0]:
        raise ValueError(f"{context}.initial_state time must be non-negative and precede the first sample")
    return (
        np.concatenate(([initial_time], time)),
        np.concatenate(([initial_radius], radius)),
        np.concatenate(([initial_speed], speed)),
    )


def _truncate_at(
    time: np.ndarray,
    radius: np.ndarray,
    speed: np.ndarray,
    t_max: float,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time[0] > 0 or time[-1] < t_max:
        raise ValueError(
            f"{context} must cover the requested interval 0 <= t <= {t_max:g}; "
            f"available interval is {time[0]:g} <= t <= {time[-1]:g}"
        )
    keep = time <= t_max
    clipped_time = time[keep]
    clipped_radius = radius[keep]
    clipped_speed = speed[keep]
    if clipped_time[-1] < t_max:
        clipped_time = np.concatenate((clipped_time, [t_max]))
        clipped_radius = np.concatenate((clipped_radius, [np.interp(t_max, time, radius)]))
        clipped_speed = np.concatenate((clipped_speed, [np.interp(t_max, time, speed)]))
    return clipped_time, clipped_radius, clipped_speed


def load_history(entry: Any, manifest_dir: Path, t_max: float, index: int) -> History:
    context = f"histories[{index}]"
    item = _mapping(entry, context=context)
    label = item.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{context}.label must be a non-empty string")
    for name in ("Ec", "Oh"):
        value = float(item[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{context}.{name} must be finite and non-negative")
    de = item["De"]
    if not (
        de == "inf"
        or (not isinstance(de, bool) and math.isfinite(float(de)) and float(de) >= 0)
    ):
        raise ValueError(f"{context}.De must be finite and non-negative or 'inf'")
    source_value = item.get("data")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"{context}.data must be a non-empty path string")
    source = (manifest_dir / source_value).resolve()
    expected_sha256 = item.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{context}.sha256 must be a SHA-256 hex digest")
    snapshot = source.read_bytes()
    actual_sha256 = hashlib.sha256(snapshot).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{context} hash mismatch for {source}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    columns = _mapping(item.get("columns"), context=f"{context}.columns")
    time_column = _column(columns, "time", context=context)
    radius_column = _column(columns, "radius", context=context)
    velocity_column = _column(columns, "velocity", context=context)
    if len({time_column, radius_column, velocity_column}) != 3:
        raise ValueError(f"{context} time, radius and velocity columns must be distinct")
    required_columns = max(time_column, radius_column, velocity_column) + 1
    table = np.loadtxt(io.BytesIO(snapshot), comments="#", ndmin=2)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < required_columns:
        raise ValueError(
            f"{source} needs at least {required_columns} columns and one data row"
        )
    selected = table[:, [time_column, radius_column, velocity_column]].astype(
        float, copy=True
    )
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{source} contains non-finite selected history values")
    if selected[0, 0] < 0 or np.any(np.diff(selected[:, 0]) <= 0):
        raise ValueError(f"{source} times must be non-negative and strictly increasing")

    velocity_divisor = _finite_positive(
        item.get("velocity_divisor", 1.0), context=f"{context}.velocity_divisor"
    )
    time, radius, speed = selected.T
    speed = speed / velocity_divisor
    time, radius, speed = _prepend_initial(
        time,
        radius,
        speed,
        item.get("initial_state"),
        velocity_divisor,
        context=context,
    )
    time, radius, speed = _truncate_at(
        time, radius, speed, t_max, context=context
    )

    colour = item.get("colour")
    if colour is not None and not isinstance(colour, str):
        raise TypeError(f"{context}.colour must be a string")
    linestyle = item.get("linestyle", "-")
    if not isinstance(linestyle, str):
        raise TypeError(f"{context}.linestyle must be a string")
    return History(
        label=label,
        time=time,
        radius=radius,
        speed=speed,
        colour=colour,
        linestyle=linestyle,
        source=source,
        source_sha256=actual_sha256,
    )


def load_manifest(
    path: Path, *, snapshot: bytes | None = None
) -> tuple[dict[str, Any], list[History], float]:
    manifest = json.loads(path.read_bytes() if snapshot is None else snapshot)
    if not isinstance(manifest, dict) or manifest.get("schema") != "de-history-comparison-v1":
        raise ValueError("manifest must use schema de-history-comparison-v1")
    t_max = _finite_positive(manifest.get("t_max"), context="t_max")
    entries = manifest.get("histories")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.histories must be a non-empty array")
    histories = [
        load_history(entry, path.parent, t_max, index)
        for index, entry in enumerate(entries)
    ]
    return manifest, histories, t_max


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(which="major", direction="out", width=2.2, length=8, labelsize=20, pad=6)
    axis.tick_params(which="minor", direction="out", width=1.4, length=4)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_linewidth(2.2)


def render(histories: list[History], t_max: float, output: Path) -> None:
    fig, (radius_axis, speed_axis) = plt.subplots(
        2, 1, figsize=(12, 13), sharex=True, constrained_layout=True
    )
    fig.set_facecolor("white")
    for history in histories:
        radius_axis.plot(
            history.time,
            history.radius,
            label=history.label,
            color=history.colour,
            linestyle=history.linestyle,
            linewidth=3.0,
        )
        speed_axis.plot(
            history.time,
            history.speed,
            label=history.label,
            color=history.colour,
            linestyle=history.linestyle,
            linewidth=3.0,
        )

    radius_axis.set_ylabel(r"$R_{\mathrm{tip}}/h_0$", fontsize=28, labelpad=10)
    speed_axis.set_ylabel(r"$U/V_{\mathrm{TC}}$", fontsize=28, labelpad=10)
    speed_axis.set_xlabel(
        r"$t\sqrt{\sigma/(\rho h_0^3)}$", fontsize=28, labelpad=10
    )
    speed_axis.axhline(0.0, color="0.45", linewidth=1.2, linestyle=(0, (5, 4)), zorder=0)
    radius_axis.text(0.02, 0.95, r"$(a)$", transform=radius_axis.transAxes, fontsize=25, va="top")
    speed_axis.text(0.02, 0.95, r"$(b)$", transform=speed_axis.transAxes, fontsize=25, va="top")
    radius_axis.legend(frameon=False, fontsize=19, loc="upper left", bbox_to_anchor=(0.08, 0.99))
    radius_axis.set_xlim(0.0, t_max)
    for axis in (radius_axis, speed_axis):
        _style_axis(axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1, dpi=300)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="output stem; defaults to manifest output")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    snapshot = manifest_path.read_bytes()
    manifest, histories, t_max = load_manifest(manifest_path, snapshot=snapshot)
    output_value = args.output or Path(str(manifest.get("output", "de_reference_comparison")))
    output = output_value if output_value.is_absolute() else manifest_path.parent / output_value
    render(histories, t_max, output)
    receipt = {
        "schema": "de-history-plot-receipt-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(snapshot).hexdigest(),
        "t_max": t_max,
        "outputs": [str(output.with_suffix(".pdf")), str(output.with_suffix(".png"))],
        "histories": [
            {
                "label": history.label,
                "source": str(history.source),
                "sha256": history.source_sha256,
                "samples_plotted": len(history.time),
                "time_end": float(history.time[-1]),
                "radius_end": float(history.radius[-1]),
                "speed_end": float(history.speed[-1]),
            }
            for history in histories
        ],
    }
    receipt_path = output.with_name(output.name + "_receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
