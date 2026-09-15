"""Load and classify Taylor--Culick campaign control histories.

The Newtonian and viscoelastic drivers deliberately use different control
schemas.  This module keeps those interpretations separate and converts both
to the common ``(t, R_tip, u*)`` representation used by campaign analysis and
plotting.  The exact initial state is reconstructed when the solver record
starts after zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TipHistory:
    """A validated tip history in capillary units."""

    time: np.ndarray
    radius: np.ndarray
    ustar: np.ndarray
    schema: str
    source: Path


@dataclass(frozen=True)
class SettlingThresholds:
    """Thresholds applied to a complete trailing window."""

    window_duration: float = 50.0
    maximum_absolute_ustar: float = 1e-3
    maximum_radius_span: float = 1e-3
    maximum_absolute_radius_slope: float = 1e-5
    opening_ustar: float = 0.02
    minimum_radius_increase: float = 0.1

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _load_table(path: Path, *, columns: int, schema: str) -> np.ndarray:
    try:
        table = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{path} is not a numeric {schema} control history") from exc
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != columns:
        raise ValueError(f"{path} must contain exactly {columns} columns for schema {schema}")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path} contains non-finite values")
    if table[0, 0] < 0 or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f"{path} times must be non-negative and strictly increasing")
    return table


def _with_initial_state(
    time: np.ndarray,
    radius: np.ndarray,
    ustar: np.ndarray,
    *,
    r0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be finite and positive")
    if time[0] > 0:
        return (
            np.concatenate(([0.0], time)),
            np.concatenate(([r0], radius)),
            np.concatenate(([0.0], ustar)),
        )
    if not math.isclose(radius[0], r0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("the recorded t=0 radius disagrees with r0")
    if not math.isclose(ustar[0], 0.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("the recorded t=0 tip speed must be zero")
    return time.copy(), radius.copy(), ustar.copy()


def load_viscoelastic_control(path: str | Path, *, r0: float = 1.0) -> TipHistory:
    """Load ``t,R,U,u*,volume,...`` from the eight-column elastic driver."""
    source = Path(path)
    table = _load_table(source, columns=8, schema="viscoelastic")
    time, radius, ustar = _with_initial_state(
        table[:, 0], table[:, 1], table[:, 3], r0=r0
    )
    return TipHistory(time, radius, ustar, "viscoelastic", source)


def load_newtonian_control(path: str | Path, *, r0: float = 1.0) -> TipHistory:
    """Load ``t,t*,R,Rdot,u*,volume,...`` from the seven-column Newtonian driver."""
    source = Path(path)
    table = _load_table(source, columns=7, schema="newtonian")
    time, radius, ustar = _with_initial_state(
        table[:, 0], table[:, 2], table[:, 4], r0=r0
    )
    return TipHistory(time, radius, ustar, "newtonian", source)


def load_control(path: str | Path, *, schema: str, r0: float = 1.0) -> TipHistory:
    """Dispatch only between the two explicit supported schemas."""
    if schema == "viscoelastic":
        return load_viscoelastic_control(path, r0=r0)
    if schema == "newtonian":
        return load_newtonian_control(path, r0=r0)
    raise ValueError("schema must be 'newtonian' or 'viscoelastic'")


def _linear_slope(time: np.ndarray, radius: np.ndarray) -> float:
    centred = time - float(np.mean(time))
    denominator = float(np.dot(centred, centred))
    if denominator == 0:
        return math.nan
    return float(np.dot(centred, radius - float(np.mean(radius))) / denominator)


def _trailing_window(
    time: np.ndarray,
    radius: np.ndarray,
    ustar: np.ndarray,
    *,
    start: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a trailing window with an interpolated sample at its exact start."""
    later = time > start
    return (
        np.concatenate(([start], time[later])),
        np.concatenate(([np.interp(start, time, radius)], radius[later])),
        np.concatenate(([np.interp(start, time, ustar)], ustar[later])),
    )


def analyse_settling(
    history: TipHistory,
    thresholds: SettlingThresholds | None = None,
) -> dict[str, object]:
    """Classify the horizon using one complete trailing window.

    A zero crossing is never sufficient: the history must first establish
    opening and must subsequently cover the full requested window.  ``settled``
    additionally requires the radius span and fitted radius drift to pass.
    """
    if thresholds is None:
        thresholds = SettlingThresholds()
    thresholds.validate()
    t, radius, ustar = history.time, history.radius, history.ustar
    opened = (ustar >= thresholds.opening_ustar) & (
        radius - radius[0] >= thresholds.minimum_radius_increase
    )
    opening_indices = np.flatnonzero(opened)
    established_opening = bool(len(opening_indices))
    opening_time = float(t[opening_indices[0]]) if established_opening else None
    window_start_target = float(t[-1] - thresholds.window_duration)
    complete_window = bool(
        established_opening
        and window_start_target >= t[opening_indices[0]]
        and window_start_target >= t[0]
    )

    if complete_window:
        tw, rw, uw = _trailing_window(
            t, radius, ustar, start=window_start_target
        )
        max_abs_ustar = float(np.max(np.abs(uw)))
        radius_span = float(np.ptp(rw))
        radius_slope = _linear_slope(tw, rw)
        quiet = max_abs_ustar < thresholds.maximum_absolute_ustar
        strict = bool(
            quiet
            and radius_span < thresholds.maximum_radius_span
            and abs(radius_slope) < thresholds.maximum_absolute_radius_slope
        )
    else:
        tw = rw = uw = np.array([], dtype=float)
        max_abs_ustar = radius_span = radius_slope = None
        quiet = strict = False

    if strict:
        status = "strict settled"
    elif quiet:
        status = "quiet-tip plateau"
    elif established_opening and ustar[-1] >= thresholds.maximum_absolute_ustar:
        status = "retracting"
    else:
        status = "not arrested"

    return {
        "status": status,
        "schema": history.schema,
        "samples": len(t),
        "time_start": float(t[0]),
        "time_end": float(t[-1]),
        "radius_end": float(radius[-1]),
        "ustar_end": float(ustar[-1]),
        "established_opening": established_opening,
        "opening_time": opening_time,
        "complete_trailing_window": complete_window,
        "window": {
            "requested_duration": thresholds.window_duration,
            "time_start": float(tw[0]) if len(tw) else None,
            "time_end": float(tw[-1]) if len(tw) else None,
            "samples": len(tw),
            "maximum_absolute_ustar": max_abs_ustar,
            "radius_span": radius_span,
            "linear_radius_slope": radius_slope,
        },
        "thresholds": asdict(thresholds),
    }
