"""Smooth Rayleigh damping for a translating axisymmetric ALE window."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyoomph.expressions import maximum, minimum


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class RayleighSpongeParameters:
    """Parameters for ``f=-rho*gamma(x_rel)*u_lab``.

    ``start`` and ``width`` are measured from the live tip position, so the
    absorber translates with the ALE window without changing the laboratory-
    frame velocity being damped.
    """

    start: float
    width: float
    gamma_max: float
    profile: str = "quintic"

    def __post_init__(self) -> None:
        start = _finite(self.start, name="start")
        width = _finite(self.width, name="width")
        gamma_max = _finite(self.gamma_max, name="gamma_max")
        if start < 0:
            raise ValueError("start must be non-negative")
        if width <= 0:
            raise ValueError("width must be positive")
        if gamma_max < 0:
            raise ValueError("gamma_max must be non-negative")
        if self.profile != "quintic":
            raise ValueError("profile must be 'quintic'")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "gamma_max", gamma_max)

    def validate_window(self, length: float) -> None:
        length = _finite(length, name="length")
        if length <= 0:
            raise ValueError("length must be positive")
        if self.start + self.width > length:
            raise ValueError("sponge start + width must not exceed window length")

    def as_json_dict(self) -> dict[str, float | str]:
        return {
            "start": self.start,
            "width": self.width,
            "gamma_max": self.gamma_max,
            "profile": self.profile,
        }


def quintic_smoothstep(s: Any) -> Any:
    """Return the C2 ramp ``6s^5-15s^4+10s^3`` for clipped ``s``."""

    clipped = minimum(1, maximum(0, s))
    return clipped**3 * (10 + clipped * (-15 + 6 * clipped))


def rayleigh_coefficient(x_rel: Any, parameters: RayleighSpongeParameters) -> Any:
    """Return the symbolic damping coefficient at tip-relative coordinate."""

    return parameters.gamma_max * quintic_smoothstep(
        (x_rel - parameters.start) / parameters.width
    )


def rayleigh_coefficient_numeric(
    x_rel: ArrayLike, parameters: RayleighSpongeParameters
) -> NDArray[np.float64]:
    """Independent numerical evaluator for tests and run diagnostics."""

    x = np.asarray(x_rel, dtype=float)
    s = np.clip((x - parameters.start) / parameters.width, 0.0, 1.0)
    return parameters.gamma_max * s**3 * (10.0 + s * (-15.0 + 6.0 * s))
