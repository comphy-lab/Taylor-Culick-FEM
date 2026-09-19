"""Software tests for the laboratory-velocity Rayleigh sponge."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.absorbing_layer import (
    RayleighSpongeParameters,
    rayleigh_coefficient_numeric,
)


def test_quintic_profile_is_bounded_monotone_and_flat_at_endpoints() -> None:
    parameters = RayleighSpongeParameters(start=6, width=4, gamma_max=3)
    x = np.linspace(0, 14, 1401)
    gamma = rayleigh_coefficient_numeric(x, parameters)

    assert np.all(gamma[x <= 6] == 0)
    assert np.all(gamma[x >= 10] == 3)
    assert np.all(np.diff(gamma) >= -1e-14)
    assert gamma.min() == 0
    assert gamma.max() == 3

    # The analytic quintic has zero first and second derivatives at both
    # joins; evaluate its exact polynomial derivatives independently.
    for s in (0.0, 1.0):
        first = 30 * s**2 * (s - 1) ** 2
        second = 60 * s * (2 * s**2 - 3 * s + 1)
        assert first == 0
        assert second == 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"start": -1, "width": 1, "gamma_max": 1}, "start"),
        ({"start": 0, "width": 0, "gamma_max": 1}, "width"),
        ({"start": 0, "width": 1, "gamma_max": -1}, "gamma_max"),
        ({"start": 0, "width": 1, "gamma_max": 1, "profile": "linear"}, "profile"),
    ],
)
def test_sponge_parameters_reject_invalid_values(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        RayleighSpongeParameters(**kwargs)


def test_sponge_must_fit_inside_window() -> None:
    parameters = RayleighSpongeParameters(start=7, width=4, gamma_max=1)
    with pytest.raises(ValueError, match="must not exceed"):
        parameters.validate_window(10)


def test_rayleigh_power_is_nonpositive_for_arbitrary_velocities() -> None:
    rng = np.random.default_rng(2718)
    parameters = RayleighSpongeParameters(start=3, width=5, gamma_max=2)
    x = rng.uniform(0, 10, 1000)
    velocity = rng.normal(size=(1000, 2))
    gamma = rayleigh_coefficient_numeric(x, parameters)
    power_density = -gamma * np.einsum("ij,ij->i", velocity, velocity)
    assert np.all(power_density <= 0)


def test_zero_gamma_is_identically_zero() -> None:
    parameters = RayleighSpongeParameters(start=3, width=5, gamma_max=0)
    np.testing.assert_array_equal(
        rayleigh_coefficient_numeric(np.linspace(-10, 20, 101), parameters), 0
    )
