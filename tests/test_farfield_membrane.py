"""Physics and restart contracts for the thin-film far-field membrane."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.farfield_membrane import EdgeState, FarFieldMembrane


def _membrane(**overrides) -> FarFieldMembrane:
    values = {
        "elastic_modulus": 1.0,
        "viscosity": math.sqrt(2.0),
        "inner": 50.0,
        "outer": 100.0,
        "nodes": 21,
    }
    values.update(overrides)
    return FarFieldMembrane(**values)


@pytest.mark.parametrize("relaxation_time", [0.0, 31.622776601683793, math.inf])
def test_undriven_film_stays_at_rest(relaxation_time: float) -> None:
    membrane = _membrane(relaxation_time=relaxation_time)
    for step in range(4):
        membrane.advance(0.5 * (step + 1), membrane.inner + 0.5, 0.0, 0.0)
    assert np.max(np.abs(membrane.u)) < 1e-12
    assert np.max(np.abs(membrane.h - 1.0)) < 1e-12
    assert np.max(np.abs(membrane.Y - membrane.radii())) < 1e-9
    assert membrane.state_at(75.0).log_conformation() == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-10
    )


def test_homogeneous_conformation_relaxes_to_identity_exactly() -> None:
    de = 2.5
    membrane = _membrane(
        relaxation_time=de, nodes=17, rtol=1e-10, atol=1e-12
    )
    initial_c = 1.8
    initial_log = math.log(initial_c)
    for name in ("log_c_rr", "log_c_zz", "log_c_th"):
        getattr(membrane, name)[:] = initial_log

    membrane.advance(1.75, membrane.inner, 0.0, 0.0)

    expected_c = 1.0 + (initial_c - 1.0) * math.exp(-membrane.time / de)
    observed = np.exp(membrane.state_at(75.0).log_conformation())
    np.testing.assert_allclose(observed, expected_c, rtol=2e-9, atol=2e-11)
    assert np.max(np.abs(membrane.u)) < 1e-11


def test_finite_log_rates_approach_the_infinite_de_limit() -> None:
    qrr = np.array([math.log(1.4)])
    qzz = np.array([math.log(0.8)])
    qth = np.array([math.log(1.1)])
    radial_rate = np.array([0.3])
    hoop_rate = np.array([-0.1])
    rates = FarFieldMembrane._log_conformation_source(
        qrr, qzz, qth, radial_rate, hoop_rate, 1e14
    )
    expected = (2 * radial_rate, -2 * (radial_rate + hoop_rate), 2 * hoop_rate)
    for observed, limit in zip(rates, expected, strict=True):
        np.testing.assert_allclose(observed, limit, rtol=0, atol=1e-14)

    de = 4.0
    rates = FarFieldMembrane._log_conformation_source(
        qrr, qzz, qth, np.zeros(1), np.zeros(1), de
    )
    for rate, log_c in zip(rates, (qrr, qzz, qth), strict=True):
        np.testing.assert_allclose(rate, -(1 - np.exp(-log_c)) / de, rtol=0, atol=0)


def test_prescribed_homogeneous_extension_matches_exact_oldroyd_b() -> None:
    alpha, de, end = 0.2, 3.0, 0.8
    initial_c = np.array([1.3, 0.9, 1.1])
    principal_rates = np.array([alpha, -2 * alpha, alpha])

    def rhs(_time, logs):
        rates = FarFieldMembrane._log_conformation_source(
            *(np.array([value]) for value in logs),
            np.array([alpha]),
            np.array([alpha]),
            de,
        )
        return np.array([rate[0] for rate in rates])

    solution = solve_ivp(
        rhs, (0.0, end), np.log(initial_c), method="DOP853", rtol=1e-12, atol=1e-14
    )
    observed_c = np.exp(solution.y[:, -1])
    growth = 2 * principal_rates - 1 / de
    expected_c = initial_c * np.exp(growth * end) + np.expm1(growth * end) / (
        de * growth
    )
    np.testing.assert_allclose(observed_c, expected_c, rtol=2e-11, atol=2e-13)
    thickness_squared = math.exp(-4 * alpha * end)
    assert not math.isclose(observed_c[1], thickness_squared, rel_tol=1e-3)


def test_finite_de_state_round_trip_preserves_continuation() -> None:
    membrane = _membrane(relaxation_time=31.622776601683793)
    membrane.advance(0.5, 50.5, 0.0, 0.02)
    restored = FarFieldMembrane.from_arrays(membrane.to_arrays())
    assert restored.relaxation_time == membrane.relaxation_time
    for name in ("u", "h", "Y", "log_c_rr", "log_c_zz", "log_c_th"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(membrane, name))

    membrane.advance(1.0, 51.0, 0.02, 0.03)
    restored.advance(1.0, 51.0, 0.02, 0.03)
    for name in ("u", "h", "Y", "log_c_rr", "log_c_zz", "log_c_th"):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(membrane, name), rtol=0, atol=2e-13
        )


def test_legacy_infinite_de_state_remains_loadable_and_unchanged() -> None:
    membrane = _membrane()
    membrane.advance(0.5, 50.5, 0.0, 0.02)
    arrays = membrane.to_arrays()
    assert "relaxation_time" not in arrays
    assert not any(name.startswith("log_c_") for name in arrays)
    restored = FarFieldMembrane.from_arrays(arrays)
    assert math.isinf(restored.relaxation_time)
    for name in ("u", "h", "Y"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(membrane, name))


def test_zero_de_is_exact_solvent_newtonian_endpoint() -> None:
    zero_de = _membrane(elastic_modulus=7.0, relaxation_time=0.0)
    zero_g = _membrane(elastic_modulus=0.0, relaxation_time=math.inf)
    for membrane in (zero_de, zero_g):
        membrane.advance(0.5, 50.5, 0.0, 0.02)
    np.testing.assert_allclose(zero_de.u, zero_g.u, rtol=0, atol=1e-13)
    np.testing.assert_allclose(zero_de.h, zero_g.h, rtol=0, atol=1e-13)
    assert zero_de.state_at(75.0).log_conformation() == (0.0, 0.0, 0.0)


def test_extension_pads_finite_de_state_with_undisturbed_film() -> None:
    membrane = _membrane(relaxation_time=31.622776601683793)
    membrane.advance(0.5, 50.5, 0.0, 0.02)
    extended = membrane.extended(outer=150.0, node_spacing=0.5)
    assert extended.relaxation_time == membrane.relaxation_time
    beyond = extended.state_at(125.0)
    assert (beyond.velocity, beyond.thickness) == (0.0, 1.0)
    assert beyond.log_conformation() == (0.0, 0.0, 0.0)
    assert extended.outer_state_deviation() == {
        "velocity": 0.0,
        "thickness": 0.0,
        "log_conformation": 0.0,
        "polymer_stress": 0.0,
    }


def test_finite_de_inner_inflow_is_rejected_without_incoming_2d_state() -> None:
    finite = _membrane(relaxation_time=31.622776601683793)
    with pytest.raises(RuntimeError, match="became inflow"):
        finite.advance(0.5, finite.inner, 0.0, 0.01)

    legacy = _membrane(relaxation_time=math.inf)
    legacy.advance(0.5, legacy.inner, 0.0, 0.01)

    neutral = _membrane(relaxation_time=31.622776601683793)
    inner_speed = 1.0
    neutral.advance(
        0.5,
        neutral.inner + 0.5,
        np.nextafter(inner_speed, math.inf),
        inner_speed,
    )


def test_edge_state_requires_a_complete_explicit_conformation() -> None:
    incomplete = EdgeState(0.0, 1.0, 1.0, 1.0, log_c_rr=0.0)
    with pytest.raises(ValueError, match="all three"):
        incomplete.log_conformation()


def test_restore_rejects_relaxation_parameter_change() -> None:
    membrane = _membrane(relaxation_time=31.622776601683793)
    with pytest.raises(ValueError, match="disagrees"):
        FarFieldMembrane.from_arrays(membrane.to_arrays(), relaxation_time=10.0)
