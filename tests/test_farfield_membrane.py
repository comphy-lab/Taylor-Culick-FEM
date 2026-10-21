"""Physics and contract tests for the thin-film far-field membrane."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.farfield_membrane import EdgeState, FarFieldMembrane


def _membrane(**overrides) -> FarFieldMembrane:
    values = {
        "elastic_modulus": 1.0, "viscosity": math.sqrt(2.0),
        "inner": 50.0, "outer": 400.0, "nodes": 701,
    }
    values.update(overrides)
    return FarFieldMembrane(**values)


def test_undriven_film_stays_at_rest() -> None:
    membrane = _membrane()
    for step in range(4):
        membrane.advance(0.5 * (step + 1), membrane.inner + 0.5, 0.0, 0.0)
    assert np.max(np.abs(membrane.u)) < 1e-12
    assert np.max(np.abs(membrane.h - 1.0)) < 1e-12
    assert np.max(np.abs(membrane.Y - membrane.radii())) < 1e-9
    state = membrane.state_at(200.0)
    assert state.log_conformation() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)


def test_a_small_pulse_travels_at_the_dilatational_wave_speed() -> None:
    # Weakly damped, small amplitude: the linear front moves at 2 sqrt(G).
    G = 1.0
    membrane = _membrane(elastic_modulus=G, viscosity=1e-3, nodes=1401)
    probe, arrival, dt = 150.0, None, 0.25
    while membrane.time < 80.0 and arrival is None:
        t0 = membrane.time
        drive = lambda t: 1e-4 * math.sin(math.pi * min(t, 4.0) / 4.0) ** 2
        membrane.advance(t0 + dt, membrane.inner, drive(t0), drive(t0 + dt))
        if abs(membrane.state_at(probe).velocity) > 1e-5:
            arrival = membrane.time
    assert arrival is not None
    # Half-amplitude point of the leading edge sits ~2 time units behind t = 0.
    speed = (probe - membrane.inner) / (arrival - 2.0)
    assert speed == pytest.approx(2.0 * math.sqrt(G), rel=0.1)


def test_state_round_trips_through_arrays() -> None:
    membrane = _membrane()
    membrane.advance(1.0, 51.0, 0.0, 0.01)
    copy = FarFieldMembrane.from_arrays(membrane.to_arrays())
    for name in ("u", "h", "Y"):
        np.testing.assert_array_equal(getattr(copy, name), getattr(membrane, name))
    assert (copy.time, copy.inner, copy.outer) == (membrane.time, membrane.inner, membrane.outer)
    copy.advance(1.5, 51.5, 0.01, 0.02)
    membrane.advance(1.5, 51.5, 0.01, 0.02)
    np.testing.assert_allclose(copy.u, membrane.u, rtol=0, atol=1e-14)


def test_edge_conformation_is_incompressible() -> None:
    state = EdgeState(velocity=0.1, thickness=1.03, stretch_r=1 / (1.03 * 1.2), stretch_theta=1.2)
    assert sum(state.log_conformation()) == pytest.approx(0.0, abs=1e-14)


def test_domain_and_time_are_validated() -> None:
    with pytest.raises(ValueError, match="inner < outer"):
        _membrane(inner=500.0)
    membrane = _membrane()
    with pytest.raises(ValueError, match="later time"):
        membrane.advance(0.0, 50.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="outside"):
        membrane.state_at(10.0)


def test_extended_membrane_pads_undisturbed_film_and_keeps_the_state() -> None:
    membrane = _membrane(nodes=351)
    membrane.advance(1.0, 50.5, 0.0, 0.02)
    longer = membrane.extended(outer=600.0, node_spacing=1.0)
    assert longer.outer == 600.0 and longer.time == membrane.time
    assert longer.inner == membrane.inner
    # Inside the old domain the state is reproduced (linear interpolation).
    for radius in (60.0, 120.0, 300.0):
        old, new = membrane.state_at(radius), longer.state_at(radius)
        assert new.velocity == pytest.approx(old.velocity, abs=1e-6)
        assert new.thickness == pytest.approx(old.thickness, abs=1e-6)
    # Beyond the old end the film is at rest, unstretched.
    beyond = longer.state_at(500.0)
    assert (beyond.velocity, beyond.thickness) == (0.0, 1.0)
    assert beyond.log_conformation() == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    longer.advance(1.5, 51.0, 0.02, 0.03)


def test_extending_needs_a_longer_domain() -> None:
    membrane = _membrane()
    with pytest.raises(ValueError, match="must exceed"):
        membrane.extended(outer=membrane.outer, node_spacing=0.5)
    with pytest.raises(ValueError, match="spacing"):
        membrane.extended(outer=membrane.outer + 10, node_spacing=0.0)


def _advance_with_solve_ivp(membrane: FarFieldMembrane, t1: float, inner1: float,
                            drive0: float, drive1: float) -> None:
    """The advance as it was before it held the BDF solver itself."""

    from scipy.integrate import solve_ivp

    dt = t1 - membrane.time
    y0 = np.concatenate([membrane.u, membrane.h, membrane.Y])
    y0[0] = drive0
    sol = solve_ivp(
        membrane._rhs, (membrane.time, t1), y0, method="BDF",
        args=(membrane.time, membrane.inner, inner1, drive0, drive1, dt),
        jac_sparsity=membrane._sparsity, rtol=membrane.rtol, atol=membrane.atol,
    )
    assert sol.success
    n = len(membrane.xi)
    yend = sol.y[:, -1]
    membrane.u, membrane.h, membrane.Y = yend[:n].copy(), yend[n:2 * n].copy(), yend[2 * n:].copy()
    membrane.u[0], membrane.u[-1] = drive1, 0.0
    membrane.time, membrane.inner = float(t1), float(inner1)


def test_advance_takes_the_same_steps_as_solve_ivp() -> None:
    held, reference = _membrane(), _membrane()
    for step in range(5):
        t1, inner1 = 0.25 * (step + 1), 50.0 - 0.01 * (step + 1)
        drive0, drive1 = 0.02 * step, 0.02 * (step + 1)
        held.advance(t1, inner1, drive0, drive1)
        _advance_with_solve_ivp(reference, t1, inner1, drive0, drive1)
        for field in ("u", "h", "Y"):
            assert np.array_equal(getattr(held, field), getattr(reference, field))
    assert np.max(np.abs(held.u)) > 1e-3  # the comparison saw a driven film


def test_advance_leaves_no_garbage_cycle() -> None:
    # The solver's Jacobian and LU must go when the advance returns, not wait for a
    # full garbage collection: a long membrane would otherwise pile up memory.
    import gc

    membrane = _membrane()
    membrane.advance(0.25, 50.0, 0.0, 0.02)
    enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        membrane.advance(0.5, 50.0, 0.02, 0.04)
        assert gc.collect() == 0
    finally:
        if enabled:
            gc.enable()
