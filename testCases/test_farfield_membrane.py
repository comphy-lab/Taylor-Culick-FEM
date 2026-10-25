"""Physics and contract tests for the thin-film far-field membrane."""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.farfield_membrane import EdgeState, FarFieldMembrane, graded_fractions


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


# ------------------------------------------------------------ graded node fractions
def _reference_uniform_rhs(membrane: FarFieldMembrane, t, y, t0, a0, a1, v0, v1, dt):
    """Frozen copy of the uniform ``_rhs`` (and ``_upwind``) before grading existed."""

    def upwind(f, a, dr):
        d = np.empty_like(f)
        fwd = np.empty_like(f); bwd = np.empty_like(f)
        fwd[:-2] = (-3 * f[:-2] + 4 * f[1:-1] - f[2:]) / (2 * dr)
        fwd[-2:] = (f[-1] - f[-2]) / dr
        bwd[2:] = (3 * f[2:] - 4 * f[1:-1] + f[:-2]) / (2 * dr)
        bwd[:2] = (f[1] - f[0]) / dr
        d[:] = np.where(a > 0, fwd, bwd)
        return d

    n = len(membrane.xi)
    u, h, Y = y[:n].copy(), y[n:2 * n], y[2 * n:]
    s = (t - t0) / dt
    a = a0 + s * (a1 - a0); adot = (a1 - a0) / dt
    u[0] = v0 + s * (v1 - v0)
    u[-1] = 0.0
    r = a + membrane.xi * (membrane.outer - a)
    dr = (membrane.outer - a) / (n - 1)
    w = (1.0 - membrane.xi) * adot
    G, mu = membrane.G, membrane.mu
    rm = 0.5 * (r[1:] + r[:-1]); um = 0.5 * (u[1:] + u[:-1])
    hm = 0.5 * (h[1:] + h[:-1]); Ym = 0.5 * (Y[1:] + Y[:-1])
    urm = (u[1:] - u[:-1]) / dr
    ltm = rm / Ym; lrm = 1.0 / (ltm * hm)
    Nrr = 2 * mu * hm * (2 * urm + um / rm) + G * hm * (lrm**2 - hm**2)
    ur = np.gradient(u, dr)
    lt = r / Y
    Ntt = 2 * mu * h * (ur + 2 * u / r) + G * h * (lt**2 - h**2)
    rel = w - u
    dudt = np.zeros(n); dhdt = np.zeros(n); dYdt = np.zeros(n)
    F = rm * Nrr
    dudt[1:-1] = rel[1:-1] * upwind(u, rel, dr)[1:-1] + (
        (F[1:] - F[:-1]) / dr / r[1:-1] - Ntt[1:-1] / r[1:-1]
    ) / h[1:-1]
    dudt[0] = (v1 - v0) / dt
    dhdt[:-1] = (rel * upwind(h, rel, dr) - h * (ur + u / r))[:-1]
    dYdt[:-1] = (rel * upwind(Y, rel, dr))[:-1]
    return np.concatenate([dudt, dhdt, dYdt])


def _random_state(membrane: FarFieldMembrane, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(membrane.xi)
    return np.concatenate([
        1e-2 * rng.standard_normal(n), 1.0 + 0.1 * rng.random(n),
        membrane.radii() * (1.0 + 0.01 * rng.random(n)),
    ])


def _graded(fractions: np.ndarray, **overrides) -> FarFieldMembrane:
    values = {
        "elastic_modulus": 1.0, "viscosity": math.sqrt(2.0),
        "inner": 50.0, "outer": 400.0, "node_fractions": fractions,
    }
    values.update(overrides)
    return FarFieldMembrane(**values)


def _pulse(t: float) -> float:
    return 1e-4 * math.sin(math.pi * min(t, 4.0) / 4.0) ** 2


def _drive_pulse(membrane: FarFieldMembrane, t_end: float, dt: float) -> None:
    while membrane.time < t_end - 1e-9:
        t0 = membrane.time
        membrane.advance(t0 + dt, membrane.inner, _pulse(t0), _pulse(t0 + dt))


def test_default_membrane_keeps_the_uniform_scheme_bit_for_bit() -> None:
    membrane = _membrane()
    assert not membrane.graded
    assert np.array_equal(membrane.xi, np.linspace(0.0, 1.0, 701))
    assert "graded" not in membrane.to_arrays()
    y = _random_state(membrane)
    for args in [(0.3, 0.0, 50.0, 50.7, 0.01, 0.03, 1.0), (0.1, 0.0, 50.0, 49.0, -0.1, 0.2, 0.5)]:
        assert np.array_equal(membrane._rhs(args[0], y, *args[1:]),
                              _reference_uniform_rhs(membrane, args[0], y, *args[1:]))


def test_graded_path_reduces_to_the_uniform_scheme_on_uniform_fractions() -> None:
    uniform = _membrane()
    explicit = _graded(np.linspace(0.0, 1.0, 701))
    assert explicit.graded
    y = _random_state(uniform)
    args = (0.3, 0.0, 50.0, 50.7, 0.01, 0.03, 1.0)
    reference = uniform._rhs(args[0], y, *args[1:])
    np.testing.assert_allclose(explicit._rhs(args[0], y, *args[1:]), reference,
                               rtol=1e-9, atol=1e-12 * np.max(np.abs(reference)))


def test_graded_fractions_layout() -> None:
    inner, outer, spacing, near, growth = 50.0, 2000.0, 0.4, 100.0, 1.04
    xi = graded_fractions(inner, outer, spacing, near, growth)
    assert xi[0] == 0.0 and xi[-1] == 1.0 and np.all(np.diff(xi) > 0)
    d = np.diff(inner + xi * (outer - inner))
    uniform_cells = math.ceil(near / spacing)
    np.testing.assert_allclose(d[:uniform_cells], spacing, rtol=1e-9)
    ratios = d[uniform_cells:] / d[uniform_cells - 1:-1]
    # One growth factor throughout the tail, just below the requested one.
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)
    assert 1.0 < ratios[0] <= growth and ratios[0] == pytest.approx(growth, rel=1e-2)
    # A production-sized domain stays small.
    long = graded_fractions(1000.0, 1e9, 0.5, 2000.0, 1.01)
    assert long[-1] == 1.0 and np.all(np.diff(long) > 0) and len(long) < 6000
    # A domain that the uniform part covers falls back to the ``extended`` layout.
    np.testing.assert_array_equal(graded_fractions(50.0, 60.0, 0.5, 100.0, 1.05),
                                  np.linspace(0.0, 1.0, 21))
    for bad in [(60.0, 50.0, 0.5, 1.0, 1.1), (50.0, 60.0, 0.0, 1.0, 1.1),
                (50.0, 60.0, 0.5, -1.0, 1.1), (50.0, 60.0, 0.5, 1.0, 1.0)]:
        with pytest.raises(ValueError):
            graded_fractions(*bad)


def test_node_fractions_are_validated() -> None:
    good = graded_fractions(50.0, 400.0, 0.5, 50.0, 1.05)
    for bad, match in [
        (good[:7] / good[6], "at least 8"),
        (good * 0.5, "exactly 0 to exactly 1"),
        (good + 1e-3, "exactly 0 to exactly 1"),
        (np.r_[good[:3], good[2], good[3:]], "exactly 0 to exactly 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            _graded(bad)
    with pytest.raises(ValueError, match="disagrees"):
        _graded(good, nodes=len(good) + 1)
    assert _graded(good, nodes=len(good)).graded
    with pytest.raises(ValueError, match="nodes or node_fractions"):
        FarFieldMembrane(elastic_modulus=1.0, viscosity=1.0, inner=50.0, outer=400.0)


def test_graded_upwind_is_exact_for_quadratics() -> None:
    r = 50.0 + 350.0 * graded_fractions(50.0, 400.0, 0.5, 20.0, 1.1)
    f, exact = 3.0 * r**2 - 2.0 * r + 1.0, 6.0 * r - 2.0
    d = np.diff(r)
    fwd = FarFieldMembrane._upwind_graded(f, np.ones_like(r), d)
    bwd = FarFieldMembrane._upwind_graded(f, -np.ones_like(r), d)
    np.testing.assert_allclose(fwd[:-2], exact[:-2], rtol=1e-8)
    np.testing.assert_allclose(bwd[2:], exact[2:], rtol=1e-8)
    # Endpoint fallbacks are the two-point differences of the uniform scheme.
    assert fwd[-1] == fwd[-2] == pytest.approx((f[-1] - f[-2]) / d[-1], rel=1e-12)
    assert bwd[0] == bwd[1] == pytest.approx((f[1] - f[0]) / d[0], rel=1e-12)
    # On a uniform grid it is the uniform scheme.
    ru = np.linspace(50.0, 400.0, 101)
    fu = np.sin(ru / 30.0)
    for sign in (1.0, -1.0):
        np.testing.assert_allclose(
            FarFieldMembrane._upwind_graded(fu, sign * np.ones(101), np.diff(ru)),
            FarFieldMembrane._upwind(fu, sign * np.ones(101), 3.5), rtol=1e-9, atol=1e-12)


def test_graded_rhs_converges_at_second_order() -> None:
    # Manufactured smooth state; the exact operator uses analytic u_r and
    # complex-step derivatives of the analytic h, Y and r N_rr.
    a0, b, G, mu = 50.0, 2000.0, 0.3, 0.7

    def u_f(r):
        return 0.05 * np.sin(2.0 * np.log(r / a0)) * (b - r) / (b - a0)

    def ur_f(r):
        return 0.05 * (2.0 * np.cos(2.0 * np.log(r / a0)) / r * (b - r)
                       - np.sin(2.0 * np.log(r / a0))) / (b - a0)

    def h_f(r):
        return 1.0 + 0.1 * np.cos(np.log(r / a0))

    def Y_f(r):
        return r * (1.0 + 0.05 * np.sin(1.5 * np.log(r / a0)))

    def derivative(f, r):
        return np.imag(f(r + 1e-20j)) / 1e-20

    def rNrr(r):
        h = h_f(r)
        return r * (2 * mu * h * (2 * ur_f(r) + u_f(r) / r) + G * h * ((Y_f(r) / (r * h))**2 - h**2))

    def errors(fractions, adot):
        m = FarFieldMembrane(elastic_modulus=G, viscosity=mu, inner=a0, outer=b,
                             node_fractions=fractions)
        r, n = m.radii(), len(fractions)
        u, h, Y, ur = u_f(r), h_f(r), Y_f(r), ur_f(r)
        rel = (1.0 - m.xi) * adot - u
        Ntt = 2 * mu * h * (ur + 2 * u / r) + G * h * ((r / Y)**2 - h**2)
        exact = (rel * ur + (derivative(rNrr, r) / r - Ntt / r) / h,
                 rel * derivative(h_f, r) - h * (ur + u / r),
                 rel * derivative(Y_f, r))
        y = np.concatenate([u, h, Y])
        y[n - 1] = 0.0
        out = m._rhs(0.0, y, 0.0, a0, a0 + adot, u[0], u[0], 1.0)
        interior = slice(2, n - 2)   # the endpoint rows use first-order fallbacks
        return np.array([np.max(np.abs(out[k * n:(k + 1) * n][interior] - exact[k][interior]))
                         for k in range(3)])

    # Refining the grid mapping: half the spacing, square root of the growth.
    levels = [(0.4, 1.04), (0.2, 1.04**0.5), (0.1, 1.04**0.25)]
    for adot in (0.5, -0.3):
        errs = [errors(graded_fractions(a0, b, s, 100.0, g), adot) for s, g in levels]
        for coarse, fine in itertools.pairwise(errs):
            assert np.all((coarse / fine > 3.5) & (coarse / fine < 4.5)), coarse / fine


def test_graded_jacobian_fits_the_sparsity_pattern() -> None:
    membrane = _graded(graded_fractions(50.0, 60.0, 0.5, 2.0, 1.3))
    n = len(membrane.xi)
    assert 8 <= n < 30
    y = _random_state(membrane, seed=5)
    args = (0.0, 50.0, 50.4, 0.02, 0.03, 0.5)
    base = membrane._rhs(0.2, y, *args)
    pattern = membrane._sparsity.toarray() != 0
    for j in range(3 * n):
        bumped = y.copy()
        bumped[j] += 1e-7 * max(1.0, abs(y[j]))
        column = membrane._rhs(0.2, bumped, *args) - base
        assert not np.any(column[~pattern[:, j]]), j


def test_graded_membrane_matches_uniform_where_the_grids_coincide() -> None:
    # The existing small-pulse setup, before the front reaches the graded part
    # (r = 100). Tight tolerances, so the time error is below the comparison.
    tight = {"rtol": 1e-9, "atol": 1e-13}
    uniform = _membrane(viscosity=1e-3, nodes=1401, **tight)
    graded = _graded(graded_fractions(50.0, 400.0, 0.25, 50.0, 1.02), viscosity=1e-3, **tight)
    assert len(graded.xi) < 400
    _drive_pulse(uniform, 20.0, 0.25)
    _drive_pulse(graded, 20.0, 0.25)
    near = graded.radii() <= 100.0
    np.testing.assert_allclose(graded.radii()[near], uniform.radii()[:near.sum()], rtol=0, atol=1e-11)
    amplitude = np.max(np.abs(uniform.u))
    assert amplitude > 5e-5
    assert np.max(np.abs(graded.u[near] - uniform.u[:near.sum()])) < 1e-5 * amplitude
    assert np.max(np.abs(graded.h[near] - uniform.h[:near.sum()])) < 1e-5 * np.max(np.abs(uniform.h - 1))


def test_graded_membrane_tracks_the_uniform_response_once_the_pulse_is_in_the_tail() -> None:
    # Production viscosity, default tolerances; by t = 40 the pulse peak is at
    # r ~ 128, past the uniform part (r <= 100).
    uniform = _membrane()
    graded = _graded(graded_fractions(50.0, 400.0, 0.5, 50.0, 1.02))
    _drive_pulse(uniform, 40.0, 0.5)
    _drive_pulse(graded, 40.0, 0.5)
    amplitude = np.max(np.abs(uniform.u))
    assert uniform.radii()[np.argmax(np.abs(uniform.u))] > 110.0
    near = graded.radii() <= 100.0
    assert np.max(np.abs(graded.u[near] - uniform.u[:near.sum()])) < 1e-3 * amplitude
    everywhere = np.interp(graded.radii(), uniform.radii(), uniform.u)
    assert np.max(np.abs(graded.u - everywhere)) < 1e-2 * amplitude


def test_a_small_pulse_travels_at_the_wave_speed_on_a_graded_grid() -> None:
    G, probe = 1.0, 150.0
    membrane = _graded(graded_fractions(50.0, 400.0, 0.25, 50.0, 1.02),
                       elastic_modulus=G, viscosity=1e-3)
    r = membrane.radii()
    assert np.interp(probe, r[1:], np.diff(r)) > 4 * 0.25   # the probe sits in the coarse tail
    arrival, dt = None, 0.25
    while membrane.time < 80.0 and arrival is None:
        t0 = membrane.time
        membrane.advance(t0 + dt, membrane.inner, _pulse(t0), _pulse(t0 + dt))
        if abs(membrane.state_at(probe).velocity) > 1e-5:
            arrival = membrane.time
    assert arrival is not None
    speed = (probe - membrane.inner) / (arrival - 2.0)
    assert speed == pytest.approx(2.0 * math.sqrt(G), rel=0.1)


def test_graded_state_round_trips_through_arrays_and_files(tmp_path: Path) -> None:
    membrane = _graded(graded_fractions(50.0, 400.0, 0.5, 50.0, 1.03))
    membrane.advance(1.0, 51.0, 0.0, 0.01)
    data = membrane.to_arrays()
    assert bool(data["graded"])
    path = tmp_path / "far.npz"
    with path.open("xb") as stream:
        np.savez(stream, **data)
    with np.load(path) as stored:
        from_file = FarFieldMembrane.from_arrays(stored)
    copies = [FarFieldMembrane.from_arrays(data), from_file]
    # A state stored without the flag is still recognised by its fractions.
    copies.append(FarFieldMembrane.from_arrays({k: v for k, v in data.items() if k != "graded"}))
    for copy in copies:
        assert copy.graded and np.array_equal(copy.xi, membrane.xi)
        assert (copy.time, copy.inner, copy.outer) == (membrane.time, membrane.inner, membrane.outer)
        for name in ("u", "h", "Y"):
            np.testing.assert_array_equal(getattr(copy, name), getattr(membrane, name))
    membrane.advance(1.5, 51.5, 0.01, 0.02)
    for copy in copies:
        copy.advance(1.5, 51.5, 0.01, 0.02)
        for name in ("u", "h", "Y"):
            np.testing.assert_array_equal(getattr(copy, name), getattr(membrane, name))
    # Uniform fractions flagged as graded stay graded; a uniform state stays uniform.
    flagged = _graded(np.linspace(0.0, 1.0, 64))
    assert FarFieldMembrane.from_arrays(flagged.to_arrays()).graded
    assert not FarFieldMembrane.from_arrays(_membrane().to_arrays()).graded


def test_extended_graded_membrane_pads_undisturbed_film_and_keeps_the_state() -> None:
    membrane = _membrane(nodes=351)
    membrane.advance(1.0, 50.5, 0.0, 0.02)
    longer = membrane.extended_graded(outer=20000.0, spacing=1.0, near_length=400.0, growth=1.03)
    assert longer.graded and len(longer.xi) < 1000
    assert longer.outer == 20000.0 and longer.time == membrane.time
    assert longer.inner == membrane.inner
    for radius in (60.0, 120.0, 300.0):
        old, new = membrane.state_at(radius), longer.state_at(radius)
        assert new.velocity == pytest.approx(old.velocity, abs=1e-6)
        assert new.thickness == pytest.approx(old.thickness, abs=1e-6)
    beyond = longer.radii() > membrane.outer
    assert np.all(longer.u[beyond] == 0.0) and np.all(longer.h[beyond] == 1.0)
    np.testing.assert_array_equal(longer.Y[beyond], longer.radii()[beyond])
    assert longer.state_at(5000.0).log_conformation() == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    longer.advance(1.5, 51.0, 0.02, 0.03)
    # A graded membrane extends again; the new end must lie beyond the old one.
    again = longer.extended_graded(outer=1e6, spacing=1.0, near_length=400.0, growth=1.03)
    assert again.graded and again.time == longer.time
    np.testing.assert_allclose(again.state_at(200.0).velocity, longer.state_at(200.0).velocity,
                               rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="must exceed"):
        again.extended_graded(outer=again.outer, spacing=1.0, near_length=400.0, growth=1.03)
