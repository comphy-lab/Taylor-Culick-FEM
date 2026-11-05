"""Software tests for the viscoelastic arrest-run control logic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taylor_culick_fem.viscoelastic_driver import (
    ArrestTracker,
    HeightPlateauTracker,
    MembraneCoupling,
    RelaxingMembraneCoupling,
    _characteristic_valid,
    _check_extension_not_shorter,
    _constitutive_metadata,
    _interval_step_bounds,
    _output_times,
    _without_extension,
    _zero_crossing,
    build_parser,
    main,
    validate_args,
)


def test_output_step_includes_a_short_final_interval() -> None:
    args = argparse.Namespace(t_end=1.0, output_step=0.3, n_outputs=2)
    np.testing.assert_allclose(_output_times(args), [0.3, 0.6, 0.9, 1.0])


def test_output_count_is_used_without_an_output_step() -> None:
    args = argparse.Namespace(t_end=1.0, output_step=None, n_outputs=4)
    np.testing.assert_allclose(_output_times(args), [0.25, 0.5, 0.75, 1.0])


def test_zero_crossing_interpolates_time_and_radius() -> None:
    crossing = _zero_crossing(
        [2.0, 3.0, 0.2, 0, 0, 0, 0, 0],
        [3.0, 3.3, -0.1, 0, 0, 0, 0, 0],
    )
    assert crossing["time"] == pytest.approx(2.0 + 2.0 / 3.0)
    assert crossing["radius"] == pytest.approx(3.2)
    assert crossing["bracket_start"] == 2.0
    assert crossing["bracket_end"] == 3.0
    assert crossing["time_sampling_uncertainty"] == 0.5
    assert crossing["radius_sampling_uncertainty"] == pytest.approx(0.15)


def _tracker() -> ArrestTracker:
    return ArrestTracker(
        opening_ustar=0.02,
        minimum_radius_increase=1e-3,
        arrest_ustar=1e-3,
        consecutive_samples=3,
        trend_tolerance=1e-4,
        radius_span_tolerance=1e-3,
    )


def test_startup_recoil_cannot_trigger_arrest() -> None:
    tracker = _tracker()
    assert tracker.observe([0.05, 0.999, -0.01, -0.007, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.10, 1.0005, 0.01, 0.007, 0, 0, 0, 0], 1.0) is None
    assert not tracker.opening_established
    assert tracker.observe([0.15, 1.002, 0.04, 0.028, 0, 0, 0, 0], 1.0) is None
    arrest = tracker.observe([0.20, 1.003, -0.01, -0.007, 0, 0, 0, 0], 1.0)
    assert arrest is not None
    assert arrest["kind"] == "sampled_zero_crossing"


def test_near_zero_rebound_fails_trend_and_radius_plateau() -> None:
    tracker = _tracker()
    assert tracker.observe([0.1, 1.01, 0.04, 0.03, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.2, 1.0100, 0.0010, 0.0007, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.3, 1.0108, 0.0011, 0.0008, 0, 0, 0, 0], 1.0) is None
    # All three speeds are below threshold, but both the fitted trend and
    # radius span indicate continued motion rather than arrest.
    assert tracker.observe([0.4, 1.0116, 0.0013, 0.0009, 0, 0, 0, 0], 1.0) is None


def test_steep_negative_near_zero_trend_is_not_asymptotic_arrest() -> None:
    tracker = _tracker()
    assert tracker.observe([0.1, 1.01, 0.04, 0.03, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.2, 1.0110, 0.0012, 0.0009, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.3, 1.0111, 0.0008, 0.0005, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.4, 1.0112, 0.0004, 0.0001, 0, 0, 0, 0], 1.0) is None


def test_persistent_near_zero_plateau_is_certified() -> None:
    tracker = _tracker()
    assert tracker.observe([0.1, 1.01, 0.04, 0.03, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.2, 1.0110, 0.0012, 0.0008, 0, 0, 0, 0], 1.0) is None
    assert tracker.observe([0.3, 1.0112, 0.0011, 0.000805, 0, 0, 0, 0], 1.0) is None
    arrest = tracker.observe([0.4, 1.0113, 0.0010, 0.00081, 0, 0, 0, 0], 1.0)
    assert arrest is not None
    assert arrest["kind"] == "threshold_certified_asymptotic"
    assert arrest["radius"] == pytest.approx((1.0110 + 1.0112 + 1.0113) / 3)


def test_arrest_trend_requires_at_least_two_samples(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_viscoelastic_axi.py",
            "--results",
            str(tmp_path),
            "--arrest-consecutive",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "--arrest-consecutive must be at least 2" in capsys.readouterr().err


def test_allow_recoil_is_an_explicit_fixed_horizon_option(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--results", str(tmp_path), "--allow-recoil"])
    validate_args(parser, args)
    assert args.allow_recoil
    assert not args.stop_when_arrested


def test_zero_modulus_defaults_to_exact_newtonian(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--results", str(tmp_path), "--elastic-modulus", "0"])
    validate_args(parser, args)
    assert args.passive_conformation is False
    assert _constitutive_metadata(0.0, args.relaxation_time, False) == {
        "constitutive_model": "Newtonian",
        "polymer_stress": None,
        "conformation_formulation": None,
    }

    passive = parser.parse_args(
        [
            "--results",
            str(tmp_path),
            "--elastic-modulus",
            "0",
            "--passive-conformation",
        ]
    )
    validate_args(parser, passive)
    assert passive.passive_conformation is True
    assert _constitutive_metadata(0.0, passive.relaxation_time, True)[
        "constitutive_model"
    ] == "Oldroyd-B"


def test_zero_relaxation_cli_is_newtonian(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--results", str(tmp_path), "--relaxation-time", "0"])
    validate_args(parser, args)
    assert args.relaxation_time == 0.0
    assert _constitutive_metadata(1.0, 0.0, False)["constitutive_model"] == "Newtonian"


@pytest.mark.parametrize("value", ["-1", "nan"])
def test_invalid_relaxation_cli_is_rejected(value: str, tmp_path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--results", str(tmp_path), "--relaxation-time", value])


def _step_args(tmp_path, *extra):
    parser = build_parser()
    args = parser.parse_args(["--results", str(tmp_path), "--max-step", "0.01", *extra])
    validate_args(parser, args)
    return args


def test_relaxation_start_cap_is_opt_in_and_caps_only_evolved_finite_relaxation(tmp_path) -> None:
    finite = ["--relaxation-time", "1e-3"]
    # Without the option the first step of an interval is unchanged (1% of it).
    assert _interval_step_bounds(_step_args(tmp_path, *finite), 0.0, 0.5, None) \
        == pytest.approx((0.01, 0.005))
    capped = _step_args(tmp_path, *finite, "--relaxation-start-fraction", "0.1")
    assert _interval_step_bounds(capped, 0.0, 0.5, None) == pytest.approx((0.01, 1e-4))
    for endpoint in (["--relaxation-time", "inf"], ["--relaxation-time", "0"],
                     [*finite, "--elastic-modulus", "0"]):
        args = _step_args(tmp_path, *endpoint, "--relaxation-start-fraction", "0.1")
        assert _interval_step_bounds(args, 0.0, 0.5, None) == pytest.approx((0.01, 0.005))


@pytest.mark.parametrize("value", ["0", "1.5"])
def test_relaxation_start_fraction_outside_the_unit_interval_is_refused(value, tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(
            ["--results", str(tmp_path), "--relaxation-start-fraction", value]))


def test_characteristic_gate_accepts_only_roundoff_neutrality():
    assert _characteristic_valid(0.0, 0.0, 0.0)
    assert _characteristic_valid(-np.finfo(float).eps, 0.0, 0.0)
    assert not _characteristic_valid(-1e-10, 0.0, 0.0)
    # Positive rim velocity alone is insufficient if edge material is faster.
    assert not _characteristic_valid(0.1 - 0.2, 0.1, 0.2)


def test_relaxing_coupling_rejects_inner_material_inflow_before_advance():
    class Membrane:
        time, inner = 0.0, 5.0

        def advance(self, *_):
            pytest.fail("invalid characteristic must reject before advancing")

    coupling = RelaxingMembraneCoupling(Membrane(), inner_fraction=0.5, length=8.0)
    with pytest.raises(RuntimeError, match="inner material outflow validity rejected"):
        coupling.before_interval(0.0, 0.1, radius=1.0, speed=0.1, drive=0.2)


def test_relaxing_outflow_is_half_sheet_full_thickness_identity():
    from types import SimpleNamespace

    coupling = RelaxingMembraneCoupling(SimpleNamespace(time=0.0), inner_fraction=0.5, length=8.0)
    coupling.predicted_edge_radius = 9.1
    edge = SimpleNamespace(thickness=0.8, velocity=0.1)
    coupling.after_interval(edge, 0.1, 9.0, 9.1, 1.0, 1.0, 0.1)
    # The half-sheet factor pi belongs in expected_volume, not this integral.
    assert coupling.outflow == pytest.approx(0.1 * 9.05 * (0.8 * 0.1 - (0.8 - 1.0) * 1.0))


def test_relaxing_sidecar_roundtrip_binds_checkpoint_and_finite_relaxation(tmp_path):
    import hashlib

    from taylor_culick_fem.relaxing_farfield_membrane import RelaxingFarFieldMembrane

    def make(relaxation_time=1.0):
        return RelaxingMembraneCoupling(
            RelaxingFarFieldMembrane(
                elastic_modulus=1.0, relaxation_time=relaxation_time, viscosity=1.0,
                inner=5.0, outer=20.0, nodes=31,
            ), inner_fraction=0.5, length=8.0,
        )

    original = make()
    # A homogeneous exterior is a deterministic zero state; this exercises
    # the sidecar contract without running a native FEM solve.
    edge = original.before_interval(0.0, 0.1, 1.0, 0.0, 0.0)
    original.after_interval(edge, 0.1, 9.0, 9.0, 0.0, 0.0, 0.1)
    checkpoint = tmp_path / "ckpt.dump"
    checkpoint.write_bytes(b"selected FEM state")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = tmp_path / "ckpt.farfield.npz"
    original.save(sidecar, checkpoint)
    restored = make()
    restored.load(sidecar, digest)
    assert restored.membrane.relaxation_time == 1.0
    assert restored.previous == original.previous
    assert restored.outflow == original.outflow
    for name in ("u", "h", "log_c_rr", "log_c_zz", "log_c_th"):
        np.testing.assert_array_equal(getattr(restored.membrane, name), getattr(original.membrane, name))
    with pytest.raises(ValueError, match="does not bind"):
        restored.load(sidecar, "0" * 64)
    with pytest.raises(ValueError, match="constitutive identity"):
        make(2.0).load(sidecar, digest)


def test_relaxing_extension_may_grow_but_never_shrink():
    parent = {"mode": "relaxing_membrane", "inner_fraction": 0.5, "extension": 100.0,
              "node_spacing": 0.5}
    _check_extension_not_shorter({**parent, "extension": 200.0}, parent)
    with pytest.raises(ValueError, match="never shorten"):
        _check_extension_not_shorter({**parent, "extension": 50.0}, parent)
    assert _without_extension(parent) == {"mode": "relaxing_membrane", "inner_fraction": 0.5,
                                          "node_spacing": 0.5}


def test_tip_arrest_requires_a_field_at_every_scalar_sample(tmp_path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--results",
            str(tmp_path),
            "--stop-when-arrested",
            "--field-stride",
            "2",
        ]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "tip-arrest completion requires --field-stride=1" in capsys.readouterr().err


def _height_certificate(function) -> dict | None:
    tracker = HeightPlateauTracker()
    certificate = None
    for time in np.arange(0.5, 100.5, 0.5):
        certificate = tracker.observe(float(time), float(function(time)))
        if certificate is not None:
            break
    return certificate


def test_height_plateau_requires_the_full_ten_time_unit_hold() -> None:
    certificate = _height_certificate(lambda _time: 2.0)
    assert certificate is not None
    assert certificate["held_for"] >= 10.0
    assert certificate["certified_at"] - certificate["candidate_since"] >= 10.0
    json.dumps(certificate, allow_nan=False)


@pytest.mark.parametrize(
    "height",
    [
        lambda time: np.sqrt(time + 1.0),
        lambda time: 2.0 + 0.01 * time,
        lambda time: 2.0 + 5e-5 * time * np.sin(0.5 * np.pi * time),
    ],
    ids=["sqrt", "linear-drift", "growing-oscillation"],
)
def test_non_saturated_height_histories_do_not_pass(height) -> None:
    assert _height_certificate(height) is None


class _StubMembrane:
    """Stands in for the far-field solver: records calls, returns a fixed edge."""

    def __init__(self, edge):
        self.edge, self.calls = edge, []
        self.inner, self.outer = 10.0, 100.0
        self.u = np.zeros(11)

    def radii(self):
        return np.linspace(self.inner, self.outer, 11)

    def advance(self, t1, inner1, drive0, drive1, watch=None):
        self.calls.append((t1, inner1, drive0, drive1))
        self.watched_peak = float(np.max(np.abs(self.u[watch]))) if watch is not None else 0.0

    def state_at(self, radius):
        self.calls.append(("state", radius))
        return self.edge


def test_membrane_coupling_extrapolates_the_drive_and_predicts_the_edge() -> None:
    from taylor_culick_fem.farfield_membrane import EdgeState

    edge = EdgeState(0.1, 1.0, 1.0, 1.0)
    stub = _StubMembrane(edge)
    coupling = MembraneCoupling(stub, inner_fraction=0.5, length=40.0)
    assert coupling.before_interval(0.0, 0.5, 5.0, 2.0, 0.2) is edge
    assert stub.calls[0] == (0.5, 5.0 + 1.0 + 20.0, 0.2, 0.2)
    assert stub.calls[1] == ("state", 5.0 + 1.0 + 40.0)
    coupling.before_interval(0.5, 1.0, 6.0, 2.0, 0.3)
    # Linear extrapolation of the drive from (0, 0.2) and (0.5, 0.3).
    assert stub.calls[2][3] == pytest.approx(0.4)


def test_membrane_coupling_outflow_is_the_edge_volume_correction() -> None:
    from taylor_culick_fem.farfield_membrane import EdgeState

    coupling = MembraneCoupling(_StubMembrane(None), inner_fraction=0.5, length=40.0)
    coupling.after_interval(EdgeState(0.0, 1.0, 1.0, 1.0), 0.5, 50.0, 51.0, 2.0, 2.0, 0.5)
    assert coupling.outflow == 0.0
    # Film moving at u_e with unit thickness: outflow is the integral of r u_e.
    coupling.after_interval(EdgeState(0.1, 1.0, 1.0, 1.0), 1.0, 50.0, 51.0, 2.0, 2.0, 0.5)
    assert coupling.outflow == pytest.approx(0.5 * 0.5 * (50.0 + 51.0) * 0.1)
    # A thicker film at rest swallowed at Xdot = 2 adds (h - 1) Xdot of inflow.
    thick = MembraneCoupling(_StubMembrane(None), inner_fraction=0.5, length=40.0)
    thick.after_interval(EdgeState(0.0, 1.1, 1.0, 1.0), 1.0, 50.0, 50.0, 2.0, 2.0, 1.0)
    assert thick.outflow == pytest.approx(-50.0 * 0.1 * 2.0)


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--far-field-extension", "100"], "require --far-field=membrane"),
        (["--far-field", "membrane"], "explicit --far-field-extension"),
        (["--far-field", "membrane", "--far-field-extension", "100",
          "--far-field-inner-fraction", "1.5"], "strictly inside the window"),
    ],
)
def test_far_field_membrane_settings_are_validated(tmp_path, capsys, extra, message) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--results", str(tmp_path), "--oh", "1", "--elastic-modulus", "1", *extra]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert message in capsys.readouterr().err


def test_restart_identity_ignores_but_bounds_the_membrane_extension() -> None:
    parent = {"mode": "membrane", "inner_fraction": 0.5, "extension": 1000.0, "node_spacing": 0.5}
    longer = {**parent, "extension": 2000.0}
    assert _without_extension(parent) == _without_extension(longer)
    _check_extension_not_shorter(longer, parent)
    _check_extension_not_shorter(parent, parent)
    with pytest.raises(ValueError, match="never shorten"):
        _check_extension_not_shorter(parent, longer)
    assert _without_extension({"mode": "rest"}) == {"mode": "rest"}
    # Anything other than the extension must still agree.
    assert _without_extension({**parent, "inner_fraction": 0.4}) != _without_extension(parent)
