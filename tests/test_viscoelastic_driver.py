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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

from run_viscoelastic_axi import (
    ArrestTracker,
    HeightPlateauTracker,
    _output_times,
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


def test_zero_modulus_requires_explicit_passive_conformation(tmp_path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--results", str(tmp_path), "--elastic-modulus", "0"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "--elastic-modulus=0 requires --passive-conformation" in capsys.readouterr().err

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
