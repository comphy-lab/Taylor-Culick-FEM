"""Software-contract tests for the viscoelastic video renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[1] / "postProcess" / "animate_viscoelastic_comoving.py"
_SPEC = importlib.util.spec_from_file_location("animate_viscoelastic_comoving", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
video = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(video)


def test_axisymmetric_affine_dissipation_includes_hoop_term() -> None:
    points = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    subtris = np.array([[0, 1, 2]])
    a, b, viscosity = 0.4, -0.3, 1.7
    velocity = np.column_stack([a * points[:, 0], b * points[:, 1]])

    result = video.axisymmetric_dissipation(
        points, subtris, velocity, viscosity=viscosity
    )

    expected = 2.0 * viscosity * (2.0 * a**2 + b**2)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("fps", [24, 30])
def test_playback_holds_are_specified_in_seconds(fps: int) -> None:
    assert video.playback_repeats(
        0.0,
        fps=fps,
        t0_hold_seconds=1.0,
        early_hold_seconds=0.5,
        early_time_limit=0.2,
    ) == fps
    assert video.playback_repeats(
        0.15,
        fps=fps,
        t0_hold_seconds=1.0,
        early_hold_seconds=0.5,
        early_time_limit=0.2,
    ) == round(0.5 * fps)
    assert video.playback_repeats(
        0.200000095,
        fps=fps,
        t0_hold_seconds=1.0,
        early_hold_seconds=0.5,
        early_time_limit=0.2,
    ) == round(0.5 * fps)
    assert video.playback_repeats(
        0.25,
        fps=fps,
        t0_hold_seconds=1.0,
        early_hold_seconds=0.5,
        early_time_limit=0.2,
    ) == 1


def test_accepted_source_schedule_encodes_553_frames() -> None:
    times = [0.0, 0.05, 0.10, 0.15, 0.20, *np.linspace(0.25, 23.35, 463)]
    encoded = sum(
        video.playback_repeats(
            time,
            fps=30,
            t0_hold_seconds=1.0,
            early_hold_seconds=0.5,
            early_time_limit=0.2,
        )
        for time in times
    )
    assert len(times) == 468
    assert encoded == 553


def test_case_metadata_validates_project_conventions(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "axisymmetric",
                "velocity_frame": "laboratory",
                "time_scale": "capillary",
                "constitutive_model": "Oldroyd-B",
                "Oh": 1.0,
                "mu_s": np.sqrt(2.0),
                "elastic_modulus": 1.0,
                "relaxation_time": "inf",
                "V_TC": np.sqrt(2.0),
                "r0": 1.0,
            }
        )
    )

    metadata = video.load_case_metadata(path)

    assert metadata["elastic_modulus"] == 1.0
    assert metadata["relaxation_time"] == "inf"


def test_case_metadata_rejects_wrong_velocity_frame(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "axisymmetric",
                "velocity_frame": "comoving",
                "time_scale": "capillary",
                "constitutive_model": "Oldroyd-B",
                "Oh": 1.0,
                "mu_s": np.sqrt(2.0),
                "elastic_modulus": 1.0,
                "relaxation_time": "inf",
                "V_TC": np.sqrt(2.0),
                "r0": 1.0,
            }
        )
    )

    with pytest.raises(ValueError, match="velocity_frame"):
        video.load_case_metadata(path)
