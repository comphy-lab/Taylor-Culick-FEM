"""Software-contract tests for the viscoelastic video renderer."""

from __future__ import annotations

import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[1] / "postProcess" / "animate_viscoelastic_comoving.py"
_SPEC = importlib.util.spec_from_file_location("animate_viscoelastic_comoving", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
video = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(video)


def _write_pvd(path: Path, times: list[float]) -> None:
    datasets = []
    for index, time in enumerate(times):
        frame = path.parent / f"{path.stem}-{index}.vtu"
        frame.touch()
        datasets.append(
            f'<DataSet timestep="{time}" file="{frame.name}"/>'
        )
    path.write_text(
        "<VTKFile><Collection>" + "".join(datasets) + "</Collection></VTKFile>"
    )


def test_pvd_segments_concatenate_in_explicit_order(tmp_path: Path) -> None:
    first = tmp_path / "first.pvd"
    second = tmp_path / "second.pvd"
    _write_pvd(first, [0.05, 0.10])
    _write_pvd(second, [0.15, 0.20])

    entries = video.concatenate_pvd_segments([first, second])

    assert [time for time, _ in entries] == [0.05, 0.10, 0.15, 0.20]


@pytest.mark.parametrize("second_times", [[0.10, 0.15], [0.09, 0.15]])
def test_pvd_segments_reject_duplicate_or_overlapping_times(
    tmp_path: Path, second_times: list[float]
) -> None:
    first = tmp_path / "first.pvd"
    second = tmp_path / "second.pvd"
    _write_pvd(first, [0.05, 0.10])
    _write_pvd(second, second_times)

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        video.concatenate_pvd_segments([first, second])


def test_pvd_segment_rejects_out_of_order_entries(tmp_path: Path) -> None:
    path = tmp_path / "segment.pvd"
    _write_pvd(path, [0.10, 0.05])

    with pytest.raises(ValueError, match="strictly increasing"):
        video.concatenate_pvd_segments([path])


def test_pvd_segment_rejects_missing_vtu(tmp_path: Path) -> None:
    path = tmp_path / "segment.pvd"
    _write_pvd(path, [0.05, 0.10])
    (tmp_path / "segment-1.vtu").unlink()

    with pytest.raises(FileNotFoundError, match="missing frame"):
        video.concatenate_pvd_segments([path])


def test_pvd_receipt_records_every_ordered_source_and_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.pvd"
    second = tmp_path / "second.pvd"
    _write_pvd(first, [0.05])
    _write_pvd(second, [0.10])

    receipt = video.pvd_source_receipt([first, second])

    assert [item["path"] for item in receipt] == [str(first), str(second)]
    assert [item["sha256"] for item in receipt] == [
        video.sha256(first),
        video.sha256(second),
    ]


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


def test_kinematic_field_uses_requested_speed_limits() -> None:
    points = np.array([[0.1, 0.0], [1.0, 0.0], [0.1, 0.5]])
    subtris = np.array([[0, 1, 2]])
    figure, axis = video.plt.subplots()
    try:
        speed_map, _ = video.plot_kinematic_fields(
            axis,
            points,
            subtris,
            np.array([0.0, 0.5, 1.0]),
            np.full(3, 1e-3),
            speed_limits=(0.0, 1.0),
        )
        assert speed_map.norm.vmin == 0.0
        assert speed_map.norm.vmax == 1.0
    finally:
        video.plt.close(figure)


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


def test_stride_selection_keeps_early_states_and_endpoint() -> None:
    entries = [(0.05 * index, Path(f"frame-{index}")) for index in range(101)]
    entries[4] = (0.200000095, entries[4][1])

    selected = video.stride_selection(entries, stride=8, early_time_limit=0.2)

    indices = [index for index, _ in selected]
    assert indices[:6] == [0, 1, 2, 3, 4, 5]
    assert indices[-1] == 100
    assert len(indices) == len(set(indices))
    assert all((right - left) == 8 for left, right in pairwise(indices[5:-1]))


def test_unit_stride_keeps_every_frame() -> None:
    entries = [(float(index), Path(f"frame-{index}")) for index in range(5)]
    assert video.stride_selection(entries, stride=1, early_time_limit=0.2) == list(
        enumerate(entries)
    )


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
