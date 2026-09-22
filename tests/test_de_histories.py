"""Focused checks for manifest-driven Deborah-number history loading."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postProcess"))

from plot_de_histories import load_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, *, columns: dict[str, int] | None = None) -> dict:
    return {
        "label": "case",
        "Ec": 1.0,
        "De": 1.0,
        "Oh": 1.0,
        "data": path.name,
        "sha256": _sha256(path),
        "columns": columns or {"time": 0, "radius": 1, "velocity": 2},
        "velocity_divisor": 2.0,
    }


def _manifest(tmp_path: Path, entries: list[dict], *, t_max: float = 1.5) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "de-history-comparison-v1",
                "t_max": t_max,
                "histories": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_declared_columns_normalises_and_interpolates(tmp_path: Path) -> None:
    source = tmp_path / "history.txt"
    np.savetxt(source, [[10.0, 0.5, 1.5, 2.0], [12.0, 1.0, 2.0, 4.0], [14.0, 2.0, 3.0, 6.0]])
    entry = _entry(source, columns={"time": 1, "radius": 2, "velocity": 3})
    entry["initial_state"] = {"time": 0.0, "radius": 1.0, "velocity": 2.0}

    _, histories, t_max = load_manifest(_manifest(tmp_path, [entry]))

    history = histories[0]
    assert t_max == 1.5
    np.testing.assert_allclose(history.time, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(history.radius, [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(history.speed, [1.0, 1.0, 2.0, 2.5])


def test_rejects_nonmonotone_time(tmp_path: Path) -> None:
    source = tmp_path / "history.txt"
    np.savetxt(source, [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.5, 3.0, 1.0]])

    with pytest.raises(ValueError, match="strictly increasing"):
        load_manifest(_manifest(tmp_path, [_entry(source)], t_max=0.5))


def test_rejects_duplicate_column_mapping(tmp_path: Path) -> None:
    source = tmp_path / "history.txt"
    np.savetxt(source, [[0.0, 1.0, 0.0], [2.0, 2.0, 1.0]])
    entry = _entry(source, columns={"time": 0, "radius": 1, "velocity": 1})
    with pytest.raises(ValueError, match="distinct"):
        load_manifest(_manifest(tmp_path, [entry]))


def test_rejects_negative_initial_time(tmp_path: Path) -> None:
    source = tmp_path / "history.txt"
    np.savetxt(source, [[0.5, 1.0, 0.0], [2.0, 2.0, 1.0]])
    entry = _entry(source)
    entry["initial_state"] = {"time": -1.0, "radius": 1.0, "velocity": 0.0}
    with pytest.raises(ValueError, match="non-negative"):
        load_manifest(_manifest(tmp_path, [entry]))


@pytest.mark.parametrize(
    ("rows", "columns", "message"),
    [
        ([[0.0, 1.0], [2.0, 3.0]], {"time": 0, "radius": 1, "velocity": 2}, "at least 3 columns"),
        ([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]], None, "must cover the requested interval"),
    ],
)
def test_rejects_insufficient_columns_or_time_coverage(
    tmp_path: Path,
    rows: list[list[float]],
    columns: dict[str, int] | None,
    message: str,
) -> None:
    source = tmp_path / "history.txt"
    np.savetxt(source, rows)

    with pytest.raises(ValueError, match=message):
        load_manifest(_manifest(tmp_path, [_entry(source, columns=columns)], t_max=1.5))
