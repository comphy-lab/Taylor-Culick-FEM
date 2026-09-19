from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "postProcess"))
import viscoelastic_campaign_history as campaign


def _history(time: np.ndarray, radius: np.ndarray, ustar: np.ndarray):
    return campaign.TipHistory(time, radius, ustar, "viscoelastic", Path("synthetic"))


def test_loader_reconstructs_exact_t0_for_each_schema(tmp_path: Path) -> None:
    viscoelastic = np.array([[0.1, 1.01, 0.1, 0.07, 1, 1, 0, 0]])
    newtonian = np.array([[0.1, 0.2, 1.01, 0.1, 0.07, 1, 1]])
    ve_path, n_path = tmp_path / "ve.txt", tmp_path / "newtonian.txt"
    np.savetxt(ve_path, viscoelastic)
    np.savetxt(n_path, newtonian)

    ve = campaign.load_viscoelastic_control(ve_path)
    newton = campaign.load_newtonian_control(n_path)

    for loaded in (ve, newton):
        assert loaded.time[0] == 0
        assert loaded.radius[0] == 1
        assert loaded.ustar[0] == 0
    assert ve.radius[-1] == newton.radius[-1] == 1.01
    assert ve.ustar[-1] == newton.ustar[-1] == 0.07


def test_newtonian_globals_uses_solved_control_speed(tmp_path: Path) -> None:
    path = tmp_path / "globals.txt"
    np.savetxt(path, np.array([[0.0, 1.0, 0.0], [2.0, 3.0, np.sqrt(2.0)]]))

    loaded = campaign.load_control(path, schema="newtonian_globals")

    assert loaded.schema == "newtonian_globals"
    assert loaded.radius.tolist() == [1.0, 3.0]
    assert loaded.ustar.tolist() == pytest.approx([0.0, 1.0])


def test_loader_rejects_wrong_schema_and_nonmonotone_time(tmp_path: Path) -> None:
    path = tmp_path / "control.txt"
    np.savetxt(path, np.ones((2, 7)))
    with pytest.raises(ValueError, match="exactly 8 columns"):
        campaign.load_viscoelastic_control(path)
    with pytest.raises(ValueError, match="strictly increasing"):
        campaign.load_newtonian_control(path)
    with pytest.raises(ValueError, match="schema must"):
        campaign.load_control(path, schema="automatic")


def test_monotone_asymptotic_history_is_strictly_settled() -> None:
    time = np.linspace(0, 100, 1001)
    ustar = 0.2 * (time / 5) * np.exp(-time / 5)
    radius = 1 + np.cumsum(ustar) * (time[1] - time[0])
    result = campaign.analyse_settling(_history(time, radius, ustar))

    assert result["status"] == "strict settled"
    assert result["complete_trailing_window"] is True
    assert result["window"]["maximum_absolute_ustar"] < 1e-3
    assert result["window"]["radius_span"] < 1e-3


def test_recoil_is_not_misclassified_at_zero_crossing() -> None:
    time = np.linspace(0, 100, 1001)
    ustar = np.where(time < 20, 0.08 * np.sin(np.pi * time / 20), -0.01)
    radius = 1 + np.cumsum(ustar) * (time[1] - time[0])
    result = campaign.analyse_settling(_history(time, radius, ustar))

    assert result["established_opening"] is True
    assert result["status"] == "not arrested"
    assert result["window"]["maximum_absolute_ustar"] == pytest.approx(0.01)


def test_quiet_tip_plateau_is_distinct_from_strict_settlement() -> None:
    time = np.linspace(0, 100, 1001)
    ustar = np.where(time < 30, 0.08 * np.sin(np.pi * time / 30), 5e-4)
    radius = 1 + np.cumsum(ustar) * (time[1] - time[0])
    result = campaign.analyse_settling(_history(time, radius, ustar))

    assert result["status"] == "quiet-tip plateau"
    assert result["window"]["maximum_absolute_ustar"] < 1e-3
    assert result["window"]["radius_span"] > 1e-3


def test_incomplete_horizon_is_not_settled() -> None:
    time = np.linspace(0, 40, 401)
    ustar = 0.1 * np.exp(-((time - 10) / 4) ** 2)
    radius = 1 + np.cumsum(ustar) * (time[1] - time[0])
    result = campaign.analyse_settling(_history(time, radius, ustar))

    assert result["status"] == "not arrested"
    assert result["complete_trailing_window"] is False
    assert result["window"]["maximum_absolute_ustar"] is None
