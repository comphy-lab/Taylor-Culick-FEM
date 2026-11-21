"""Contracts for bounded recording, independent of a pyoomph installation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from taylor_culick_fem.newton_observer import NewtonObserver


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def begin(observer, labels=("globals/X", "globals/U", "mesh/x", "unknown")):
    observer.start_solve(3.0, 0.5, 1e-12, 1.0, 1e-30, 1.0, labels)


def test_pairing_rollback_and_final_native_history(tmp_path):
    path = tmp_path / "newton.jsonl"
    observer = NewtonObserver(path)
    before = np.array([1.0, 2.0, 3.0, 4.0])
    after = np.array([1.5, 2.0, 2.5, 4.5])
    begin(observer)
    observer.before_step(before, [])
    observer.after_step(after)
    observer.after_step(None)  # native rollback is not a Newton correction
    observer.end_solve([4e-3, 3e-6], outcome="newton_rejected")
    observer.close()
    record = records(path)[0]
    assert record["iteration_count"] == 1
    assert record["rollback_callbacks"] == 1
    assert record["native_residuals"] == [4e-3, 3e-6]
    assert record["temporal_acceptance"] == "unknown"
    assert record["iterations"][0]["applied_abs_max_by_field"] == {
        "globals/X": 0.5, "globals/U": 0.0, "mesh/x": 0.5, "unknown": 0.5,
    }
    np.testing.assert_array_equal(before, [1, 2, 3, 4])
    np.testing.assert_array_equal(after, [1.5, 2, 2.5, 4.5])


def test_rollback_discards_pending_before_snapshot(tmp_path):
    path = tmp_path / "rollback.jsonl"
    observer = NewtonObserver(path)
    begin(observer, ("X",))
    observer.before_step(np.array([1.0]), [])
    # A linear solve may fail before its matching after-step callback.
    observer.rollback()
    observer.end_solve([1.0], outcome="temporal_rejected")
    observer.close()
    record = records(path)[0]
    assert record["iteration_count"] == 0
    assert record["rollback_callbacks"] == 1
    assert not record["unpaired_before"]
    assert record["temporal_acceptance"] == "rejected"


def test_shadow_strict_threshold_and_native_residual_timing(tmp_path):
    path = tmp_path / "shadow.jsonl"
    observer = NewtonObserver(path)
    begin(observer, ("globals/X", "globals/U"))
    eps = np.finfo(float).eps
    # Native max history is [initial, after update 1, after update 2].
    initial = np.array([10.0, 20.0])
    first = np.array([11.0, 19.0])
    final = np.array([11.5, 18.75])
    observer.before_step(initial, [])
    observer.after_step(first)
    observer.before_step(first, [1.0, 60 * eps])
    observer.after_step(final)
    observer.end_solve([1.0, 60 * eps, 59 * eps], outcome="converged")
    observer.close()
    record = records(path)[0]
    assert record["shadow"]["60"]["iteration"] == 2  # equality is not accepted
    assert record["shadow"]["1000"]["iteration"] == 1
    assert "30" not in record["shadow"]
    assert record["shadow"]["1000"]["signed_final_minus_shadow"] == {"X": 0.5, "U": -0.25}
    assert record["native_residuals"][-1] == pytest.approx(59 * eps)


def test_zero_update_and_temporal_outcome_remain_distinct(tmp_path):
    path = tmp_path / "zero.jsonl"
    observer = NewtonObserver(path)
    begin(observer, ("globals/X",))
    observer.before_step(np.array([1.0]), [])
    # Initial convergence can return without an after-update callback.
    observer.end_solve([1e-15], outcome="converged")
    observer.close()
    record = records(path)[0]
    assert record["iteration_count"] == 0
    assert record["unpaired_before"]
    assert record["temporal_acceptance"] == "unknown"
    assert record["shadow"] == {}


def test_summary_mode_does_not_require_dofs_or_labels(tmp_path):
    path = tmp_path / "bounded.jsonl"
    observer = NewtonObserver(path, max_full_solves=1)
    assert observer.wants_details
    begin(observer, ("X",))
    assert observer.wants_details
    observer.before_step(np.array([1.0]), [])
    observer.after_step(np.array([2.0]))
    observer.end_solve([1.0, 0.1], outcome="temporal_accepted")
    assert not observer.wants_details
    begin(observer, ())
    observer.before_step(None, [])
    observer.after_step(None)
    observer.end_solve([1.0, 0.1], outcome="temporal_rejected")
    observer.close()
    full, summary = records(path)
    assert full["sampling"] == "full" and full["temporal_acceptance"] == "accepted"
    assert summary["sampling"] == "summary" and summary["temporal_acceptance"] == "rejected"
    assert "iterations" not in summary


def test_unfinished_solve_is_not_silently_merged(tmp_path):
    path = tmp_path / "unfinished.jsonl"
    observer = NewtonObserver(path)
    begin(observer, ("X",))
    observer.before_step(np.array([1.0]), [])
    begin(observer, ("X",))
    observer.end_solve([], outcome="unknown")
    observer.close()
    first, second = records(path)
    assert first["attempt_outcome"] == "incomplete"
    assert first["unpaired_before"]
    assert second["solve_id"] == 2


def test_output_and_snapshot_limits_disable_without_raising(tmp_path):
    path = tmp_path / "cap.jsonl"
    observer = NewtonObserver(path, max_bytes=16)
    begin(observer, ("X",))
    observer.end_solve([], outcome="unknown")
    assert not observer.enabled
    assert "byte limit" in observer.failure
    assert path.stat().st_size <= 16

    other = NewtonObserver(tmp_path / "memory.jsonl", max_snapshot_bytes=12)
    begin(other, ("X",))
    other.before_step(np.array([1.0]), [])
    assert not other.enabled
    assert "snapshot byte limit" in other.failure


def test_existing_file_is_never_overwritten_and_errors_disable_once(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("keep")
    observer = NewtonObserver(path)
    assert not observer.enabled and path.read_text() == "keep"
    failure = observer.failure
    observer.disable(RuntimeError("later"))
    assert observer.failure == failure


def test_new_labels_are_used_after_each_solve(tmp_path):
    path = tmp_path / "renumbered.jsonl"
    observer = NewtonObserver(path)
    begin(observer, ("X", "U"))
    observer.before_step(np.array([1.0, 2.0]), [])
    observer.after_step(np.array([3.0, 2.0]))
    observer.end_solve([1.0, 0.1])
    begin(observer, ("U", "X"))
    observer.before_step(np.array([1.0, 2.0]), [])
    observer.after_step(np.array([3.0, 2.0]))
    observer.end_solve([1.0, 0.1])
    observer.close()
    first, second = records(path)
    assert first["iterations"][0]["applied_abs_max_by_field"]["X"] == 2.0
    assert second["iterations"][0]["applied_abs_max_by_field"]["U"] == 2.0


def test_sampled_residual_is_phased_bounded_and_separate_from_native(tmp_path):
    path = tmp_path / "sample.jsonl"
    observer = NewtonObserver(path, max_residual_samples=1)
    assert not observer.wants_residual_sample
    begin(observer, ("X",))
    assert observer.wants_residual_sample
    observer.sample_residual({"phase": "after_convergence_hook", "row": 3, "max_abs": 2.0})
    assert not observer.wants_residual_sample
    observer.sample_residual({"phase": "ignored", "row": 4})
    observer.end_solve([1.0, 0.1])
    begin(observer, ("X",))
    assert not observer.wants_residual_sample
    observer.end_solve([2.0, 0.2])
    observer.close()
    first, second = records(path)
    assert first["sampled_residual"] == {
        "phase": "after_convergence_hook", "row": 3, "max_abs": 2.0,
    }
    assert first["native_residuals"] == [1.0, 0.1]
    assert second["sampled_residual"] is None


def test_invalid_sample_disables_observer_without_raising(tmp_path):
    observer = NewtonObserver(tmp_path / "bad-sample.jsonl")
    begin(observer, ("X",))
    observer.sample_residual({"phase": "post_hook", "value": float("nan")})
    assert not observer.enabled
    assert observer.failure is not None
