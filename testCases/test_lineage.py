"""Restart evidence follows the parent's recorded numerics schedule across a cadence change."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taylor_culick_fem.lineage import control_time, restart_evidence

# Four rows at cadence 0.05, then a continuation at cadence 1 from row 4 (t = 0.2).
SCHEDULE = [
    {"time": 0.0, "control_index": 0, "max_step": 0.01, "interval_start": "legacy"},
    {"time": 0.2, "control_index": 4, "max_step": 1.0, "interval_start": "carry", "output_step": 1.0},
]
TIMES = [0.05, 0.1, 0.15, 0.2, 1.2, 2.2]


def write_control(path: Path, times: list[float]) -> Path:
    path.write_text("# t R\n" + "".join(f"{t:.10f} 1.0\n" for t in times), encoding="utf-8")
    return path


class ControlTimeTests(unittest.TestCase):
    def test_rows_follow_the_schedule_and_its_landing_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = write_control(Path(tmp) / "control.txt", TIMES)
            self.assertAlmostEqual(control_time(control, 6, 1.0, SCHEDULE), 2.2)
            self.assertAlmostEqual(control_time(control, 3, 1.0, SCHEDULE), 0.15)
            # Without the schedule the cadence would be taken as uniform from t = 0.
            with self.assertRaisesRegex(ValueError, "not its scheduled"):
                control_time(control, 6, 1.0)
            # pyoomph lands up to 1e-5 of a step late: allowed at cadence 1, not beyond it.
            write_control(control, TIMES[:5] + [2.2 + 1.5e-5])
            self.assertAlmostEqual(control_time(control, 6, 1.0, SCHEDULE), 2.2 + 1.5e-5)
            write_control(control, TIMES[:5] + [2.2 + 5e-5])
            with self.assertRaisesRegex(ValueError, "not its scheduled"):
                control_time(control, 6, 1.0, SCHEDULE)

    def test_a_uniform_parent_keeps_the_old_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = write_control(Path(tmp) / "control.txt", [0.05 * k for k in range(1, 5)])
            self.assertAlmostEqual(control_time(control, 4, 0.05), 0.2)
            with self.assertRaisesRegex(ValueError, "fewer than"):
                control_time(control, 5, 0.05)


class RestartEvidenceTests(unittest.TestCase):
    def test_evidence_reads_the_parent_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            run = parent / "run_viscoelastic_axi"
            (parent / "checkpoints").mkdir(parents=True)
            run.mkdir()
            (parent / "input-case.json").write_text(json.dumps({"output": {"cadence": 1.0}}), encoding="utf-8")
            (parent / "case.json").write_text(json.dumps({"output_step": 1.0, "numerics_schedule": SCHEDULE}),
                                              encoding="utf-8")
            write_control(run / "control.txt", TIMES)
            (run / "rim-height.txt").write_text("# t H\n", encoding="utf-8")
            (run / "liquid.pvd").write_text("<VTKFile/>\n", encoding="utf-8")
            state = parent / "checkpoints" / "ckpt_00006.dump"
            state.write_bytes(b"state")
            evidence = restart_evidence(state, parent)
            self.assertAlmostEqual(evidence["time"], 2.2)
            self.assertEqual((evidence["control_index"], evidence["field_index"]), (6, 6))


if __name__ == "__main__":
    unittest.main()
