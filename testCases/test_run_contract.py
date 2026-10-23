"""The shared run contract: which runs may feed a figure, and Newtonian case dispatch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from taylor_culick_fem.cases import case_problems
from taylor_culick_fem import run_contract
from taylor_culick_fem.run_contract import figure_problem, output_problems

ROOT = Path(__file__).resolve().parents[1]
PLANAR = ROOT / "validationCases" / "savva-bush-planar" / "cases" / "planar-oh1.json"


class LauncherAffinityTests(unittest.TestCase):
    def test_native_threads_return_to_the_mask_captured_before_imports(self) -> None:
        masks = {101: set(range(64)), 102: set(range(64))}
        tasks = mock.Mock()
        tasks.is_dir.return_value = True
        tasks.iterdir.return_value = [SimpleNamespace(name="101"), SimpleNamespace(name="102")]
        with mock.patch.object(run_contract, "Path", return_value=tasks), \
                mock.patch.object(run_contract.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]), \
                mock.patch.object(run_contract.os, "sched_setaffinity",
                                  side_effect=lambda tid, cpus: masks.update({tid: set(cpus)})):
            receipt = run_contract.enforce_launcher_affinity({14, 46})
        self.assertEqual(masks, {101: {14, 46}, 102: {14, 46}})
        self.assertEqual(receipt["repaired_threads"], 2)

    def test_a_native_thread_that_rejects_binding_is_refused(self) -> None:
        tasks = mock.Mock()
        tasks.is_dir.return_value = True
        tasks.iterdir.return_value = [SimpleNamespace(name="101")]
        with mock.patch.object(run_contract, "Path", return_value=tasks), \
                mock.patch.object(run_contract.os, "sched_getaffinity", return_value=set(range(64))), \
                mock.patch.object(run_contract.os, "sched_setaffinity"):
            with self.assertRaisesRegex(RuntimeError, "rejected the launcher CPU mask"):
                run_contract.enforce_launcher_affinity({14, 46})


class FigureProblemTests(unittest.TestCase):
    def run_folder(self, tmp: str, **manifest) -> Path:
        folder = Path(tmp) / "run"
        folder.mkdir()
        if manifest:
            (folder / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return folder

    def test_a_run_without_a_manifest_cannot_feed_a_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no run-manifest.json", figure_problem(self.run_folder(tmp)))

    def test_a_dirty_or_unknown_commit_cannot_feed_a_figure(self) -> None:
        for commit in ("a" * 40 + "-dirty", "unknown", "abc123"):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(commit=commit):
                folder = self.run_folder(tmp, component_commit=commit, pyoomph={"commit": "b" * 40})
                self.assertIn("not a clean full commit", figure_problem(folder))

    def test_a_clean_run_feeds_a_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.run_folder(tmp, component_commit="a" * 40, pyoomph={"commit": "b" * 40})
            self.assertIsNone(figure_problem(folder))

    def test_an_occupied_output_folder_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(output_problems(Path(tmp) / "new"), [])
            (Path(tmp) / "file").write_text("x", encoding="utf-8")
            self.assertTrue(output_problems(Path(tmp)))


class NewtonianCaseTests(unittest.TestCase):
    def test_each_runner_accepts_only_its_own_problem(self) -> None:
        case = json.loads(PLANAR.read_text(encoding="utf-8"))
        self.assertEqual(case_problems(case), [])
        self.assertEqual(case_problems(case, "planar_comoving"), [])
        self.assertTrue(case_problems(case, "axisymmetric_comoving"))
        self.assertTrue(case_problems(case, "axisymmetric_viscoelastic"))

    def test_newtonian_case_needs_a_positive_oh(self) -> None:
        case = json.loads(PLANAR.read_text(encoding="utf-8"))
        case["physics"]["Oh"] = 0
        self.assertTrue(any("physics.Oh" in p for p in case_problems(case)))


if __name__ == "__main__":
    unittest.main()
