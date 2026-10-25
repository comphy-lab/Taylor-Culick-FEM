"""Runner contracts that need no solver run: refusal paths, the manifest and run scripts.

The runner refuses an invalid case, and an output folder that already holds anything,
for a fresh run and a restart alike, before it creates anything. It writes its manifest
before any other output and never replaces it. Every ``run_*.sh`` must parse against the
options of the runner it calls.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulationCases" / "run_axisymmetric_viscoelastic.py"
CASE = ROOT / "simulationCases" / "axisymmetric_viscoelastic.json"
SCRIPT_FOLDERS = ("simulationCases", "validationCases", "verificationCases", "modelComparisonCases")


def load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"runner_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_calls(script: Path) -> list[tuple[str, list[str]]]:
    """Each ``run_*.py`` invocation in a shell script: the runner's file name and its arguments."""
    text = re.sub(r"\\\n", " ", script.read_text(encoding="utf-8"))
    calls = []
    for line in text.splitlines():
        tokens = shlex.split(line, comments=True)
        for index, token in enumerate(tokens):
            if re.fullmatch(r"run_[A-Za-z0-9_-]+\.py", Path(token).name):
                arguments = [item for item in tokens[index + 1:] if item not in {"$@", "${@}"}]
                calls.append((Path(token).name, arguments))
                break
    return calls


def resolve_runner(script: Path, name: str) -> Path:
    """The runner a script calls: beside the script, then in simulationCases/, then unique elsewhere."""
    for candidate in (script.parent / name, ROOT / "simulationCases" / name):
        if candidate.is_file():
            return candidate
    matches = [path for folder in SCRIPT_FOLDERS for path in (ROOT / folder).rglob(name)]
    if len(matches) != 1:
        raise AssertionError(f"{script.name}: cannot identify runner {name}: {matches}")
    return matches[0]


class RefusalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load(RUNNER)

    def refused(self, *argv: str) -> str:
        with self.assertRaises(SystemExit) as stop:
            self.runner.main(list(argv))
        return str(stop.exception)

    def parent_run(self, folder: Path, case: Path = CASE) -> Path:
        """A finished run folder of ``case`` with one state file."""
        folder.mkdir(parents=True)
        (folder / "run-manifest.json").write_text(json.dumps({"case_sha256": sha256(case)}), encoding="utf-8")
        state = folder / "state_0001.dump"
        state.write_bytes(b"state")
        return state

    def test_exit_status_is_zero_only_at_the_declared_stop(self) -> None:
        for status in ("passed", "horizon_reached"):
            self.assertEqual(self.runner.exit_status({"status": status}), 0, status)
        for status in ("failed", "error", "initialising", "running", None):
            self.assertEqual(self.runner.exit_status({"status": status}), 1, status)

    def test_invalid_case_is_refused_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad, out = Path(tmp) / "bad.json", Path(tmp) / "out"
            bad.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
            self.assertIn("invalid case", self.refused(str(bad), "--out", str(out)))
            self.assertFalse(out.exists())

    def test_invalid_option_is_refused_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            self.assertIn("--t-end", self.refused(str(CASE), "--out", str(out), "--t-end", "-1"))
            self.assertFalse(out.exists())

    def test_fresh_run_into_an_existing_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run-manifest.json").write_text("{}", encoding="utf-8")
            self.assertIn("own new output folder", self.refused(str(CASE), "--out", str(out)))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["run-manifest.json"])

    def test_restart_into_an_existing_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self.parent_run(Path(tmp) / "parent")
            out = Path(tmp) / "continuation"
            out.mkdir()
            (out / "progress.jsonl").write_text("", encoding="utf-8")
            message = self.refused(str(CASE), "--out", str(out), "--restart-from", str(state))
            self.assertIn("own new output folder", message)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["progress.jsonl"])

    def test_restart_state_must_exist_and_belong_to_this_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "continuation"
            missing = Path(tmp) / "missing.dump"
            self.assertIn("does not exist", self.refused(str(CASE), "--out", str(out), "--restart-from", str(missing)))
            orphan = Path(tmp) / "orphan" / "state.dump"
            orphan.parent.mkdir()
            orphan.write_bytes(b"state")
            self.assertIn("cannot be checked", self.refused(str(CASE), "--out", str(out), "--restart-from", str(orphan)))
            other = Path(tmp) / "other.json"
            other.write_text(json.dumps({"case_id": "other"}), encoding="utf-8")
            foreign = self.parent_run(Path(tmp) / "foreign", case=other)
            self.assertIn("another case", self.refused(str(CASE), "--out", str(out), "--restart-from", str(foreign)))
            state = self.parent_run(Path(tmp) / "parent")
            nested = state.parent / "continuation"
            self.assertIn("not inside its parent", self.refused(str(CASE), "--out", str(nested), "--restart-from", str(state)))
            self.assertFalse(out.exists() or nested.exists())

    def test_a_continuation_may_change_only_its_numerics(self) -> None:
        base = json.loads(CASE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            state = self.parent_run(Path(tmp) / "parent")
            (state.parent / "input-case.json").write_text(CASE.read_text(encoding="utf-8"), encoding="utf-8")
            out = Path(tmp) / "continuation"
            faster = copy.deepcopy(base)
            faster["output"]["cadence"] = 2 * base["output"]["cadence"]
            faster["solver"].update(max_step=2 * base["output"]["cadence"], initial_guess="extrapolate")
            path = Path(tmp) / "faster.json"
            path.write_text(json.dumps(faster), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(self.runner.main([str(path), "--out", str(out), "--restart-from",
                                                   str(state), "--check"]), 0)
            self.assertEqual(json.loads(printed.getvalue())["check"], "passed")
            from taylor_culick_fem.cases import continuation_case_problems
            from taylor_culick_fem.run_contract import check_restart
            _, record = check_restart(state, out, sha256(path), case=faster,
                                      continuation_check=continuation_case_problems)
            self.assertEqual(record["case_change"], {
                "parent_case_sha256": sha256(CASE),
                "changed": ["output.cadence", "solver.initial_guess", "solver.max_step"]})
            # Without the check, as for the other runners, a changed case is another case.
            with self.assertRaises(SystemExit) as stop:
                check_restart(state, out, sha256(path))
            self.assertIn("another case", str(stop.exception))
            heavier = copy.deepcopy(faster)
            heavier["physics"]["Oh"] = 2 * base["physics"]["Oh"]
            path.write_text(json.dumps(heavier), encoding="utf-8")
            message = self.refused(str(path), "--out", str(out), "--restart-from", str(state), "--check")
            self.assertIn("another case", message)
            self.assertIn("physics.Oh", message)
            (state.parent / "input-case.json").unlink()
            path.write_text(json.dumps(faster), encoding="utf-8")
            message = self.refused(str(path), "--out", str(out), "--restart-from", str(state), "--check")
            self.assertIn("no input-case.json", message)
            self.assertFalse(out.exists())

    def test_check_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(self.runner.main([str(CASE), "--out", str(out), "--check"]), 0)
            self.assertEqual(json.loads(printed.getvalue())["check"], "passed")
            self.assertFalse(out.exists())

    def test_manifest_is_written_first_and_never_replaced(self) -> None:
        # This runner imports the real driver after its checks, so the test needs the pinned
        # environment; the solve itself is replaced by a stand-in that stops at once.
        if importlib.util.find_spec("pyoomph") is None:
            self.skipTest("needs the pinned pyoomph environment")
        from taylor_culick_fem import viscoelastic_driver
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(viscoelastic_driver, "run_case", side_effect=RuntimeError("solver stopped")):
            out = Path(tmp) / "out"
            with self.assertRaises(RuntimeError):
                self.runner.main([str(CASE), "--out", str(out)])
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["progress.jsonl", "run-manifest.json"])
            manifest = json.loads((out / "run-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_sha256"], sha256(CASE))
            self.assertEqual(manifest["case_id"], json.loads(CASE.read_text(encoding="utf-8"))["case_id"])
            self.assertEqual(manifest["case_fields_used"], self.runner.CASE_FIELDS_USED)
            self.assertIn(str(CASE), manifest["argv"])
            self.assertEqual(set(manifest["pyoomph"]), {"version", "url", "commit"})
            self.assertTrue(manifest["component_commit"])
            events = [json.loads(line)["event"] for line in (out / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(events, ["started", "failed"])
            before = (out / "run-manifest.json").read_bytes()
            self.assertIn("own new output folder", self.refused(str(CASE), "--out", str(out)))
            self.assertEqual((out / "run-manifest.json").read_bytes(), before)

    def test_restart_needs_a_checkpoint_of_its_parent(self) -> None:
        # A state that is attributable but is not <parent>/checkpoints/ckpt_NNNNN.dump.
        if importlib.util.find_spec("pyoomph") is None:
            self.skipTest("needs the pinned pyoomph environment")
        with tempfile.TemporaryDirectory() as tmp:
            state = self.parent_run(Path(tmp) / "parent")
            out = Path(tmp) / "continuation"
            message = self.refused(str(CASE), "--out", str(out), "--restart-from", str(state))
            self.assertIn("ckpt_NNNNN.dump", message)
            self.assertFalse(out.exists())


class RunScriptTests(unittest.TestCase):
    def test_every_run_script_parses_against_its_runner(self) -> None:
        scripts = sorted(path for folder in SCRIPT_FOLDERS for path in (ROOT / folder).rglob("run_*.sh"))
        if not scripts:
            self.skipTest("no run_*.sh scripts yet")
        for script in scripts:
            with self.subTest(script=str(script.relative_to(ROOT))):
                if shutil.which("bash"):
                    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
                    self.assertEqual(syntax.returncode, 0, syntax.stderr)
                calls = runner_calls(script)
                self.assertTrue(calls, "the script never calls a run_*.py runner")
                for name, arguments in calls:
                    runner = load(resolve_runner(script, name))
                    try:
                        args = runner.build_parser().parse_args(arguments)
                    except SystemExit:
                        self.fail(f"{script.name}: {name} rejects {arguments}")
                    case = str(args.case).replace("${here}", str(script.parent)).replace("$here", str(script.parent))
                    case = case.replace("${root}", str(ROOT)).replace("$root", str(ROOT))
                    if "$" not in case:
                        self.assertTrue(Path(case).is_file(), f"{script.name}: case {case} is missing")


if __name__ == "__main__":
    unittest.main()
