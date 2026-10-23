"""Provenance fallbacks: the component commit, the installed pyoomph revision and the manifest."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from taylor_culick_fem.provenance import component_commit, create_run_manifest, pyoomph_revision

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
       "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"]


def git(root: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


class ComponentCommitTests(unittest.TestCase):
    def test_commit_file_of_an_archived_source_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "COMMIT").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
            self.assertEqual(component_commit(Path(tmp)), "0123456789abcdef0123456789abcdef01234567")

    def test_source_without_commit_or_git_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(component_commit(Path(tmp)), "unknown")

    @unittest.skipIf(shutil.which("git") is None, "needs git")
    def test_git_head_is_marked_dirty_only_for_tracked_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git(root, "init", "-q")
            (root / "source.py").write_text("x = 1\n", encoding="utf-8")
            git(root, "add", "source.py")
            git(root, "commit", "-q", "-m", "Add source")
            head = git(root, "rev-parse", "HEAD")
            self.assertEqual(component_commit(root), head)
            (root / "untracked.txt").write_text("ignored by the dirty check\n", encoding="utf-8")
            self.assertEqual(component_commit(root), head)
            (root / "source.py").write_text("x = 2\n", encoding="utf-8")
            self.assertEqual(component_commit(root), head + "-dirty")

    @unittest.skipIf(shutil.which("git") is None, "needs git")
    def test_a_folder_inside_another_work_tree_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            git(Path(tmp), "init", "-q")
            (Path(tmp) / "component").mkdir()
            self.assertEqual(component_commit(Path(tmp) / "component"), "unknown")


class PyoomphRevisionTests(unittest.TestCase):
    def test_pyoomph_without_installation_record_is_unknown(self) -> None:
        missing = importlib.metadata.PackageNotFoundError("pyoomph")
        with mock.patch("importlib.metadata.distribution", side_effect=missing):
            self.assertEqual(pyoomph_revision(), {"version": None, "url": None, "commit": None})

    def test_direct_url_record_supplies_repository_and_commit(self) -> None:
        record = json.dumps({"url": "https://example.invalid/pyoomph", "vcs_info": {"vcs": "git", "commit_id": "c" * 40}})
        distribution = mock.Mock(version="0.0.0", read_text=mock.Mock(return_value=record))
        with mock.patch("importlib.metadata.distribution", return_value=distribution):
            self.assertEqual(pyoomph_revision(), {"version": "0.0.0", "url": "https://example.invalid/pyoomph",
                                                  "commit": "c" * 40})

    def test_installed_pyoomph_is_the_pinned_commit(self) -> None:
        if importlib.util.find_spec("pyoomph") is None:
            self.skipTest("needs the pinned pyoomph environment")
        info = pyoomph_revision()
        self.assertTrue(info["version"])
        pins = "".join(path.read_text(encoding="utf-8") for path in (ROOT / "pyproject.toml", ROOT / "uv.lock")
                       if path.is_file())
        if info["commit"] is not None:
            self.assertIn(info["commit"], pins)


class RunManifestTests(unittest.TestCase):
    def test_run_manifest_is_created_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = create_run_manifest(Path(tmp), {"case_id": "first"})
            with self.assertRaises(FileExistsError):
                create_run_manifest(Path(tmp), {"case_id": "second"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"case_id": "first"})


if __name__ == "__main__":
    unittest.main()
