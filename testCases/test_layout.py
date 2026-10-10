"""The layout contract holds independently of where the code is run from.

Library code lives under ``src/``; no entry point shadows a package; the case validator
and the provenance helper need no solver; and the runner finds its package from any
working directory without an installed package or ``PYTHONPATH``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "taylor_culick_fem"
RUNNER = ROOT / "simulationCases" / "run_axisymmetric_viscoelastic.py"
CASE = ROOT / "simulationCases" / "axisymmetric_viscoelastic.json"
SCRIPT_FOLDERS = ("simulationCases", "validationCases", "verificationCases",
                  "modelComparisonCases", "postProcess")


def imported_names(path: Path) -> set[str]:
    """Top-level names of every absolute import in ``path``."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


class LayoutTests(unittest.TestCase):
    def test_no_docs_and_no_runner_at_the_root(self) -> None:
        self.assertFalse((ROOT / "docs").exists())
        self.assertEqual(sorted(p.name for p in ROOT.glob("run_*")), [])

    def test_no_script_shadows_a_package(self) -> None:
        packages = {p.name for p in (ROOT / "src").iterdir() if (p / "__init__.py").is_file()}
        for folder in SCRIPT_FOLDERS:
            for script in sorted((ROOT / folder).rglob("*.py")):
                with self.subTest(script=str(script.relative_to(ROOT))):
                    self.assertNotIn(script.stem, packages)

    def test_validator_and_provenance_need_no_solver(self) -> None:
        for name in ("cases.py", "provenance.py"):
            with self.subTest(module=name):
                self.assertNotIn("pyoomph", imported_names(PACKAGE / name))

    def test_library_imports_no_entry_point(self) -> None:
        forbidden = set(SCRIPT_FOLDERS) | {"testCases"}
        for path in sorted((ROOT / "src").rglob("*.py")):
            with self.subTest(module=str(path.relative_to(ROOT))):
                self.assertEqual(imported_names(path) & forbidden, set())

    def test_runner_finds_its_package_from_any_working_directory(self) -> None:
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for cwd in (Path(tmp), RUNNER.parent):
                with self.subTest(cwd=str(cwd)):
                    done = subprocess.run([sys.executable, str(RUNNER), str(CASE), "--out", str(out), "--check"],
                                          cwd=cwd, env=environment, capture_output=True, text=True,
                                          check=False)
                    self.assertEqual(done.returncode, 0, done.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
