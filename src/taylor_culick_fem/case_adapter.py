"""Build the viscoelastic driver's arguments from a case file.

``cases.driver_arguments`` maps the case onto the driver's options without the solver;
this module parses them with the driver's own parser and cross-checks, so it imports
pyoomph through the driver. The runner calls it before it creates any output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taylor_culick_fem.cases import driver_arguments
from taylor_culick_fem.viscoelastic_driver import build_parser, validate_args


def case_to_driver_args(case_path: Path, out: Path, *, restart_evidence: dict | None = None,
                        t_end: float | None = None, far_field_extension: float | None = None,
                        newton_tolerance_safety: float | None = None,
                        run_dir: Path | None = None) -> argparse.Namespace:
    """Driver arguments for ``case_path`` writing to ``out``.

    The case bytes are read once and carried as ``serialized_case``, so the input-case
    snapshot is exactly what was validated. ``restart_evidence`` (from
    ``lineage.restart_evidence``) makes the run a continuation of its parent; ``run_dir``
    tells the driver to write into an existing folder instead of creating one under ``out``.
    """
    case_path = Path(case_path).resolve()
    serialized_case = case_path.read_bytes()
    case = json.loads(serialized_case)
    arguments = driver_arguments(case, out, t_end=t_end, far_field_extension=far_field_extension,
                                 newton_tolerance_safety=newton_tolerance_safety)
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    validate_args(parser, parsed)
    parsed.serialized_case = serialized_case
    # The runner's command line is not pyoomph's: keep pyoomph from reading it.
    parsed.ignore_command_line = True
    if restart_evidence is not None:
        parsed.restart_case = case_path
        parsed.restart_evidence = restart_evidence
    if run_dir is not None:
        parsed.run_dir = Path(run_dir)
    return parsed
