# ruff: noqa: N999 -- the layout contract names the test folder testCases/.
"""Software tests. Puts ``src/`` on the import path so tests run without an installed package.

Run from the repository root: ``python -m pytest testCases``. The unittest-style modules
also run under ``python -m unittest discover -s testCases -t .``.
"""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# hwloc's GL component probes graphics displays when MPI starts, and an unresponsive
# display blocks that probe indefinitely; production launches disable it the same way.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
