# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Extend the sheet block of a retained reference grid beyond its outer end.

Keeps every sheet column up to the last full-spacing element end, drops the short closing element, and appends
quadratic elements of vertical columns with the last column's Z distribution, the spacing growing by GROWTH from
300 until X >= X_MAX. The cap blocks are copied unchanged (the removed-region quadrature of the window prototypes
uses them).
usage: extend_grid.py <out.npz> [X_MAX=6e5] [GROWTH=1.05]
env (required): SEED (folder of the retained seed segment; its grids.npz is the grid extended)
"""
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(os.environ["SEED"]) / "grids.npz"
out = Path(sys.argv[1])
x_max = float(sys.argv[2]) if len(sys.argv) > 2 else 6e5
growth = float(sys.argv[3]) if len(sys.argv) > 3 else 1.05
g = dict(np.load(SRC))
sheet = g["sheet"]
assert sheet.shape[0] % 2 == 1
keep = sheet[:-2]  # drop the short closing element (mid and end columns)
zcol = keep[-1, :, 1]
x, h, cols = float(keep[-1, 0, 0]), 300.0, []
while x < x_max:
    h *= growth
    for frac in (0.5, 1.0):
        cols.append(np.column_stack([np.full_like(zcol, x + frac * h), zcol]))
    x += h
g["sheet"] = np.concatenate([keep, np.array(cols)], axis=0)
assert g["sheet"].shape[0] % 2 == 1
np.savez(out, **g)
xs = g["sheet"][:, 0, 0]
print("sheet", g["sheet"].shape, "kept to %.3f, extended to %.1f with %d elements; last spacing %.1f" % (keep[-1, 0, 0], xs[-1], len(cols) // 2, xs[-1] - xs[-3]))
