"""Savva & Bush (JFM 626, 2009) conventions and reference-data access.

These conversions exist only to talk to that paper, so they live with the
validation evidence rather than in shared source. Their non-dimensionalisation
is not ours, and mixing the two is the single easiest way to get a wrong answer
here:

    Oh_SB   = mu / sqrt(2 H rho gamma)      ours omits the sqrt(2)
    u_c     = sqrt(2 gamma / (rho H))       identical to our V_TC = sqrt(2)
    tau_vis = mu H / (2 gamma)              the *viscous* clock

``H`` is the FULL sheet thickness. In our units (rho = gamma = H = 1) this gives
``mu = sqrt(2) Oh_SB`` and ``tau_vis = mu / 2``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Taylor-Culick speed in our units; identical to their u_c.
V_TC = math.sqrt(2.0)


def viscosity(oh_sb: float) -> float:
    """Our coded dynamic viscosity for a Savva-Bush Ohnesorge number."""
    return oh_sb * math.sqrt(2.0)


def tau_vis(mu: float) -> float:
    """Their viscous timescale in our units.

    This is the *viscous* clock: their curves collapse onto it only for
    ``Oh_SB >~ 0.5``. At lower Oh the approach to u_c is inertial, set by rim
    growth in capillary time, so ``t* = t/tau_vis`` is not a meaningful target
    (``t* = 30`` at ``Oh_SB = 0.035`` is ``t = 0.75`` -- the very start).
    """
    return mu / 2.0


def influence_length(mu: float) -> float:
    """Extent of the region the tip disturbs, in sheet thicknesses.

    Deka & Pierson (PRF 5, 093603, 2020): "the region of influence of the tip
    motion is proportional to Oh". Their Eq. (6) carries the inertial term with
    prefactor ``(L0/h0)^2 / Oh^2``, giving pseudo-infinite (Taylor-Culick)
    behaviour for ``L/h0 >> Oh`` -- LINEAR in Oh. The square in that prefactor
    is the square of the ratio, not a different scaling.
    """
    return mu


def load_reference(figure: str) -> dict[float, np.ndarray]:
    """Digitised curves for ``fig5`` or ``fig6``, keyed by Oh_SB.

    Each value is an (N, 2) array of ``(tstar, ustar)``. Provenance and the
    digitisation uncertainty are recorded in each CSV header.
    """
    out: dict[float, np.ndarray] = {}
    for f in sorted(DATA_DIR.glob(f"savva_bush_2009_{figure}_Oh*.csv")):
        m = re.search(r"_Oh([0-9p.]+)", f.stem)
        if not m:
            continue
        oh = float(m.group(1).replace("p", "."))
        out[oh] = np.loadtxt(f, delimiter=",", comments="#")
    return out
