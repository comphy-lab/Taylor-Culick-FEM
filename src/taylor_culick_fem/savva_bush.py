"""Savva & Bush (JFM 626, 2009) conventions and reference-data access.

PROJECT CONVENTION. ``Oh`` throughout this repository is the Savva & Bush
Ohnesorge number, and nothing else:

    Oh      = mu / sqrt(2 H rho gamma)
    u_c     = sqrt(2 gamma / (rho H))       identical to our V_TC = sqrt(2)
    tau_vis = mu H / (2 gamma)              the *viscous* clock

``H`` is the FULL sheet thickness. In our units (rho = gamma = H = 1) this gives
``mu = sqrt(2) Oh`` and ``tau_vis = mu / 2``.

The quantity ``mu / sqrt(H rho gamma)`` -- numerically the coded viscosity, and
what Basilisk's ``mu1`` carries -- was previously also called "Oh" here. It no
longer has that name anywhere in this repository. Call it ``mu``, the coded
dynamic viscosity. The two differ by exactly sqrt(2), which is the single
easiest way to get a wrong answer in this project; that is why only one of them
keeps the name.

Historical note for anyone reading older output: ``case.json`` files written
before 2026-09-13 carry this same quantity under the key ``Oh_savva_bush``,
and carry ``mu`` under ``Oh_ours``. Both keys are still written for
compatibility, alongside the canonical ``Oh``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

#: Taylor-Culick speed in our units; identical to their u_c.
V_TC = math.sqrt(2.0)


def viscosity(oh: float) -> float:
    """Our coded dynamic viscosity mu for an Ohnesorge number Oh."""
    return oh * math.sqrt(2.0)


def tau_vis(mu: float) -> float:
    """Their viscous timescale in our units.

    This is the *viscous* clock: their curves collapse onto it only for
    ``Oh >~ 0.5``. At lower Oh the approach to u_c is inertial, set by rim
    growth in capillary time, so ``t* = t/tau_vis`` is not a meaningful target
    (``t* = 30`` at ``Oh = 0.035`` is ``t = 0.75`` -- the very start).
    """
    return mu / 2.0


def influence_length(mu: float) -> float:
    """Extent of the region the tip disturbs, in sheet thicknesses.

    Deka & Pierson (PRF 5, 093603, 2020): "the region of influence of the tip
    motion is proportional to Oh". Their Ohnesorge number is ``mu/sqrt(rho
    sigma h0)``, i.e. our coded ``mu``, NOT this project's ``Oh``; the argument
    and return value here are both ``mu``. Their Eq. (6) carries the inertial
    term with prefactor ``(L0/h0)^2 / mu^2``, giving pseudo-infinite
    (Taylor-Culick) behaviour for ``L/h0 >> mu`` -- LINEAR in mu. The square in
    that prefactor is the square of the ratio, not a different scaling.
    """
    return mu


def load_reference(figure: str, data_dir: Path) -> dict[float, np.ndarray]:
    """Digitised curves for ``fig5`` or ``fig6`` in ``data_dir``, keyed by Oh.

    Each value is an (N, 2) array of ``(tstar, ustar)``. Provenance and the
    digitisation uncertainty are recorded in each CSV header.
    """
    out: dict[float, np.ndarray] = {}
    for f in sorted(Path(data_dir).glob(f"savva_bush_2009_{figure}_Oh*.csv")):
        m = re.search(r"_Oh([0-9p.]+)", f.stem)
        if not m:
            continue
        oh = float(m.group(1).replace("p", "."))
        out[oh] = np.loadtxt(f, delimiter=",", comments="#")
    return out


SAFETY = 3.0  # multiples of the larger influence scale (see window_length)
MIN_WINDOW = 12.0  # never shorter than this many sheet thicknesses


def window_length(mu: float, t_end: float) -> float:
    """Co-moving window length, from the larger of two influence scales.

    Two things set how far the tip disturbs the sheet, and the window must
    contain the larger:

    1. The STEADY Stokes length ``Oh * h0`` -- Deka & Pierson's "region of
       influence of the tip motion is proportional to Oh".
    2. The TRANSIENT diffusive spread ``sqrt(4 nu t)``, which GROWS with run
       length. In our units ``nu = mu``, so this is ``sqrt(4 mu t_end)``.

    The second dominates for any long run and was missing from the first
    version of this function, which sized windows from (1) alone. That is why
    an Oh = 1 run at L = 12 gave u*(t*=30) = 0.819 while L = 25 gave 0.949:
    the window was shorter than the diffusive footprint and the inlet clamp
    was doing work it should not have been.

    Because the requirement grows as sqrt(t_end), pushing to larger t* to get
    closer to V_TC needs a proportionally longer window -- the co-moving frame
    removes the need to span the retraction PATH, but not the need to span the
    disturbed REGION.
    """
    diffusive = math.sqrt(4.0 * mu * t_end)
    return max(MIN_WINDOW, SAFETY * max(influence_length(mu), diffusive))
