"""Run the planar Newtonian retraction sweep that Savva & Bush figures 5 and 6 test.

One case per Ohnesorge number, each marched to ``t* = t/tau_vis = 30`` so it
spans both published figures (figure 5 covers ``t* <= 1``, figure 6 ``t* <= 30``).
The recovered control velocity ``U(t)`` is written as ``control.txt`` per case
and compared against the digitised curves by ``compare_savva_bush.py``.

Conventions, which differ from ours and are the usual source of error:

    Oh_SB    = mu / sqrt(2 H rho gamma)        (ours omits the sqrt(2))
    u_c      = sqrt(2 gamma / (rho H))         (identical to our V_TC = sqrt(2))
    tau_vis  = mu H / (2 gamma)                (the *viscous* clock)

with ``H`` the FULL sheet thickness. In our units (rho = gamma = H = 1) this
gives ``mu = sqrt(2) Oh_SB`` and ``tau_vis = mu/2``.

DOMAIN LENGTH. The co-moving window must contain the region the tip disturbs.
Deka & Pierson (PRF 5, 093603, 2020) state it directly -- "the region of
influence of the tip motion is proportional to Oh" -- and their Eq. (6) carries
the inertial term with prefactor ``(L0/h0)^2 / Oh^2``, so the sheet behaves as
pseudo-infinite (and reaches the Taylor-Culick speed) when

    L / h0  >>  Oh .

The dependence is LINEAR in Oh, not quadratic: the square in that prefactor is
the square of the ratio, not a different scaling. ``window_length`` below
applies a safety factor to that requirement and is reported per case so the
assumption is visible rather than implicit.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "simulationCases"))

from savva_bush import V_TC, influence_length, tau_vis, viscosity  # noqa: E402
from planar_comoving import PlanarTaylorCulickProblem  # noqa: E402

# Ohnesorge numbers in the SAVVA & BUSH convention, matching their figures.
# Figure 5 shows Oh = 0.5, 1, 5, >100; figure 6 shows Oh = 0.5, 1, 10, 100/1000.
# The very large values are omitted: they are indistinguishable from each other
# in the source figures and cost the most to run.
DEFAULT_OH_SB = [0.5, 1.0, 5.0, 10.0]

TSTAR_END = 30.0  # spans both figures
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
    an Oh_SB = 1 run at L = 12 gave u*(t*=30) = 0.819 while L = 25 gave 0.949:
    the window was shorter than the diffusive footprint and the inlet clamp
    was doing work it should not have been.

    Because the requirement grows as sqrt(t_end), pushing to larger t* to get
    closer to V_TC needs a proportionally longer window -- the co-moving frame
    removes the need to span the retraction PATH, but not the need to span the
    disturbed REGION.
    """
    diffusive = math.sqrt(4.0 * mu * t_end)
    return max(MIN_WINDOW, SAFETY * max(influence_length(mu), diffusive))


def run_case(oh_sb: float, outdir: Path, resolution: float, tstar_end: float,
             length: float | None = None, temporal_error: float = 1e-3) -> dict:
    mu = viscosity(oh_sb)
    tau = tau_vis(mu)
    t_end = tstar_end * tau
    L = length if length is not None else window_length(mu, t_end)

    meta = {
        "Oh_savva_bush": oh_sb,
        "Oh_ours": mu,  # mu/sqrt(rho*gamma*h0) with rho=gamma=h0=1
        "mu_coded": mu,
        "tau_vis": tau,
        "tstar_end": tstar_end,
        "t_end": t_end,
        "domain_length": L,
        "L_over_Oh": L / mu,
        "L_over_diffusive": L / math.sqrt(4.0 * mu * t_end),
        "resolution": resolution,
        "temporal_error": temporal_error,
        "V_TC": V_TC,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "case.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[Oh_SB={oh_sb}] mu={mu:.5f} tau_vis={tau:.5f} t_end={t_end:.3f} "
        f"L={L:.1f} (L/Oh={L / mu:.1f})",
        flush=True,
    )

    os.chdir(outdir)
    with PlanarTaylorCulickProblem() as problem:
        problem.Oh = mu
        problem.domain_length = L
        problem.resolution = resolution
        problem.initialise()

        history = problem.create_text_file_output(
            "control.txt", header=["t", "tstar", "U", "U_over_VTC"]
        )
        # GEOMETRIC output spacing, not uniform. The interesting structure is at
        # early time -- u* rises through most of its range in the first few per
        # cent of the run -- and in the inertial regime tau_vis is so small that
        # uniform sampling in t gives a step of several units in t*, which
        # renders the early curve as straight segments. Geometric spacing puts
        # the samples where the curve actually bends.
        out_times = np.geomspace(t_end * 1e-4, t_end, 400)
        prev = 0.0
        for t in out_times:
            problem.run(t, startstep=max((t - prev) * 1e-3, 1e-9),
                        maxstep=max(t - prev, 1e-9), temporal_error=temporal_error)
            u = problem.control_velocity()
            now = float(problem.get_current_time(as_float=True))
            history.add_row(now, now / tau, u, u / V_TC)
            prev = t
        meta["U_final"] = problem.control_velocity()
        meta["ustar_final"] = meta["U_final"] / V_TC

    (outdir / "case.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oh", type=float, nargs="*", default=None,
                    help="Ohnesorge numbers in the Savva-Bush convention")
    ap.add_argument("--results", type=Path,
                    default=Path("/home/vatsal/running-simulations/elastic-taylor-culick/"
                                 "newtonian-comoving-pyoomph/validation"))
    ap.add_argument("--resolution", type=float, default=0.06)
    ap.add_argument("--length", type=float, default=None,
                    help="override the Oh-derived co-moving window length")
    ap.add_argument("--temporal-error", type=float, default=1e-3)
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the case directory, for convergence studies")
    ap.add_argument("--tstar-end", type=float, default=TSTAR_END)
    args = ap.parse_args()

    oh_list = args.oh if args.oh else DEFAULT_OH_SB
    root = args.results.resolve()
    for oh in oh_list:
        name = f"Oh_SB_{oh:g}" + (f"_{args.tag}" if args.tag else "")
        run_case(oh, root / name, args.resolution, args.tstar_end,
                 args.length, args.temporal_error)


if __name__ == "__main__":
    main()
