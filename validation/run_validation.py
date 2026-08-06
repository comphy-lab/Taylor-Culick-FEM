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
import sys
from pathlib import Path

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
SAFETY = 8.0  # window length in units of the Oh-proportional influence region
MIN_WINDOW = 12.0  # never shorter than this many sheet thicknesses


def window_length(mu: float) -> float:
    """Co-moving window length satisfying L >> Oh*h0 with a safety factor."""
    return max(MIN_WINDOW, SAFETY * influence_length(mu))


def run_case(oh_sb: float, outdir: Path, resolution: float, tstar_end: float) -> dict:
    mu = viscosity(oh_sb)
    tau = tau_vis(mu)
    t_end = tstar_end * tau
    L = window_length(mu)

    meta = {
        "Oh_savva_bush": oh_sb,
        "Oh_ours": mu,  # mu/sqrt(rho*gamma*h0) with rho=gamma=h0=1
        "mu_coded": mu,
        "tau_vis": tau,
        "tstar_end": tstar_end,
        "t_end": t_end,
        "domain_length": L,
        "L_over_Oh": L / mu,
        "resolution": resolution,
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
        t, dt_out = 0.0, t_end / 300.0
        while t < t_end - 1e-12:
            t = min(t + dt_out, t_end)
            problem.run(t, startstep=dt_out * 1e-3, maxstep=dt_out, temporal_error=1e-3)
            u = problem.control_velocity()
            now = float(problem.get_current_time(as_float=True))
            history.add_row(now, now / tau, u, u / V_TC)
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
    ap.add_argument("--tstar-end", type=float, default=TSTAR_END)
    args = ap.parse_args()

    oh_list = args.oh if args.oh else DEFAULT_OH_SB
    root = args.results.resolve()
    for oh in oh_list:
        run_case(oh, root / f"Oh_SB_{oh:g}", args.resolution, args.tstar_end)


if __name__ == "__main__":
    main()
