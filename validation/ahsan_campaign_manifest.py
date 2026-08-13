#!/usr/bin/env python3
"""Write MANIFEST.md for the 1D validation tree.

The inventory is generated rather than typed: every data file is listed with
the exact command that produced it, taken from ``logs/jobs.txt``, plus the
parameters the driver recorded in the matching ``*_meta.json``. Hand-written
manifests drift; this one cannot, because it reads the same job list the
campaign ran.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

PREAMBLE = r"""# 1D validation data set: TC1D-FEM vs ABDS-CN vs Ahsan et al. asymptotics

Generated on host `reynolds`. Nothing in this tree was produced on `stokes`.

## The two codes

| name | what it is | entry point |
|---|---|---|
| **TC1D-FEM** | ours. pyoomph FEM in a Lagrangian mass coordinate, staggered `h`=D0 / `u`=C1, BDF2, `with_capillarity=False`. | `validation/ahsan_campaign.py tcfem` driving `simulationCases/trouton1d_pyoomph.py` |
| **ABDS-CN** | theirs. numpy port of the modified Crank-Nicolson MATLAB released with Ahsan, Brandao, Davidovitch & Stone, arXiv:2507.04603. Two independent schemes: `reduced-inf` (tip frame, L = infinity, their Sec. VII) and `reduced-fin` (static frame, finite L, their Appendix D). | `validation/ahsan_campaign.py abds` driving `validation/ahsan_thinfilm.py` |

Theory lines are their eqs (5.10), (6.10), (7.11), (7.13), (7.14), (3.10a),
(7.17); every CSV that can carry one does, evaluated on its own abscissa, so no
downstream consumer has to re-derive them.

## Unit conversion

Re-verified symbolically at the start of this campaign by
`validation/ahsan_campaign.py check-units`, which passes all six checks:

    Oh_paper = 2 Oh_SB                  their Oh is built on the HALF thickness h0 = H_full/2
    T        = t*/4                     tI = 4 mu h0/gamma = 4 tau_vis
    V        = u* = u/V_TC              their V and our u* are the same number, no factor
    H        = h                        their H is h_half/h0, ours h_full/H_full: same number
    lI       = 4 h0 Oh_paper = 4 Oh_SB  in our code units, where rho = gamma = H_full = 1
    L        = l0/lI = M/(4 Oh_SB)      M is our Lagrangian mass domain
    X - X_c  = (x - x_tip)/(4 Oh_SB)

Pushing our momentum equation `u_t + u u_x = (4 mu/(rho h))(h u_x)_x` through
this substitution returns their (2.16) `H V_T + H V V_X = (H V_X)_X` with
residual exactly zero.

## The finite sheet in TC1D-FEM

Their finite-L problem imposes `V(L,T) = 0` and `H_X(L,T) = 0` (their 2.19a,b).
Our staggered scheme imposes `DirichletBC(u=0)` at `m = M` with `h`
cell-constant and therefore Neumann. These are the same conditions, so a
TC1D-FEM run with mass `M` **is** their sheet of `L = M/(4 Oh_SB)`; there is no
structural obstacle and no theory-only panel anywhere in this tree. The mapping
was checked before the campaign was launched rather than assumed: at
`Oh_SB = 1, M = 20` the deceleration lands at `T_c = 4.97` against `L = 5`.

`drift_H_far` in the TC1D-FEM series is `|h - 1|` at `m = M`. In the finite-L
sections it is LARGE by design -- it is the disturbance arriving at the far end,
which is the physics of section 2, and it approaches `T` because the whole sheet
thickens to `1+T`. It is a domain-adequacy failure ONLY in section 3, where it
must stay near zero, and it does.

## Column conventions

Every series CSV has a header line. Common columns:

| column | meaning |
|---|---|
| `T` | their time, `T = t/tI` |
| `tstar` | our time, `t* = t/tau_vis = 4T` |
| `V` | `dX_c/dT`, i.e. our `u*` -- the same number in both codes |
| `Xc` | free-edge position in units of `lI` |
| `L_minus_Xc` | current sheet length `L - X_c`, in units of `lI` |
| `H_max`, `H_tip` | thickness at the rim |
| `H_tip_exact` | `1 + T`, their (3.10a) |
| `U_eq5p10`, `U_eq6p10`, `U_eq7p11` | theory lines evaluated at the same `T` |
| `L_minus_Xc_eq6p10`, `Xc_eq7p17` | integrated theory lines |
| `invariant_max`, `invariant_rms` | TC1D-FEM only. Residual of their exact invariant (3.1) `H V + H_X = 0`, which in our mass coordinate is `V = -lI dh/dm`. TC1D-FEM never uses this relation, so it is an independent correctness check, not a tautology. |
| `drift_V_far`, `drift_H_far` | TC1D-FEM only. `\|u\|` and `\|h-1\|` at `m = M`. `drift_V_far` is identically zero because `u = 0` there is a Dirichlet condition; `drift_H_far` is the informative one. |

Profile CSVs are per `(case, T)`. TC1D-FEM writes two, because `V` lives on
nodes and `H` is cell-constant and interpolating one onto the other would
invent structure the discretisation does not have:

    <tag>_profile_T<T>_V.csv   Xhat, X, V, m_code, V_eq7p14, V_eq6p7
    <tag>_profile_T<T>_H.csv   Xhat, X, H, m_code, H_eq7p13, H_eq6p3

ABDS-CN writes one, since both live on its uniform grid:

    <tag>_profile_T<T>.csv     Xhat, X, H, V, H_eq7p13, V_eq7p14 [, H_eq6p3, V_eq6p7]

`Xhat = X - X_c` is the tip frame (their Sec. VII); `X` is the lab frame, the
abscissa of their fig. 2. `m_code` is our Lagrangian mass label, in code units.

## Sections

- `sec1-early/` -- T << 1, finite L = 0.1, 1, 20, against (5.10) `2 sqrt(T/pi)`.
  Each case is TWO runs spliced at `T = 1e-2`: no single timestep resolves
  `T = 1e-4` and `T = 1` at once.
- `sec2-late-finite/` -- `L - X_c << 1`, L = 1, 5, 20, 40, against (6.10)
  `L/(1+T)^2`, plus one case at L = 5 pushed to T = 400 to show the RATE of
  approach, plus the L = 0.1 and L = 1 profile cases of their fig. 2(a-d).
- `sec3-main/` -- `L >> T >> 1` at L = 200, T <= 50, against (7.11), (3.10a),
  (7.13), (7.14), (7.17); plus resolution twins and an Oh-independence check.
- `sec4-fig5/` -- their fig. 5, `T_c(L)` and `Delta T_c(L)` over L = 3 to 50.
- `sec3-refine/` -- the ABDS-CN resolution sequence that settles the value of
  `U_-2`. Their tip-frame scheme at dX = 0.02 (their released default), 0.01,
  0.005 and 0.0025, plus a dt twin and a truncation-length twin. At dX = 0.02
  the deficit `1 - dXc/dT` comes out NEGATIVE by T = 40; the sequence converges
  at second order to the TC1D-FEM answer.
- `figures/` -- the publication figures, PDF plus PNG.
- `appendix-port-fidelity/` -- their original MATLAB under GNU Octave 10.3
  against ABDS-CN at identical parameters.
- `analysis/` -- reductions written by `validation/ahsan_campaign_report.py`.
  Nothing here is a new computation; it is the campaign CSVs combined.
- `logs/` -- `jobs.txt` (every command), `run.sh` (the runner), one log per
  case, `campaign.log` (start/OK/FAIL per case).

## T_c and Delta T_c: the definition used

Their paper (Sec. VIII) says only that `T_c` is where "the retraction speed
U(T) falls most sharply ... the minimum of the derivative dU/dT", and that
`Delta T_c` "is estimated as the full width at half maximum (FWHM) of this
minimum". Their released `thin_film_Oh_large.m` is explicit, and this campaign
uses THEIR CODE'S definition verbatim for both codes:

    a = D(U, dT);  a = -a .* (a < 0);          % deceleration only
    [amax, i] = max(a);   T_c = T(i);
    half = amax/2;
    i1 = find(a >= half, 1, 'first');  i2 = find(a >= half, 1, 'last');
    Delta T_c = 2 * (T(i2) - T(i1));

Note that this is TWICE the FWHM, not the FWHM as the paper's text says. Fig.
5(b)'s ordinate runs 2.4 to 3.1, which matches the code's `duration`, so the
plotted quantity is `2 x FWHM`. Both are reported: `dTc_code` reproduces their
figure, `fwhm` is what their text describes. `dTc_local` restricts the
half-maximum search to the contiguous run through the peak, since their
`find(...,'first'/'last')` scans the whole record.

## Reproducing

    cd ~/CMP-codes/Taylor-Culick-FEM
    export ROOT=<this directory> UV=~/.local/bin/uv
    python3 validation/ahsan_campaign_jobs.py | envsubst > $ROOT/logs/jobs.txt
    $ROOT/logs/run.sh                       # 11-way parallel, ~1 h on reynolds
    $UV run python validation/ahsan_campaign.py fig5 ...   # see logs/jobs3.txt
    $UV run python validation/ahsan_campaign_report.py $ROOT

Environment for every command: `HWLOC_COMPONENTS=-gl MPLBACKEND=Agg
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

Each case gets its OWN `--out` directory and the runner promotes the CSVs up
afterwards. This is load-bearing: pyoomph compiles and dlopens a shared object
into `<--out>/_ccode`, so concurrent cases sharing a directory compile over each
other and segfault.

---

## File inventory
"""


def main() -> None:
    jobs = {}
    jf = ROOT / "logs" / "jobs.txt"
    if jf.exists():
        for line in jf.read_text().splitlines():
            m = re.search(r"--tag (\S+)", line)
            if m:
                jobs[m.group(1)] = line.strip()
    for extra in ("jobs2.txt", "jobs3.txt", "jobs4.txt", "jobs5.txt",
                  "jobs6.txt"):
        p = ROOT / "logs" / extra
        if p.exists():
            for line in p.read_text().splitlines():
                m = re.search(r"--tag (\S+)", line)
                if m:
                    jobs[m.group(1)] = line.strip()

    out = [PREAMBLE]
    for sec in sorted(d for d in ROOT.iterdir()
                      if d.is_dir() and d.name not in ("logs",)):
        files = sorted(f for f in sec.iterdir() if f.suffix in (".csv", ".json"))
        if not files:
            continue
        out.append(f"\n### `{sec.name}/`\n")
        tags = sorted({f.name.split("_series")[0].split("_profile")[0]
                       .split("_meta")[0] for f in files})
        for tag in tags:
            grp = [f for f in files if f.name.startswith(tag + "_")
                   or f.name == tag + ".csv"]
            if not grp:
                continue
            meta = sec / f"{tag}_meta.json"
            out.append(f"\n**`{tag}`**\n")
            if meta.exists():
                try:
                    j = json.loads(meta.read_text())
                except json.JSONDecodeError:
                    j = {}
                keys = [k for k in ("code", "L", "L_reported", "Oh_SB",
                                    "Oh_paper", "mass", "nelem", "Tend",
                                    "dT", "nsteps", "N", "NT", "dx", "dt",
                                    "Lhat", "dX_wall", "tip_cell_mass",
                                    "spacing", "scheme", "wall_s")
                        if k in j]
                out.append("- parameters: "
                           + ", ".join(f"`{k}={j[k]}`" for k in keys) + "\n")
            if tag in jobs:
                cmd = jobs[tag].replace(str(ROOT), "$ROOT")
                cmd = cmd.replace(str(Path.home()) + "/.local/bin/uv", "$UV")
                out.append(f"- command:\n\n      {cmd}\n")
            out.append("- files: "
                       + ", ".join(f"`{f.name}`" for f in grp) + "\n")

    (ROOT / "MANIFEST.md").write_text("".join(out))
    print(f"wrote {ROOT/'MANIFEST.md'}  ({len(out)} blocks)")


if __name__ == "__main__":
    main()
