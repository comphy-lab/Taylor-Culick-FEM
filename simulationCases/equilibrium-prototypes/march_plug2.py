# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Marching-plug driver for the window prototypes, version 2.

Runs a window segment script segment by segment for one chi* family. Each segment cuts the window at the sheet
column nearest X_c = chi* sqrt(R_S) (R_S the radius of the segment's start state, never moving the cut inwards),
calibrates the plug on the start state, and continues through the next targets whose theoretical radius
R(Ec) = exp(2/Ec - C + D_cap) stays within GROWTH x R_S (at least one target). The next segment starts from the
last point. The family summary (family.json) records every point with its segment, cut and calibration.
Version 2 adds GRIDS (the reference grid, passed through to the segment script, which must accept it) and QSCRIPT
(the segment script; Q4B is the version-1 name of the same variable).

usage: march_plug2.py <family-dir> <chi*> <start-state-folder> <X_m> <targets...>
env: QSCRIPT (segment script), PY (interpreter), GRIDS (reference grid, default SEED/grids.npz), SEED (folder of the
retained seed segment; required), GROWTH (default 2). The environment, including CASE and the segment script's
own options (STEP, NEWTON_TOL, STALL_CAP, DTOL_Z, ...), is passed to every segment.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

C_MINUS_DCAP = 0.472506669349 - 0.097162998572
SEG_GRIDS = Path(os.environ.get("GRIDS", str(Path(os.environ["SEED"]) / "grids.npz")))


def r_theory(ec):
    return math.exp(2 / ec - C_MINUS_DCAP)


def main():
    fam, chi = Path(sys.argv[1]), float(sys.argv[2])
    state, x_m = Path(sys.argv[3]), sys.argv[4]
    targets = sorted((float(t) for t in sys.argv[5:]), reverse=True)
    growth = float(os.environ.get("GROWTH", "2"))
    q4b = os.environ.get("QSCRIPT") or os.environ["Q4B"]
    py = os.environ["PY"]
    fam.mkdir(parents=True, exist_ok=False)
    columns = np.load(SEG_GRIDS)["sheet"][:, 0, 0]
    saved = np.load(state / "state.npz")
    r_start = float(np.min(saved["Xe"][..., 0] + saved["ue"][..., 0]))
    ec_start = math.exp(float(saved["log_ec"]))
    summary = {"chi_star": chi, "start": str(state), "start_Ec": ec_start, "start_R": r_start, "X_m": float(x_m),
               "growth": growth, "segments": [], "points": []}
    i_prev = 0
    k = 0
    while targets:
        seg_targets = [t for t in targets if r_theory(t) <= growth * r_start] or targets[:1]
        i_c = int(np.argmin(np.abs(columns - chi * math.sqrt(r_start))))
        i_c = max(i_c - i_c % 2, i_prev)
        out = fam / f"seg{k:02d}-i{i_c}"
        env = dict(os.environ, INNER=str(i_c), START=str(state))
        env.pop("F_NOSE", None)
        env.pop("C_NOSE", None)
        cmd = [py, "-u", q4b, str(out), x_m, *[f"{t:g}" for t in seg_targets]]
        with open(fam / f"seg{k:02d}-i{i_c}.log", "w") as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
        seg = {"segment": k, "i_c": i_c, "X_c": float(columns[i_c]), "chi_at_start": float(columns[i_c]) / math.sqrt(r_start),
               "start": str(state), "start_R": r_start, "targets": seg_targets, "exit": code,
               "c_nose": res.get("inner", {}).get("c_nose"), "calibration": res.get("calibration"),
               "failure": res.get("failure", {}).get("Ec") if res.get("failure") else None}
        summary["segments"].append(seg)
        for p in res.get("points", []):
            summary["points"].append({"Ec": p["Ec"], "R": p["R"], "H": p["H"], "segment": k, "i_c": i_c,
                                      "X_c": float(columns[i_c]), "chi": float(columns[i_c]) / math.sqrt(p["R"])})
        (fam / "family.json").write_text(json.dumps(summary, indent=1) + "\n")
        done = [p["Ec"] for p in res.get("points", [])]
        if code != 0 or len(done) < len(seg_targets):
            print("family stopped in segment", k, "exit", code, flush=True)
            return
        targets = [t for t in targets if t not in seg_targets]
        state = out / f"Ec{seg_targets[-1]:g}"
        r_start = res["points"][-1]["R"]
        i_prev = i_c
        k += 1
        print("segment", k - 1, "done:", [(round(p["Ec"], 4), round(p["R"], 3), round(p["H"], 9)) for p in res["points"]], flush=True)
    print("family complete", flush=True)


if __name__ == "__main__":
    main()
