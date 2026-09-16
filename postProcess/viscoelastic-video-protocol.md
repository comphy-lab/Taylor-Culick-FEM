# Axisymmetric viscoelastic video protocol

`animate_viscoelastic_comoving.py` renders one axisymmetric Oldroyd-B run as a
four-panel video. It is intended for the $G$–$\lambda$ formulation produced by
`validation/run_viscoelastic_axi.py` and reads the physical parameters from
the run's `case.json`.

## Panel contract

- Upper left: a horizontally reflected rim-following window, with the tip at
  the right and undisturbed film extending left. The upper half shows
  $|\mathbf u|/V_\mathrm{TC}$ in `Blues`; the lower half shows the solvent
  dissipation $2\mu\mathbf D:\mathbf D$ in `hot_r`, from white at zero to
  black at high dissipation.
- Upper right: the unreflected rim-following window, with the tip at the left.
  The upper half shows $\log_{10} A_{\theta\theta}$ in `RdBu_r`; the lower
  half shows $\log_{10}[\operatorname{tr}(\mathbf A)/3]$ in `PuOr_r`. Both
  displays saturate at −1 and 1.
- Lower left: $R_\mathrm{tip}/h_0$ against capillary time.
- Lower right: $V_\mathrm{tip}/V_\mathrm{TC}$ against
  $R_\mathrm{tip}/h_0$.

Both field views follow the rim spatially. Only the lower-left history is a
lab-frame trajectory. The velocity components themselves are the lab-frame
components stored by the axisymmetric ALE solver.

The axisymmetric dissipation includes the hoop term,

$$
2\mu\mathbf D:\mathbf D = 2\mu\left(D_{rr}^2+D_{zz}^2+D_{\theta\theta}^2
  +2D_{rz}^2\right),\qquad D_{\theta\theta}=u_r/r.
$$

## Initial state and playback

If no stored $t=0$ VTU exists, the default `--initial-state auto` reconstructs
the prescribed initial state from the saved Gmsh mesh:
$R_\mathrm{tip}=R_0$, $\mathbf u=0$, and
$\boldsymbol{\Psi}=\log(\mathbf A)=0$. The receipt identifies this as a
reconstruction rather than a solver output. A stored $t=0$ VTU takes
precedence; `--initial-state omit` suppresses reconstruction.

At the default 30 fps, $t=0$ is held for one second and every stored state
through $t=0.2$ for half a second. This makes the resolved startup visible;
later outputs retain one video frame per stored state. Hold durations are
specified in seconds and therefore remain correct when `--fps` changes.

## Accepted rendering command

```bash
uv run python postProcess/animate_viscoelastic_comoving.py \
  --run-dir <case>/run_viscoelastic_axi \
  --frames-dir <work>/frames \
  --out <work>/viscoelastic.mp4 \
  --fps 30 --procs 8 \
  --radius-limits 0.8 4.8 \
  --tip-speed-limits -0.025 0.275
```

The explicit history limits reproduce the accepted $Oh=1$, $Ec=1$,
$\lambda=\infty$ presentation. Omit them for padded limits derived from a
different run. The velocity-field colour range defaults to 0–0.25; pass
`--speed-limits 0 1` when comparing cases on the shared 0–1 scale.

A restarted trajectory may be assembled without copying or rewriting its PVD
files. Supply the segments in chronological order, repeating `--segment-pvd`:

```bash
uv run python postProcess/animate_viscoelastic_comoving.py \
  --run-dir <case>/run_viscoelastic_axi \
  --segment-pvd <case>/attempt-1/run_viscoelastic_axi/liquid.pvd \
  --segment-pvd <case>/attempt-2/run_viscoelastic_axi/liquid.pvd \
  --frames-dir <work>/frames \
  --out <work>/viscoelastic.mp4 \
  --speed-limits 0 1
```

Without `--segment-pvd`, the source remains `<run-dir>/liquid.pvd`. Each
segment must contain finite, strictly increasing times, every referenced VTU
must exist, and the first time in each later segment must exceed the final time
in the preceding segment. The script refuses overlaps as well as missing
arrays, invalid conformation values, incomplete histories, stale frame files,
and partial-frame encoding.

## Verification

Before retaining a render:

1. Render and inspect $t=0$, every stored state through $t=0.2$, one middle
   state, and the final state.
2. Encode the complete ordered frame set, then decode and inspect the same
   physical states.
3. Confirm H.264 `yuv420p`, 1600×912 pixels, the expected frame count, and no
   post-encode rescaling.
4. Retain the generated JSON receipt beside the MP4. It records metadata and
   every source PVD path and SHA-256 identity, field definitions, colour
   limits, playback holds, frame count, codec probe, and SHA-256 digest.
