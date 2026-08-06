# Validation — planar Newtonian retraction

## The claim

> The co-moving sharp-interface FEM formulation reproduces the published edge
> velocity `u*(t*)` of a retracting planar Newtonian sheet across Ohnesorge
> number, over the whole transient, to within stated uncertainty.

This is **validation**, not verification: the comparator is data produced
independently of this code. Read the distinction in `../AGENTS.md` before
reclassifying anything in here.

## What makes the comparator independent

Savva & Bush (JFM **626** (2009) 211–240) solve a **one-dimensional
long-wavelength (Trouton) lubrication model** with their own numerics. This
code solves the **two-dimensional free-surface Navier–Stokes equations** with a
sharp interface on a moving mesh. Different equations, different discretisation,
different implementation, different authors — nothing is shared, so agreement is
meaningful and disagreement is informative.

That independence carries a limit worth stating plainly: where the two disagree,
it may be the lubrication approximation failing rather than this code. The
approximation is weakest where the interface is steep — the low-`Oh` bulbous rim
— and strongest in the viscous regime where the sheet stays flat. Treat the
high-`Oh` cases as the sharper test of *this* code.

The dashed curve `u* = sqrt(t*/pi)` in figure 5 is their analytic early-time
asymptote. It is used here as the **digitisation calibration check**, not as a
physics target: if the extracted dashed curve does not reproduce
`sqrt(t*/pi)`, the axis transform is wrong.

## Conventions — the trap

Their non-dimensionalisation is not ours. Getting this wrong moves results by
`sqrt(2)` or by orders of magnitude in time:

| quantity | Savva & Bush | ours |
|---|---|---|
| Ohnesorge | `Oh = mu / sqrt(2 H rho gamma)` | `mu / sqrt(rho gamma h0)` |
| velocity | `u_c = sqrt(2 gamma / (rho H))` | `V_TC = sqrt(2)` — **identical** |
| time | `tau_vis = mu H / (2 gamma)` | capillary `sqrt(rho h0^3/gamma)` = 1 |

`H` is the **full** sheet thickness. So our coded `Oh = 0.05` is their `0.035`,
and `mu = sqrt(2) * Oh_SB` in our units.

`tau_vis` is the **viscous** clock. Their curves collapse onto it only for
`Oh >~ 0.5`. At low `Oh` the approach to `u_c` is inertial, set by rim growth in
capillary time, and `t*` is not a meaningful target — `t* = 30` at `Oh_SB=0.035`
is `t = 0.75`, which is the very start of the transient.

## Domain length

The co-moving window must contain the region the tip disturbs. Deka & Pierson
(PRF **5**, 093603, 2020) state that "the region of influence of the tip motion
is proportional to `Oh`", and their Eq. (6) carries the inertial term with
prefactor `(L0/h0)^2 / Oh^2`, giving pseudo-infinite (Taylor–Culick) behaviour
when

```
L / h0  >>  Oh
```

**linear in `Oh`.** The square in that prefactor is the square of the ratio, not
a different scaling — a finite sheet with `Oh >~ L0/h0` instead shows a
*decreasing* tip speed and never reaches `u_c`. `run_validation.py` sizes the
window from this requirement with a safety factor and records `L/Oh` per case,
so the assumption is auditable rather than implicit.

Because the Lagrange-multiplier control feeds undisturbed film in at `-U`, the
*lab-frame* sheet is effectively semi-infinite regardless of window size; the
window only has to be long enough that the disturbance decays before the inlet.

## Layout

```
literature/   source papers (kept, not just cited)
data/         digitised reference curves + provenance headers
              digitize_savva_bush.py regenerates them
run_validation.py       runs the Oh sweep to t* = 30
compare_savva_bush.py   quantitative comparison + figure
```

Simulation output does **not** live here — runs write to the project run root on
the compute host and the tracker records custody.

## Running

```bash
uv run python validation/run_validation.py                 # default Oh sweep
uv run python validation/run_validation.py --oh 0.5 1 5    # subset
uv run python validation/compare_savva_bush.py             # compare + plot
```

## Status

| `Oh_SB` | `u*` at `t* = 30`, this code | digitised Savva & Bush | difference |
|---|---|---|---|
| 0.5 | 0.92966 | 0.9321 | **0.26 %** |

That difference is inside the digitisation uncertainty (~0.5 % of the `u*`
range), so it should be read as "agrees to the precision of the comparator",
not as a measured 0.26 % discrepancy. An earlier 0.04 % figure quoted here was
against a rounded eyeball reading of the printed figure rather than the
digitised curve, and was spurious precision.

Remaining cases pending. A single agreeing point is weak evidence — the whole
point of digitising both figures is to compare the entire `u*(t*)` curve across
`Oh`, which can actually falsify the formulation.

## Split

The planar problem is validated in two parts, because the physics and the
appropriate reference differ:

- `inertial/` — low `Oh`. Rim forms, capillary waves shed ahead of it. Savva &
  Bush only reach `Oh_SB = 0.5`, so the strongly inertial regime has no
  digitised comparator.
- `viscous/` — high `Oh`. No rim. Ahsan, Brandao, Davidovitch & Stone
  (arXiv:2507.04603) give the quasi-steady-shape / linearly-growing-amplitude
  structure; Savva & Bush give the `u*(t*)` curves.
