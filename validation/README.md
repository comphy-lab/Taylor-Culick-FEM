# Validation — planar Newtonian retraction

## Scope: `Oh <= 1`

The campaign is bounded above at `Oh = 1` (Vatsal, 2026-08-07). The reason is
the window-sizing rule: the co-moving window must contain the diffusive
footprint `sqrt(4 nu t)`, which scales with `mu`, so at `t* = 100`

| `Oh` | required `L/h0` |
|---|---|
| 0.035 | 13 |
| 0.5 | 52 |
| 1 | 104 |
| 5 | 300 |
| 10 | 600 |

Beyond `Oh = 1` the cost grows sharply for the least interesting physics --
the strongly viscous limit where no rim forms and the curves collapse onto one
another. `Oh ~ 1` is where inertia and viscosity are comparable and the
dynamics are richest.

Consequence for the comparison matrix: the digitised Savva & Bush curves used
are `Oh = 0.5` and `Oh = 1`; their `Oh = 5, 10, 100` curves and the Ahsan et al.
high-`Oh` asymptotics are now out of DNS scope. The 1D Trouton solver is cheap
enough to run at high `Oh` regardless, and its `Oh = 100` point is retained as
evidence on the digitisation question (see Status), not as DNS scope.

## Campaign goal

Two questions, asked separately in each regime because the physics and the
appropriate reference differ:

> **Q1. Is the result numerically converged?**
> **Q2. Do we recover the Taylor-Culick velocity?**

and then, for each regime, a comparison matrix. The cells are NOT equivalent
evidence, and the difference matters when they disagree:

### Low `Oh` (inertial) -- `inertial/`

| compare against | what it can establish | caveat |
|---|---|---|
| our 1D Trouton solver | whether a discrepancy is model or code | **not independent** -- our implementation of their model; a shared misreading satisfies both |
| fresh Basilisk DNS | genuine cross-method agreement | independent: VOF interface capturing, two-phase, different mesh, different code |
| Wee et al. PRF 9, 103601 (2024) | published low-`Oh` behaviour | **finite drop** (`L0 = 100`, both ends), not semi-infinite; comparable only where `Oh << L0/h0` |

Savva & Bush reach only `Oh = 0.5`, past their own `Oh < 0.1` wave threshold,
so the strongly inertial regime has **no digitised comparator** -- Q2 there is
answered against `V_TC` itself plus convergence.

### High `Oh` (viscous) -- `viscous/`

| compare against | what it can establish | caveat |
|---|---|---|
| our 1D Trouton solver | whether a discrepancy is model or code | same non-independence caveat |
| Savva & Bush figs 5, 6 | whole-curve `u*(t*)` agreement | 1D lubrication vs our 2D NS; their approximation is *strongest* here, so a disagreement most likely indicts us |
| Ahsan et al. arXiv:2507.04603 | asymptotic **structure** | not a `u*(t*)` curve: compare the exponential profile with Stokes-length decay, the linear amplitude growth, and the `O(1/T^2)` velocity convergence |
| Ahsan et al. **solver**, ported in `ahsan_thinfilm.py` | whole-curve `u*(t*)` against an INDEPENDENT implementation of the same equations | independent code, independent discretisation, independent authors -- but the same continuous model, so it cannot indict the model itself |

### Ahsan et al. as an independent 1D implementation (2026-08-11)

`ahsan_thinfilm.py` is a faithful Python port of their released MATLAB
(`github.com/taosifahsan/Thin-Film-Simulations`), verified against the original
run under GNU Octave to 5e-10, the precision of Octave's ASCII output.

The load-bearing fact is that **their reduced problem is our model exactly**.
Their eqs (2.15)-(2.19), with `h0 = H/2`, `Oh_paper = 2 Oh`, `T = t*/4` and
`U = u*`, are our `h_t + (h u)_x = 0`, `Du/Dt = (4 mu/h)(h u_x)_x` with the
Savva & Bush tip closure `h u_x = -1/(2 mu)` -- i.e. our solver's DEFAULT
configuration, `with_capillarity=False`. They drop the bulk capillary term
because they prove it negligible at `Oh >> 1` (their eq. 2.11), not for
convenience. Their eq (3.10a) `H(Xc,T) = 1+T` is our `h_tip = 1 + t*/4`.

Consequence, and it is checkable: the reduced problem has **no parameter but
`L`**, so `u*(t*)` must be independent of `Oh`. Measured across
`Oh = 0.2, 1, 5`: 4e-5 apart at `t* = 0.1`, 2e-7 by `t* >= 5`.

Result at `L = infinity`, `T` in `(1e-3, 50]` (`t*` to 200):

| quantity | value |
|---|---|
| max `\|du*\|`, pyoomph vs port | 7.1e-5 |
| RMS `\|du*\|` | 4.2e-5 |
| max relative | 5.5e-4 |
| port's own 1x-vs-2x resolution floor | 8.5e-5 |

The two agree at the joint discretisation floor. Refining the reference moves
it toward pyoomph, not away.

**Does `with_capillarity=False` cost anything?** Their `thin_film.m` is the
same problem WITH the bulk capillary term and a finite `Oh` tip, so it answers
this directly. Deviation of its `U` from the capillarity-free curve, at
`L = 10`, as the grid is refined:

| `T` | `Oh_p=10`, `dx=0.067` | `dx=0.022` | `dx=0.0074` | `Oh_p=40`, `dx=0.089` | `dx=0.030` |
|---|---|---|---|---|---|
| 0.20 | -15.0% | -9.6% | -6.6% | -14.5% | -7.9% |
| 0.50 | -13.5% | -5.6% | -2.0% | -14.8% | -5.3% |
| 0.75 | -13.2% | -4.2% | -0.1% | -15.1% | -4.5% |

Monotone in `dx` towards zero at fixed `Oh`, and the same at `Oh_p = 40`. The
double-digit gaps are their scheme failing to resolve its own hard-imposed
`sqrt` tip, not physics. So the neglected bulk capillary term is a sub-per-cent
effect here and does not explain -- or threaten -- the 7e-5 agreement above.
It is not converged at the finest grid run, so treat the residual 1-2% as an
upper bound rather than a measurement.

### Status against the goal

- Q1 answered for `Oh = 1`: mesh and time tolerance converged to 0.06 %;
  **domain length is the controlling parameter** and needs the diffusive rule.
- Q2 for `Oh < 0.1` is closed against `V_TC` and against Basilisk. FEM
  terminal `u*` is `0.994`–`0.996`; Basilisk at `Oh = 0.035`, `t = 500`
  is `u* ≈ 0.977`. See `docs/validation-inertial/`. The moderate-Oh approach
  to `V_TC` is the planar-campaign write-up.

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
| Ohnesorge | `Oh = mu / sqrt(2 H rho gamma)` | coded `mu = mu / sqrt(rho gamma h0)` |
| velocity | `u_c = sqrt(2 gamma / (rho H))` | `V_TC = sqrt(2)` — **identical** |
| time | `tau_vis = mu H / (2 gamma)` | capillary `sqrt(rho h0^3/gamma)` = 1 |

`H` is the **full** sheet thickness. So coded `mu = 0.05` is `Oh = 0.035`,
and `mu = sqrt(2) * Oh` in our units.

`tau_vis` is the **viscous** clock. Their curves collapse onto it only for
`Oh >~ 0.5`. At low `Oh` the approach to `u_c` is inertial, set by rim growth in
capillary time, and `t*` is not a meaningful target — `t* = 30` at `Oh=0.035`
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
uv run python validation/run_validation.py --results RUN_ROOT                 # default Oh sweep
uv run python validation/run_validation.py --results RUN_ROOT --oh 0.5 1 5    # subset
uv run python validation/compare_savva_bush.py --results RUN_ROOT             # compare + plot
```

## Status

| `Oh` | `u*` at `t* = 30`, this code | digitised Savva & Bush | difference |
|---|---|---|---|
| 0.5 | 0.92966 | 0.9321 | **0.26 %** |

That difference is inside the digitisation uncertainty (~0.5 % of the `u*`
range), so it should be read as "agrees to the precision of the comparator",
not as a measured 0.26 % discrepancy. An earlier 0.04 % figure quoted here was
against a rounded eyeball reading of the printed figure rather than the
digitised curve, and was spurious precision.

A single agreeing point is weak evidence. The point of digitising both
figures is to compare the entire `u*(t*)` curve across `Oh`, which can
falsify the formulation. The inertial planar comparison is now in
`docs/validation-inertial/`.

## Split

The planar problem is validated in two parts, because the physics and the
appropriate reference differ:

- `inertial/` — low `Oh`. Rim forms, capillary waves shed ahead of it. Savva &
  Bush only reach `Oh = 0.5`, so the strongly inertial regime has no
  digitised comparator.
- `viscous/` — high `Oh`. No rim. Ahsan, Brandao, Davidovitch & Stone
  (arXiv:2507.04603) give the quasi-steady-shape / linearly-growing-amplitude
  structure; Savva & Bush give the `u*(t*)` curves.
