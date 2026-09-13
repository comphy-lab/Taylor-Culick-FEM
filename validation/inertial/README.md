# Inertial (low `Oh`) planar retraction

## Claim

> In the inertia-dominated regime the co-moving formulation reproduces the
> published edge velocity `u*(t*)`, and the tip approaches a **constant**
> velocity `V_TC` while the rim continues to grow.

## Regime

`Oh <~ 0.1`. A bulbous, roughly circular rim forms and **capillary waves are
shed ahead of it** (Brenner & Gueyffier 1999; Savva & Bush 2009 quote the same
`Oh < 0.1` threshold). Both are visible in our runs — see the animation in
`postProcess/` — so they are physics, not artefacts.

## What is and is not steady

The tip speed tends to a constant; the rim does not stop growing. Mass
conservation forces the rim cross-section to grow like `~ t` in area, so its
radius grows `~ sqrt(t)` in planar geometry. Those two facts are compatible and
are exactly the classical argument: `d(mV)/dt = 2 sigma` with `m = rho h V t`
gives constant `V` *because* `m` grows.

So there is no steady state, and none is needed: we time-march and measure the
speed.

## Reference data

`../data/savva_bush_2009_fig5_Oh0.5.csv`, `..._fig6_Oh0.5.csv` — their lowest
plotted `Oh`. Note that `Oh = 0.5` is already past the `Oh < 0.1` wave
threshold, so **the published curves do not reach the strongly inertial
regime**. For `Oh <~ 0.1` there is no digitised comparator here and the check
is against `V_TC` itself plus internal convergence.

## The `tau_vis` trap

`t* = t/tau_vis` is the **viscous** clock. At `Oh = 0.035` it gives
`tau_vis = 0.025`, so `t* = 30` is `t = 0.75` — the very start of the transient.
In this regime the approach is inertial, set by rim growth in capillary time
`sqrt(rho h0^3/gamma)`, and `t*` is not a meaningful target. Report low-`Oh`
results against capillary time.

## Cross-method check

The planar FEM–Basilisk comparison at `Oh < 0.1` is closed. The write-up
is `docs/validation-inertial/`. The oscillation question below is about
mechanism, not about whether `u*` approaches 1.

## Open question this can settle

Whether the planar rim approaches `V_TC` **monotonically or with oscillation**
is not resolved in the literature. Pierson et al. (PRF 5, 073602, 2020)
establish ringing about `V_TC` for the axisymmetric filament at `Oh <~ 0.1`, via
a mass-spring mechanism (growing blob inertia against a capillary restoring
force); nobody states it for the planar case.

Our `L = 12` scoping run showed exactly that signature — `U/V_TC` peaking near
0.8304 at `t ~ 1.7`, dipping to 0.8289, then recovering. That was initially
misread as a numerical defect. A converged low-`Oh` run on a domain large enough
that the shed waves do not reflect is the clean test.
