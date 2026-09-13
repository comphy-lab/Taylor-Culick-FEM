# Viscous (high `Oh`) planar retraction

## Claim

> In the viscosity-dominated regime the co-moving formulation reproduces the
> published edge velocity `u*(t*)`, and the near-tip thickness profile matches
> the predicted **quasi-steady exponential shape with a linearly growing
> amplitude**.

## Primary reference

**Ahsan, Brandão, Davidovitch & Stone**, *Retraction dynamics of a highly
viscous liquid sheet*, arXiv:2507.04603 / Phys. Rev. Fluids
(`../literature/ahsan_brandao_davidovitch_stone_arXiv2507.04603.pdf`). This is
the most rigorous treatment of the high-`Oh` limit and the one that makes the
structure of the solution explicit. Savva & Bush (2009) supply the digitised
`u*(t*)` curves at `Oh = 1, 5, 10, 100/1000`.

## The structure worth testing

In the tip-comoving frame, at long time,

```
H(X,T) = T * H1(Xhat) + H0(Xhat) + O(1/T),     H1(Xhat) = exp(-U0 * Xhat)
```

with `U0 = V_TC`. So the near-tip profile has a **fixed shape** — an exponential
whose decay length is the Stokes length `mu/(rho u_c) = Oh*h0` — riding on an
amplitude that grows **linearly** in time.

This resolves the apparent contradiction in Savva & Bush between a fixed Stokes
length and a linearly growing maximum thickness. Their tip equation is
`u_t = 4 u_xx`, a **diffusion** equation, so the disturbed region broadens like
`sqrt(t)`. A fixed decay length plus diffusive broadening plus mass conservation
gives linear thickness growth. Self-similar, not steady — and not a
contradiction but a consequence.

The sharpest practical consequence: the velocity correction to `V_TC` decays as
`O(1/T^2)`, far faster than the amplitude saturates. **The speed converges long
before the field does.** That is why measuring `U(t)` is tractable even though
the field never reaches a steady state.

## No exact solution exists

Ahsan et al. state that the reduced problem is "mathematically equivalent to a
**Stefan problem** with time-dependent boundary conditions" and that they "have
been unable to find an exact solution". This is a citable structural result, not
an absence of evidence, and it is why a *stationary* solve in the co-moving
frame is ill-posed (see `../README.md`).

## Why this is the sharper test of our code

Savva & Bush solve a 1D long-wavelength (Trouton) lubrication model. That
approximation is best where the interface is flat — the high-`Oh` regime, where
no rim forms at all — and worst where it is steep, i.e. the low-`Oh` bulbous
rim. So at high `Oh` a disagreement is most likely **our** error; at low `Oh` it
may equally be their approximation. Weight the regimes accordingly when reading
the comparison.

## Reference data

`../data/savva_bush_2009_fig5_Oh{1,5,100}.csv`,
`../data/savva_bush_2009_fig6_Oh{1,10,100}.csv`
(`Oh100` is the collapsed "Oh > 100" / "Oh = 100, 1000" curve).

Their high-`Oh` curves become indistinguishable from one another, so
`Oh = 100/1000` is a single digitised curve — do not read agreement with it
as resolving `Oh` dependence in that limit.
