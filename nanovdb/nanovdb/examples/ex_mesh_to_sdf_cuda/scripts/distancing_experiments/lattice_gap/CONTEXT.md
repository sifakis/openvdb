# Context: the 2D lattice certification-gap problem

## Origin

Fork of OpenVDB. We are redesigning how a narrow-band signed distance field decides
inside/outside ("signing"). The new construction defines the exterior as a **union of balls**:

    O = union over certified points Vi of B(Vi, di),   di = UDF(Vi) = dist(Vi, Sigma)

Each ball is surface-free by definition of `di`, so `O` never touches the surface `Sigma`.
Certification grows by a **pairwise enrichment rule**: given an already-certified `V1` with
`d1 = UDF(V1)`, another point `V2` with `d2 = UDF(V2)` is also certifiably exterior if

    d1 + d2 > dist(V1, V2)      (strict)

(Proof: 1-Lipschitzness of dist(.,Sigma) makes the whole segment [V1,V2] surface-free, so both
endpoints lie in the same component of Sigma's complement. Equivalently: the two balls overlap.)

A voxel is a **barrier** voxel if `UDF <= r_d`, where `r_d` is the lattice covering radius
(`sqrt(2)/2` in 2D, `sqrt(3)/2` in 3D) -- the surface might pass through such a cell, so its sign
cannot be settled locally. Non-barrier voxels in one connected component share a sign, so any
non-barrier in-band lattice point on the exterior side is certified by plain connected components.

**The question this note is about:** how close to `Sigma` is a lattice point still guaranteed to
be certified? Equivalently: what is the largest possible depth of an *uncertified* exterior
lattice point? Call it `eps*`.

## The reduction (2D)

Work in 2D (agreed staging ground; the real pipeline is 3D). Lattice `Z^2`, unit spacing.
`Sigma` is a straight line with unit normal `n` pointing into the exterior.

Put the **target** lattice point `v1` at the origin -- WLOG, since `Z^2` is invariant under
integer translation. Let `eps = d1 = dist(v1, Sigma) > 0` be its depth. A candidate **witness** is
another lattice point at integer vector `w`; its depth is

    d0 = eps + w.n

Substituting into the pairwise rule `d1 + d0 > |w|`:

    eps + (eps + w.n) > |w|
    2 eps > |w| - w.n = |w|(1 - cos theta_w) = 2 |w| sin^2(theta_w / 2)

where `theta_w = angle(n, w)`. So, exactly and unconditionally:

    ### v1 is certified by w  <=>  eps > |w| * sin^2(theta_w / 2)   =:  g(w, n)

(Equivalently `eps > d0 * tan^2(theta_w/2)`, valid for `theta_w < 90 deg`.)

Geometric reading: `theta_w` is literally the angle `∠P0 v0 v1` at the witness, between its
perpendicular drop to `Sigma` and the segment to the target. Verified numerically: 0 mismatches
over 400k random configurations.

Note `g` is **length-weighted**: a long vector that is well aligned with `n` competes against a
short vector that is poorly aligned. This is the crux of the whole problem.

## Witness admissibility

`w` is usable only if the witness point is itself certified. Sufficient (and what we use):

  * **non-barrier**: `d0 > sqrt(2)/2 = 0.7071067811865476`
  * **in-band**:     `d0 <= W`, the narrow-band half-width in voxel units

Witness `w` certifies targets with `eps > g(w,n)`; at that threshold its depth is
`d0 = g(w,n) + w.n = (|w| + w.n)/2`. So `w` is *usable at its own threshold* iff

    sqrt(2)/2 < (|w| + w.n)/2 <= W

This self-consistent test is the correct admissibility criterion (for smaller `eps` the witness
would be in band but would not certify; for larger `eps` its depth only grows further).

## The quantity to determine

    eps*(n, W) = min over admissible w of g(w, n)
    eps*(W)    = max over unit n of eps*(n, W)

`eps*(W)` is the guaranteed capture depth: every exterior lattice point deeper than `eps*(W)` is
certified, and some configuration leaves one at `eps*(W)` uncertified.

## Numerical facts already established (verify these, do not assume them)

At `W = 3`:

  * `eps* = 0.019202630763796` at `alpha* = 15.930625116 deg` (angle of `n` from the +x axis).
  * Closed form conjecture, matches to 1e-17:
        `eps* = (sqrt5 - 2) / (2 * (1 + sqrt5 + sqrt(4 + 2*sqrt5)))`
    equivalently `eps* = sin^2(u)` with `tan u = 5^(1/4) sin(phi/2) / (1 + 5^(1/4) cos(phi/2))`,
    `phi = arctan(1/2)`, and `alpha* = 2u`.
  * Two **tied** minimizers at `alpha*`:
        w = (1,0),  |w| = 1,       theta = 15.93063 deg, depth 0.98080, g = 0.019202630763796
        w = (2,1),  |w| = sqrt5,   theta = 10.63443 deg, depth 2.21687, g = 0.019202630763796
    i.e. `alpha*` is exactly the crossover where a short-but-misaligned vector and a
    long-but-well-aligned vector cost the same.
  * `(3,1)` would cost only `0.00151010` there, 12x better -- but its depth at that threshold is
    `3.16077 > W = 3`, so it is **out of band**. The constant at `W=3` is set by the band cutoff.
  * `eps*(W)` is a step function. Coarse scan (0.01 steps in W) of breakpoints:
        W >= 1.30   eps* = 0.146446609   alpha = 45.000 deg
        W >= 1.42   eps* = 0.044910139   alpha = 65.530 deg
        W >= 2.24   eps* = 0.019202631   alpha = 74.069 deg  (= 90 - 15.931, mirror of alpha*)
        W >= 3.17   eps* = 0.011330789   alpha = 34.729 deg  (crossover of (2,1) and (1,1))
        W >= 3.61   eps* = 0.010574711   alpha = 78.195 deg
        W >= 4.13   eps* = 0.006724317   alpha =  9.407 deg
        W >= 5.10   eps* = 0.004673711   alpha =  7.840 deg
    So the `0.0192026` plateau appears to hold for `W` in roughly `[2.24, 3.16077)`; the upper
    endpoint should be exactly where `(3,1)` enters the band at `alpha*`.

## Where the argument came from (the human's route, for intuition)

Restrict candidate witnesses to `[-2,2]^2`. Drop non-primitive vectors (two lattice points share a
direction iff one is an integer multiple of the other; the nearer is always cheaper since
`g(kw,n) = k*g(w,n)`). That leaves **16 primitive directions**:

    (1,0) 0deg, (2,1) 26.565051deg, (1,1) 45deg, (1,2) 63.434949deg, (0,1) 90deg,
    and their images under the 8-fold dihedral symmetry of the lattice.

Angular gaps alternate between `arctan(1/2) = 26.565051 deg` and `45 - arctan(1/2) = 18.434949
deg`; they sum to 45 deg, so each octant is split by its single "knight" direction. Largest gap is
`26.565051 deg`, so **any** direction `n` is within `13.2825256 deg` of one of the 16.

That gives easy but loose bounds:
  * pinning `d0 <= W = 3`:              `eps* <= 3 tan^2(13.2825/2) = 0.040671`
  * pinning `|w| <= sqrt5`:             `eps* <= sqrt5 sin^2(13.2825/2) = 0.029909`
  * selecting the *nearest-angle* witness (exhaustively swept): `0.029908`, worst `n` at 13.2826 deg
  * selecting the *cheapest* witness instead:                   `0.019203`, worst `n` at 15.931 deg

The last is the real answer, and the gap between 0.0299 and 0.0192 is exactly the length-weighting.

## Ground rules

  * Environment: Python 3, **no numpy** (not installed; `pip`/`ensurepip` unavailable). Pure
    Python is fine at these problem sizes. Use `fractions`/`decimal`/`math` freely.
  * Scratch dir for all files: `/tmp/dist/`. Do not write into the repo.
  * Prefer exact/symbolic reasoning where possible; when numerics are used, state precision.
  * Report honestly. If a claim fails to verify, say so with the counterexample. Do not paper over.
